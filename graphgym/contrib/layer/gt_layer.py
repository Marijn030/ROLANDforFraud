import math
import torch
import torch_sparse
from torch_scatter import scatter_max
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import graphgym.register as register
from graphgym.config import cfg
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import Linear

def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'prelu':
        return nn.PReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'elu':
        return nn.ELU()
    if name == 'leakyrelu':
        return nn.LeakyReLU()
    if name in register.act_dict:
        act = register.act_dict[name]
        return act() if isinstance(act, type) else act
    raise KeyError(f'Unknown activation: {name}')

class DummyRuntimeStats:
    def start_region(self, *args, **kwargs):
        pass

    def end_region(self, *args, **kwargs):
        pass

runtime_stats_cuda = DummyRuntimeStats()


class GTLayer(nn.Module):
    """
    Homogeneous FraudGT-style sparse graph transformer layer
    adapted for ROLAND / DeepSNAP batches.

    Expects:
      batch.node_feature : [N, dim_h]
      batch.edge_feature : [E, dim_h] (optional but expected for EPTN)
      batch.edge_index   : [2, E]
    """
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_out,
        metadata,
        local_gnn_type,
        global_model_type,
        index,
        num_heads=1,
        layer_norm=False,
        batch_norm=False,
        return_attention=False,
        **kwargs,
    ):
        super().__init__()

        if global_model_type != 'SparseNodeTransformer':
            raise NotImplementedError(
                f"This EPTN/ROLAND version only supports SparseNodeTransformer, got {global_model_type}."
            )

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.index = index
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = get_activation(cfg.gt.act)
        self.metadata = metadata
        self.return_attention = return_attention
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.kHop = cfg.gt.hops

        if dim_h % num_heads != 0:
            raise ValueError(f"dim_h={dim_h} must be divisible by num_heads={num_heads}")

        self.bias = Parameter(torch.Tensor(self.kHop))
        self.attn_bi = Parameter(torch.empty(self.num_heads, self.kHop))

        node_type = metadata[0][0]
        edge_type = "__".join(metadata[1][0])

        self.skip_local = torch.nn.ParameterDict()
        self.skip_global = torch.nn.ParameterDict()
        self.skip_local[node_type] = Parameter(torch.Tensor(1))
        self.skip_global[node_type] = Parameter(torch.Tensor(1))

        # SparseNodeTransformer projections
        self.k_lin = torch.nn.ModuleDict({node_type: Linear(dim_in, dim_h)})
        self.q_lin = torch.nn.ModuleDict({node_type: Linear(dim_in, dim_h)})
        self.v_lin = torch.nn.ModuleDict({node_type: Linear(dim_in, dim_h)})
        self.o_lin = torch.nn.ModuleDict({node_type: Linear(dim_h, dim_out)})

        self.e_lin = torch.nn.ModuleDict({edge_type: Linear(dim_in, dim_h)})
        self.g_lin = torch.nn.ModuleDict({edge_type: Linear(dim_h, dim_out)})
        self.oe_lin = torch.nn.ModuleDict({edge_type: Linear(dim_h, dim_out)})

        H, D = self.num_heads, self.dim_h // self.num_heads
        if cfg.gt.edge_weight:
            self.edge_weights = nn.Parameter(torch.Tensor(1, H, D, D))
            self.msg_weights = nn.Parameter(torch.Tensor(1, H, D, D))
            nn.init.xavier_uniform_(self.edge_weights)
            nn.init.xavier_uniform_(self.msg_weights)

        self.norm1_local = torch.nn.ModuleDict()
        self.norm1_global = torch.nn.ModuleDict()
        self.norm2_ffn = torch.nn.ModuleDict()
        self.project = torch.nn.ModuleDict()

        self.project[node_type] = Linear(dim_h * 2, dim_h)

        if self.layer_norm:
            self.norm1_local[node_type] = nn.LayerNorm(dim_h)
            self.norm1_global[node_type] = nn.LayerNorm(dim_h)
            self.norm2_ffn[node_type] = nn.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm1_local[node_type] = nn.BatchNorm1d(dim_h)
            self.norm1_global[node_type] = nn.BatchNorm1d(dim_h)
            self.norm2_ffn[node_type] = nn.BatchNorm1d(dim_h)

        self.norm1_edge_local = torch.nn.ModuleDict()
        self.norm1_edge_global = torch.nn.ModuleDict()
        self.norm2_edge_ffn = torch.nn.ModuleDict()

        if self.layer_norm:
            self.norm1_edge_local[edge_type] = nn.LayerNorm(dim_h)
            self.norm1_edge_global[edge_type] = nn.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm1_edge_local[edge_type] = nn.BatchNorm1d(dim_h)
            self.norm1_edge_global[edge_type] = nn.BatchNorm1d(dim_h)

        self.dropout_local = nn.Dropout(cfg.gnn.dropout)
        self.dropout_global = nn.Dropout(cfg.gt.dropout)
        self.dropout_attn = nn.Dropout(cfg.gt.attn_dropout)

        if cfg.gt.ffn == 'Single':
            self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        elif cfg.gt.ffn == 'Type':
            self.ff_linear1_type = torch.nn.ModuleDict({node_type: nn.Linear(dim_h, dim_h * 2)})
            self.ff_linear2_type = torch.nn.ModuleDict({node_type: nn.Linear(dim_h * 2, dim_h)})
            self.ff_linear1_edge_type = torch.nn.ModuleDict({edge_type: nn.Linear(dim_h, dim_h * 2)})
            self.ff_linear2_edge_type = torch.nn.ModuleDict({edge_type: nn.Linear(dim_h * 2, dim_h)})
        elif cfg.gt.ffn != 'none':
            raise ValueError(f"Invalid GT FFN option {cfg.gt.ffn}")

        self.ff_dropout1 = nn.Dropout(cfg.gt.dropout)
        self.ff_dropout2 = nn.Dropout(cfg.gt.dropout)

        self.node_type = node_type
        self.edge_type_tuple = metadata[1][0]
        self.edge_type_name = edge_type

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.attn_bi)
        nn.init.zeros_(self.bias)

    def forward(self, batch):
        x = batch.node_feature
        edge_index = batch.edge_index
        has_edge_attr = hasattr(batch, 'edge_feature') and batch.edge_feature is not None
        edge_feat = batch.edge_feature if has_edge_attr else None

        x_in = x
        edge_feat_in = edge_feat

        runtime_stats_cuda.start_region("gt-layer")

        # Pre-norm
        if self.layer_norm or self.batch_norm:
            x = self.norm1_global[self.node_type](x)
            if has_edge_attr:
                edge_feat = self.norm1_edge_global[self.edge_type_name](edge_feat)

        H, D = self.num_heads, self.dim_h // self.num_heads

        q = self.q_lin[self.node_type](x).view(-1, H, D).transpose(0, 1)   # [H, N, D]
        k = self.k_lin[self.node_type](x).view(-1, H, D).transpose(0, 1)   # [H, N, D]
        v = self.v_lin[self.node_type](x).view(-1, H, D).transpose(0, 1)   # [H, N, D]

        src_nodes, dst_nodes = edge_index
        num_edges = edge_index.shape[1]
        num_nodes = x.shape[0]

        edge_attr = None
        edge_gate = None
        if has_edge_attr:
            edge_attr = self.e_lin[self.edge_type_name](edge_feat).view(-1, H, D).transpose(0, 1)   # [H, E, D]
            edge_gate = self.g_lin[self.edge_type_name](edge_feat).view(-1, H, D).transpose(0, 1)   # [H, E, D]

        # Sparse edge attention
        if cfg.gt.attn_mask in ['Edge', 'kHop']:
            if cfg.gt.attn_mask == 'kHop':
                with torch.no_grad():
                    edge_index_list = [edge_index]
                    edge_index_k = edge_index
                    if self.kHop > 1:
                        ones = torch.ones(edge_index.shape[1], device=edge_index.device)
                        for _ in range(1, self.kHop):
                            edge_index_k, _ = torch_sparse.spspmm(
                                edge_index_k,
                                torch.ones(edge_index_k.shape[1], device=edge_index.device),
                                edge_index,
                                ones,
                                num_nodes,
                                num_nodes,
                                num_nodes,
                                True,
                            )
                            edge_index_list.append(edge_index_k)
                    else:
                        edge_index_k = edge_index

                attn_mask = torch.full(
                    (num_nodes, num_nodes),
                    -1e9,
                    dtype=torch.float32,
                    device=edge_index.device,
                )
                for idx, eidx in enumerate(reversed(edge_index_list)):
                    attn_mask[eidx[1, :], eidx[0, :]] = self.bias[idx]

                src_nodes, dst_nodes = edge_index_k
                num_edges = edge_index_k.shape[1]

                if has_edge_attr:
                    # For kHop>1, raw edge features only exist for observed edges.
                    # This EPTN version is intended for hops=1.
                    if self.kHop != 1:
                        raise NotImplementedError("Edge-aware kHop > 1 is not supported in this homogeneous adaptation.")
            else:
                attn_mask = None

            edge_q = q[:, dst_nodes, :]   # [H, E, D]
            edge_k = k[:, src_nodes, :]   # [H, E, D]
            edge_v = v[:, src_nodes, :]   # [H, E, D]

            if hasattr(self, 'edge_weights'):
                edge_weight = self.edge_weights[0].unsqueeze(1).expand(H, num_edges, D, D)
                edge_k = torch.matmul(edge_weight, edge_k.unsqueeze(-1)).squeeze(-1)

            edge_scores = edge_q * edge_k

            if has_edge_attr:
                edge_scores = edge_scores + edge_attr
                edge_v = edge_v * torch.sigmoid(edge_gate)
                edge_attr_out = edge_scores
            else:
                edge_attr_out = None

            edge_scores = torch.sum(edge_scores, dim=-1) / math.sqrt(D)  # [H, E]
            edge_scores = torch.clamp(edge_scores, min=-5, max=5)

            if cfg.gt.attn_mask == 'kHop' and attn_mask is not None:
                edge_scores = edge_scores + attn_mask[dst_nodes, src_nodes]

            expanded_dst_nodes = dst_nodes.repeat(H, 1)
            max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=num_nodes)
            max_scores = max_scores.gather(1, expanded_dst_nodes)

            exp_scores = torch.exp(edge_scores - max_scores)
            sum_exp_scores = torch.zeros((H, num_nodes), device=edge_scores.device)
            sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)

            edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
            edge_scores = self.dropout_attn(edge_scores.unsqueeze(-1))  # [H, E, 1]

            out = torch.zeros((H, num_nodes, D), device=q.device)
            out.scatter_add_(
                1,
                dst_nodes.unsqueeze(-1).expand((H, num_edges, D)),
                edge_scores * edge_v,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout_attn(scores)
            out = torch.matmul(scores, v)
            edge_attr_out = edge_attr

        out = out.transpose(0, 1).contiguous().view(-1, H * D)   # [N, dim_h]
        x_attn = self.o_lin[self.node_type](out)
        x_attn = self.dropout_global(x_attn)

        if has_edge_attr and edge_attr_out is not None:
            edge_attr_out = edge_attr_out.transpose(0, 1).contiguous().view(-1, H * D)
            edge_feat = self.oe_lin[self.edge_type_name](edge_attr_out)

        # Residual
        if cfg.gt.residual == 'Fixed':
            x = x_attn + x_in
            if has_edge_attr and edge_feat is not None and edge_feat_in is not None:
                edge_feat = edge_feat + edge_feat_in
        elif cfg.gt.residual == 'Learn':
            alpha = self.skip_global[self.node_type].sigmoid()
            x = alpha * x_attn + (1 - alpha) * x_in
        elif cfg.gt.residual == 'none':
            x = x_attn
        else:
            raise ValueError(f"Invalid attention residual option {cfg.gt.residual}")

        # FFN
        if cfg.gt.ffn != 'none':
            if self.layer_norm or self.batch_norm:
                x = self.norm2_ffn[self.node_type](x)

            if cfg.gt.ffn == 'Type':
                x = x + self._ff_block_type(x, self.node_type)
                if has_edge_attr and edge_feat is not None:
                    edge_feat = edge_feat + self._ff_block_edge_type(edge_feat, self.edge_type_tuple)
            elif cfg.gt.ffn == 'Single':
                x = x + self._ff_block(x)

        if cfg.gt.residual == 'Concat':
            x = torch.cat((x_in, x), dim=1)

        runtime_stats_cuda.end_region("gt-layer")

        batch.node_feature = x
        if has_edge_attr and edge_feat is not None:
            batch.edge_feature = edge_feat

        if self.return_attention:
            return batch, None
        return batch

    def _ff_block_type(self, x, node_type):
        x = self.ff_dropout1(self.activation(self.ff_linear1_type[node_type](x)))
        return self.ff_dropout2(self.ff_linear2_type[node_type](x))

    def _ff_block(self, x):
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def _ff_block_edge_type(self, x, edge_type):
        edge_type = "__".join(edge_type)
        x = self.ff_dropout1(self.activation(self.ff_linear1_edge_type[edge_type](x)))
        return self.ff_dropout2(self.ff_linear2_edge_type[edge_type](x))
