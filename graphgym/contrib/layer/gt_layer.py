import math
import torch
import torch_sparse
from torch_scatter import scatter_max
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import graphgym.register as register
from graphgym.config import cfg
from torch_geometric.data import HeteroData
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
        return register.act_dict[name]
    raise KeyError(f'Unknown activation: {name}')

class DummyRuntimeStats:
    def start_region(self, *args, **kwargs):
        pass

    def end_region(self, *args, **kwargs):
        pass

runtime_stats_cuda = DummyRuntimeStats()

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
        return register.act_dict[name]
    raise KeyError(f'Unknown activation: {name}')

class GTLayer(nn.Module):
    r"""Graph Transformer layer

    """
    def __init__(self, dim_in, dim_h, dim_out, metadata, local_gnn_type, global_model_type, index, num_heads=1,
                 layer_norm=False, batch_norm=False, return_attention=False, **kwargs):
        super(GTLayer, self).__init__()

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
        self.bias = Parameter(torch.Tensor(self.kHop))
        self.attn_bi = Parameter(torch.empty(self.num_heads, self.kHop))

        # Residual connection
        self.skip_local = torch.nn.ParameterDict()
        self.skip_global = torch.nn.ParameterDict()
        for node_type in metadata[0]:
            self.skip_local[node_type] = Parameter(torch.Tensor(1))
            self.skip_global[node_type] = Parameter(torch.Tensor(1))


        # Global Attention
        if global_model_type == 'None':
            self.attn = None
        elif global_model_type == 'TorchTransformer':
            self.attn = torch.nn.MultiheadAttention(
                        dim_h, num_heads, dropout=cfg.gt.attn_dropout, batch_first=True)
            # self.attn = torch.nn.ModuleDict()
            # for edge_type in metadata[1]:
            #     edge_type = '__'.join(edge_type)
            #     self.attn[edge_type] = torch.nn.MultiheadAttention(
            #             dim_h, num_heads, dropout=cfg.gt.attn_dropout, batch_first=True)
        elif global_model_type == 'SparseNodeTransformer':
            self.k_lin = torch.nn.ModuleDict()
            self.q_lin = torch.nn.ModuleDict()
            self.v_lin = torch.nn.ModuleDict()
            self.e_lin = torch.nn.ModuleDict()
            self.g_lin = torch.nn.ModuleDict()
            self.oe_lin = torch.nn.ModuleDict()
            self.o_lin = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                # Different node type have a different projection matrix
                self.k_lin[node_type] = Linear(dim_in, dim_h)
                self.q_lin[node_type] = Linear(dim_in, dim_h)
                self.v_lin[node_type] = Linear(dim_in, dim_h)
                self.o_lin[node_type] = Linear(dim_h, dim_out)
            for edge_type in metadata[1]:
                edge_type = '__'.join(edge_type)
                self.e_lin[edge_type] = Linear(dim_in, dim_h)
                self.g_lin[edge_type] = Linear(dim_h, dim_out)
                self.oe_lin[edge_type] = Linear(dim_h, dim_out)
            H, D = self.num_heads, self.dim_h // self.num_heads
            if cfg.gt.edge_weight:
                self.edge_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                self.msg_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                nn.init.xavier_uniform_(self.edge_weights)
                nn.init.xavier_uniform_(self.msg_weights)
        elif global_model_type == 'SparseEdgeTransformer':
            self.k_lin = torch.nn.ModuleDict()
            self.q_lin = torch.nn.ModuleDict()
            self.v_lin = torch.nn.ModuleDict()
            self.e_lin = torch.nn.ModuleDict()
            self.g_lin = torch.nn.ModuleDict()
            self.oe_lin = torch.nn.ModuleDict()
            self.o_lin = torch.nn.ModuleDict()
            for edge_type in metadata[1]:
                edge_type = '__'.join(edge_type)
                # Different edge type have a different projection matrix
                self.k_lin[edge_type] = Linear(dim_in, dim_h)
                self.q_lin[edge_type] = Linear(dim_in, dim_h)
                self.v_lin[edge_type] = Linear(dim_in, dim_h)
                self.e_lin[edge_type] = Linear(dim_in, dim_h)
                self.g_lin[edge_type] = Linear(dim_h, dim_out)
                self.oe_lin[edge_type] = Linear(dim_h, dim_out)
                self.o_lin[edge_type] = Linear(dim_h, dim_out)
            H, D = self.num_heads, self.dim_h // self.num_heads
            if cfg.gt.edge_weight:
                self.edge_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                self.msg_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                nn.init.xavier_uniform_(self.edge_weights)
                nn.init.xavier_uniform_(self.msg_weights)

        self.norm1_local = torch.nn.ModuleDict()
        self.norm1_global = torch.nn.ModuleDict()
        self.norm2_ffn = torch.nn.ModuleDict()
        self.project = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.project[node_type] = Linear(dim_h * 2, dim_h)
            if self.layer_norm:
                self.norm1_local[node_type] = nn.LayerNorm(dim_h)
                self.norm1_global[node_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm1_local[node_type] = nn.BatchNorm1d(dim_h)
                self.norm1_global[node_type] = nn.BatchNorm1d(dim_h)
        self.norm1_edge_local = torch.nn.ModuleDict()
        self.norm1_edge_global = torch.nn.ModuleDict()
        self.norm2_edge_ffn = torch.nn.ModuleDict()
        for edge_type in metadata[1]:
            edge_type = "__".join(edge_type)
            if self.layer_norm:
                self.norm1_edge_local[edge_type] = nn.LayerNorm(dim_h)
                self.norm1_edge_global[edge_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm1_edge_local[edge_type] = nn.BatchNorm1d(dim_h)
                self.norm1_edge_global[edge_type] = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(cfg.gnn.dropout)
        self.dropout_global = nn.Dropout(cfg.gt.dropout)
        self.dropout_attn = nn.Dropout(cfg.gt.attn_dropout)

        # if cfg.gt.residual == 'Concat':
        #     dim_h *= 2
        for node_type in metadata[0]:
            # Different node type have a different projection matrix
            if self.layer_norm:
                self.norm2_ffn[node_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm2_ffn[node_type] = nn.BatchNorm1d(dim_h)
        
        # Feed Forward block.
        if cfg.gt.ffn == 'Single':
            self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        elif cfg.gt.ffn == 'Type':
            self.ff_linear1_type = torch.nn.ModuleDict()
            self.ff_linear2_type = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                self.ff_linear1_type[node_type] = nn.Linear(dim_h, dim_h * 2)
                self.ff_linear2_type[node_type] = nn.Linear(dim_h * 2, dim_h)
            self.ff_linear1_edge_type = torch.nn.ModuleDict()
            self.ff_linear2_edge_type = torch.nn.ModuleDict()
            for edge_type in metadata[1]:
                edge_type = "__".join(edge_type)
                self.ff_linear1_edge_type[edge_type] = nn.Linear(dim_h, dim_h * 2)
                self.ff_linear2_edge_type[edge_type] = nn.Linear(dim_h * 2, dim_h)
        
        self.ff_dropout1 = nn.Dropout(cfg.gt.dropout)
        self.ff_dropout2 = nn.Dropout(cfg.gt.dropout)
        self.reset_parameters()


    def reset_parameters(self):
        pass
        zeros(self.attn_bi)
        # ones(self.skip)


    def forward(self, batch):
        has_edge_attr = False
        if isinstance(batch, HeteroData):
            node_types = batch.node_types
            edge_types = batch.edge_types
            h_dict, edge_index_dict = batch.collect('x'), batch.collect('edge_index')
            if sum(batch.num_edge_features.values()):
                edge_attr_dict = batch.collect('edge_attr')
                has_edge_attr = True
        else:
            node_types = ['node_type']
            edge_types = [('node_type', 'edge_type', 'node_type')]
            h_dict = {'node_type': batch.node_feature}
            edge_index_dict = {edge_types[0]: batch.edge_index}
            has_edge_attr = hasattr(batch, 'edge_feature') and batch.edge_feature is not None
            if has_edge_attr:
                edge_attr_dict = {edge_types[0]: batch.edge_feature}
        h_in_dict = h_dict
        if has_edge_attr:
            edge_attr_in_dict = edge_attr_dict.copy()

        h_out_dict_list = {node_type: [] for node_type in h_dict}
        runtime_stats_cuda.start_region("gt-layer")

        if self.global_model_type != 'None':
            # Pre-normalization
            if self.layer_norm or self.batch_norm:
                h_dict = {
                    node_type: self.norm1_global[node_type](h_dict[node_type])
                    for node_type in node_types
                }
                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: self.norm1_edge_global["__".join(edge_type)](edge_attr_dict[edge_type])
                        for edge_type in edge_types
                    }
            

            h_attn_dict_list = {node_type: [] for node_type in h_dict}
            if self.global_model_type == 'TorchTransformer':
                D = self.dim_h

                homo_data = batch.to_homogeneous()
                h = homo_data.x
                edge_index = homo_data.edge_index
                node_type_tensor = homo_data.node_type
                edge_type_tensor = homo_data.edge_type
                
                L = h.shape[0]
                S = h.shape[0]
                q = h.view(1, -1, D)
                k = h.view(1, -1, D)
                v = h.view(1, -1, D)

                if cfg.gt.attn_mask in ['Edge', 'kHop']:
                    attn_mask = torch.full((L, S), -1e9, dtype=torch.float32, device=edge_index.device)
                    if cfg.gt.attn_mask == 'kHop':
                        with torch.no_grad():
                            ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                            edge_index_list = [edge_index]
                            edge_index_k = edge_index
                            for i in range(1, self.kHop):
                                # print(edge_index_k.shape, int(edge_index_k.max()), L)
                                edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                                                                    edge_index, ones, 
                                                                    L, L, L, True)
                                edge_index_list.append(edge_index_k)
                        
                        for idx, edge_index in enumerate(reversed(edge_index_list)):
                            attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                    else:
                        # Avoid the nan from attention mask
                        attn_mask[edge_index[1, :], edge_index[0, :]] = 1
                
                elif cfg.gt.attn_mask == 'Bias':
                    attn_mask = batch.attn_bi[self.index, :, :, :]
                else:
                    attn_mask = None

                h, A = self.attn(q, k, v,
                            attn_mask=attn_mask,
                            need_weights=True)
                            # average_attn_weights=False)

                # attn_weights = A.detach().cpu()
                h = h.view(1, -1, D)
                for idx, node_type in enumerate(node_types):
                    out_type = h[:, node_type_tensor == idx, :]
                    h_attn_dict_list[node_type].append(out_type.squeeze())

            elif self.global_model_type == 'SparseNodeTransformer':
                # Test if Signed attention is beneficial
                # st = time.time()
                H, D = self.num_heads, self.dim_h // self.num_heads
                if isinstance(batch, HeteroData):
                    homo_data = batch.to_homogeneous()
                    h = homo_data.x
                    edge_index = homo_data.edge_index
                    node_type_tensor = homo_data.node_type
                    edge_type_tensor = homo_data.edge_type
                else:
                    h = batch.node_feature
                    edge_index = batch.edge_index
                    node_type_tensor = torch.zeros(
                        h.size(0), dtype=torch.long, device=h.device
                    )
                    edge_type_tensor = torch.zeros(
                        edge_index.size(1), dtype=torch.long, device=edge_index.device
                    )
                q = torch.empty((h.size(0), self.dim_h), device=h.device)
                k = torch.empty((h.size(0), self.dim_h), device=h.device)
                v = torch.empty((h.size(0), self.dim_h), device=h.device)
                edge_attr = torch.empty((edge_index.size(1), self.dim_h), device=h.device)
                edge_gate = torch.empty((edge_index.size(1), self.dim_h), device=h.device)
                for idx, node_type in enumerate(node_types):
                    mask = node_type_tensor == idx
                    q[mask] = self.q_lin[node_type](h_dict[node_type])
                    k[mask] = self.k_lin[node_type](h_dict[node_type])
                    v[mask] = self.v_lin[node_type](h_dict[node_type])
                for idx, edge_type_tuple in enumerate(edge_types):
                    edge_type = '__'.join(edge_type_tuple)
                    mask = edge_type_tensor == idx
                    edge_attr[mask] = self.e_lin[edge_type](edge_attr_dict[edge_type_tuple])
                    edge_gate[mask] = self.g_lin[edge_type](edge_attr_dict[edge_type_tuple])
                src_nodes, dst_nodes = edge_index
                num_edges = edge_index.shape[1]
                L = h.shape[0]
                S = h.shape[0]

                if has_edge_attr:
                    # src_nodes, dst_nodes = edge_index
                    # edge_attr = edge_attr_dict[edge_type_tuple]
                    # edge_attr = self.e_lin[edge_type](edge_attr).view(-1, H, D)
                    edge_attr = edge_attr.view(-1, H, D)
                    # edge_attr = self.e_lin[edge_type](torch.cat((h_dict[src][src_nodes], h_dict[dst][dst_nodes], edge_attr), dim=-1)).view(-1, H, D)
                    edge_attr = edge_attr.transpose(0,1) # (h, sl, d_model)

                    # edge_gate = edge_attr_dict[edge_type_tuple]
                    # edge_gate = self.g_lin[edge_type](edge_gate).view(-1, H, D)
                    edge_gate = edge_gate.view(-1, H, D)
                    # edge_gate = self.g_lin[edge_type](torch.cat((h_dict[src][src_nodes], h_dict[dst][dst_nodes], edge_gate), dim=-1)).view(-1, H, D)
                    edge_gate = edge_gate.transpose(0,1) # (h, sl, d_model)

                q = q.view(-1, H, D)
                k = k.view(-1, H, D)
                v = v.view(-1, H, D)

                # transpose to get dimensions h * sl * d_model
                q = q.transpose(0,1)
                k = k.transpose(0,1)
                v = v.transpose(0,1)

                if cfg.gt.attn_mask in ['Edge', 'kHop']:
                    if cfg.gt.attn_mask in ['kHop']:
                        with torch.no_grad():
                            edge_index_list = [edge_index]
                            edge_index_k = torch.cat(edge_index_list, dim=1)

                            # ones = torch.ones(edge_index.shape[1], device=edge_index.device)
                            # edge_index_list = [edge_index]
                            # edge_index_k = edge_index
                            # for i in range(1, self.kHop):
                            #     # print(edge_index_k.shape, int(edge_index_k.max()), L)
                            #     edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                            #                                         edge_index, ones, 
                            #                                         L, L, L, True)
                            #     edge_index_list.append(edge_index_k)
                        
                        attn_mask = torch.full((L, L), -1e9, dtype=torch.float32, device=edge_index.device)
                        for idx, edge_index in enumerate(reversed(edge_index_list)):
                            attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                        src_nodes, dst_nodes = edge_index_k
                        num_edges = edge_index_k.shape[1]
                    else:
                        src_nodes, dst_nodes = edge_index
                        num_edges = edge_index.shape[1]
                    # Compute query and key for each edge
                    edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # num_heads * num_edges * d_k
                    edge_k = k[:, src_nodes, :]  # Keys for source nodes
                    edge_v = v[:, src_nodes, :]

                    if hasattr(self, 'edge_weights'):
                        edge_weight = self.edge_weights[edge_type_tensor]  # (num_edges, num_heads, d_k, d_k)

                        edge_weight = edge_weight.transpose(0, 1)  # Transpose for batch matrix multiplication: (num_heads, num_edges, d_k, d_k)
                        # edge_k = edge_k.transpose(0, 1)  # Transpose to (num_edges, num_heads, d_k)
                        edge_k = edge_k.unsqueeze(-1) # Add dimension for matrix multiplication (num_heads, num_edges, d_k, 1)

                        # print(edge_weight.shape, edge_k.shape)
                        edge_k = torch.matmul(edge_weight, edge_k)  # (num_heads, num_edges, d_k, 1)
                        edge_k = edge_k.squeeze(-1)  # Remove the extra dimension (num_heads, num_edges, d_k)
                    # edge_k = edge_k.transpose(0, 1)  # Transpose back (num_edges, num_heads, d_k)

                    # Apply weight matrix to keys
                    # edge_k = torch.einsum('ehij,hej->hei', edge_weight, edge_k)
                    # msg_weight = self.msg_weights[edge_type_tensor]
                    # edge_v = torch.einsum('ehij,hej->hei', msg_weight, edge_v)

                    # Compute attention scores
                    edge_scores = edge_q * edge_k
                    if has_edge_attr:
                        edge_scores = edge_scores + edge_attr
                        edge_v = edge_v * F.sigmoid(edge_gate)
                        edge_attr = edge_scores
                    
                    edge_scores = torch.sum(edge_scores, dim=-1) / math.sqrt(D) # num_heads * num_edges
                    edge_scores = torch.clamp(edge_scores, min=-5, max=5)
                    if cfg.gt.attn_mask in ['kHop']:
                        edge_scores = edge_scores + attn_mask[dst_nodes, src_nodes]

                    expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head
                    
                    # Step 2: Calculate max for each destination node per head using scatter_max
                    max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
                    max_scores = max_scores.gather(1, expanded_dst_nodes)

                    # Step 3: Exponentiate scores and sum
                    exp_scores = torch.exp(edge_scores - max_scores)
                    sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
                    sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)
                    # sum_exp_scores.clamp_(min=1e-9)

                    # Step 4: Apply softmax
                    edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
                    edge_scores = edge_scores.unsqueeze(-1)
                    edge_scores = self.dropout_attn(edge_scores)

                    out = torch.zeros((H, L, D), device=q.device)
                    out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

                else:
                    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
                    scores = F.softmax(scores, dim=-1)
                    scores = self.dropout_attn(scores)
                    
                    out = torch.matmul(scores, v)

                out = out.transpose(0,1).contiguous().view(-1, H * D)

                for idx, node_type in enumerate(node_types):
                    mask = node_type_tensor == idx
                    out_type = self.o_lin[node_type](out[mask, :])
                    h_attn_dict_list[node_type].append(out_type.squeeze())
                if has_edge_attr:
                    edge_attr = edge_attr.transpose(0,1).contiguous().view(-1, H * D)
                    for idx, edge_type_tuple in enumerate(edge_types):
                        edge_type = '__'.join(edge_type_tuple)
                        mask = edge_type_tensor == idx
                        out_type = self.oe_lin[edge_type](edge_attr[mask, :])
                        edge_attr_dict[edge_type_tuple] = out_type

            h_attn_dict = {}
            for node_type in h_attn_dict_list:
                # h_attn_dict[node_type] = torch.zeros_like(h_in_dict[node_type])
                h_attn_dict[node_type] = torch.sum(torch.stack(h_attn_dict_list[node_type], dim=0), dim=0)
                h_attn_dict[node_type] = self.dropout_global(h_attn_dict[node_type])

            if cfg.gt.residual == 'Fixed':
                h_attn_dict = {
                    node_type: h_attn_dict[node_type] + h_in_dict[node_type]
                    for node_type in node_types
                }

                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: edge_attr_dict[edge_type] + edge_attr_in_dict[edge_type]
                        for edge_type in edge_types
                    }
            elif cfg.gt.residual == 'Learn':
                alpha_dict = {
                    node_type: self.skip_global[node_type].sigmoid() for node_type in node_types
                }
                h_attn_dict = {
                    node_type: alpha_dict[node_type] * h_attn_dict[node_type] + \
                        (1 - alpha_dict[node_type]) * h_in_dict[node_type]
                    for node_type in node_types
                }
            elif cfg.gt.residual != 'none':
                raise ValueError(
                    f"Invalid attention residual option {cfg.gt.residual}"
                )
            
            # Post-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_attn_dict = {
            #         node_type: self.norm1_global[node_type](h_attn_dict[node_type])
            #         for node_type in node_types
            #     }
            #     if has_edge_attr:
            #         edge_attr_dict = {
            #             edge_type: self.norm1_edge_global["__".join(edge_type)](edge_attr_dict[edge_type])
            #             for edge_type in edge_types
            #         }

            
            # Concat output
            h_out_dict_list = {
                node_type: h_out_dict_list[node_type] + [h_attn_dict[node_type]] for node_type in node_types
            }

        # Combine global information
        h_dict = {
            node_type: sum(h_out_dict_list[node_type]) for node_type in node_types
        }
        if cfg.gt.ffn != 'none':
            # Pre-normalization
            if self.layer_norm or self.batch_norm:
                h_dict = {
                    node_type: self.norm2_ffn[node_type](h_dict[node_type])
                    for node_type in node_types
                }
            
            if cfg.gt.ffn == 'Type':
                h_dict = {
                    node_type: h_dict[node_type] + self._ff_block_type(h_dict[node_type], node_type)
                    for node_type in node_types
                }
                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: edge_attr_dict[edge_type] + self._ff_block_edge_type(edge_attr_dict[edge_type], edge_type)
                        for edge_type in edge_types
                    }
            elif cfg.gt.ffn == 'Single':
                h_dict = {
                    node_type: h_dict[node_type] + self._ff_block(h_dict[node_type])
                    for node_type in node_types
                }
            else:
                raise ValueError(
                    f"Invalid GT FFN option {cfg.gt.ffn}"
                )
                
            # Post-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_dict = {
            #         node_type: self.norm2_ffn[node_type](h_dict[node_type])
            #         for node_type in node_types
            #     }
        
        if cfg.gt.residual == 'Concat':
            h_dict = {
                node_type: torch.cat((h_in_dict[node_type], h_dict[node_type]), dim=1)
                for node_type in node_types
            }

        runtime_stats_cuda.end_region("gt-layer")

        if isinstance(batch, HeteroData):
            for node_type in node_types:
                batch[node_type].x = h_dict[node_type]
            if has_edge_attr:
                for edge_type in edge_types:
                    batch[edge_type].edge_attr = edge_attr_dict[edge_type]
        else:
            batch.node_feature = h_dict['node_type']
            if has_edge_attr:
                batch.edge_feature = edge_attr_dict[('node_type', 'edge_type', 'node_type')]

        if self.return_attention:
            return batch, None
        return batch
    
    def _ff_block_type(self, x, node_type):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1_type[node_type](x)))
        return self.ff_dropout2(self.ff_linear2_type[node_type](x))
    
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
    
    def _ff_block_edge_type(self, x, edge_type):
        """Feed Forward block.
        """
        edge_type = "__".join(edge_type)
        x = self.ff_dropout1(self.activation(self.ff_linear1_edge_type[edge_type](x)))
        return self.ff_dropout2(self.ff_linear2_edge_type[edge_type](x))

    # def __repr__(self):
    #     return '{}({}, {})'.format(self.__class__.__name__, self.dim_h,
    #                                self.dim_h)
