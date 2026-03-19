import torch
import torch.nn as nn
import torch.nn.functional as F

import graphgym.register as register
from graphgym.config import cfg
from graphgym.register import register_network
from graphgym.models.head import head_dict

from graphgym.contrib.layer.gt_layer import GTLayer
from graphgym.contrib.network.utils import GTPreNN


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


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features.
    """

    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in

        if cfg.dataset.node_encoder:
            raise NotImplementedError(
                "cfg.dataset.node_encoder=True is not supported yet in this GTModel integration."
            )
        if cfg.dataset.edge_encoder:
            raise NotImplementedError(
                "cfg.dataset.edge_encoder=True is not supported yet in this GTModel integration."
            )

    def forward(self, batch):
        return batch


class GTModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()

        self.is_hetero = bool(getattr(cfg.dataset, 'is_hetero', False))
        if self.is_hetero:
            raise NotImplementedError(
                "Heterogeneous GTModel integration is not enabled in this version yet."
            )

        self.metadata = (
            ('node_type',),
            (('node_type', 'edge_type', 'node_type'),),
        )

        self.dim_h = cfg.gt.dim_hidden
        self.input_drop = nn.Dropout(cfg.gt.input_dropout)
        self.activation = get_activation(cfg.gt.act)
        self.batch_norm = cfg.gt.batch_norm
        self.layer_norm = cfg.gt.layer_norm
        self.l2_norm = cfg.gt.l2_norm

        GNNHead = head_dict[cfg.gt.head]

        self.encoder = FeatureEncoder(dim_in)
        self.dim_in = self.encoder.dim_in

        # Always align raw input dimension -> hidden dimension.
        self.input_proj = nn.Linear(self.dim_in, self.dim_h)

        if cfg.gt.layers_pre_gt > 0:
            self.pre_gt = GTPreNN(
                self.dim_h,
                self.dim_h,
                has_bn=self.batch_norm,
                has_l2norm=self.l2_norm,
            )
        else:
            self.pre_gt = None

        layer_type = cfg.gt.layer_type
        if layer_type in ['TorchTransformer', 'SparseNodeTransformer']:
            local_gnn_type, global_model_type = 'None', layer_type
        else:
            raise ValueError(f"Unexpected layer type: {layer_type}")

        self.num_virtual_nodes = cfg.gt.virtual_nodes
        if self.num_virtual_nodes > 0:
            self.virtual_nodes = nn.ParameterDict()
            for node_type in self.metadata[0]:
                self.virtual_nodes[node_type] = nn.Parameter(
                    torch.empty((self.num_virtual_nodes, self.dim_h))
                )

        self.convs = nn.ModuleList()

        dim_h_total = self.dim_h
        current_dim_h = self.dim_h

        for i in range(cfg.gt.layers):
            conv = GTLayer(
                current_dim_h,
                current_dim_h,
                current_dim_h,
                self.metadata,
                local_gnn_type,
                global_model_type,
                i,
                cfg.gt.attn_heads,
                layer_norm=self.layer_norm,
                batch_norm=self.batch_norm,
                return_attention=False,
            )
            self.convs.append(conv)

            if cfg.gt.residual == 'Concat':
                current_dim_h *= 2

            if cfg.gt.jumping_knowledge:
                dim_h_total += current_dim_h
            else:
                dim_h_total = current_dim_h

        self.post_gt = GNNHead(dim_in=dim_h_total, dim_out=dim_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        if self.num_virtual_nodes > 0:
            for node_type in self.virtual_nodes:
                torch.nn.init.normal_(self.virtual_nodes[node_type])

    def forward(self, batch):
        batch = self.encoder(batch)

        node_types = [self.metadata[0][0]]
        edge_types = [self.metadata[1][0]]

        # ROLAND/GraphGym batch format
        h = batch.node_feature

        # Project raw features (e.g. 13-dim) to transformer hidden dim (e.g. 32)
        h = self.input_proj(h)
        h = self.input_drop(h)

        if self.pre_gt is not None:
            h = self.pre_gt(h)

        h_dict = {node_types[0]: h}
        edge_index_dict = {edge_types[0]: batch.edge_index}

        interm = {node_types[0]: [h_dict[node_types[0]]]}
        num_nodes_dict = None

        if self.num_virtual_nodes > 0:
            h_dict = {
                node_type: torch.cat((h_val, self.virtual_nodes[node_type]), dim=0)
                for node_type, h_val in h_dict.items()
            }

            num_nodes_dict = {node_types[0]: batch.num_nodes}

            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                rows, cols = [], []

                for i in range(self.num_virtual_nodes):
                    rows.append(
                        torch.full(
                            (1, num_nodes_dict[dst_type]),
                            num_nodes_dict[src_type] + i,
                            device=edge_index.device,
                            dtype=edge_index.dtype,
                        )
                    )
                    cols.append(
                        torch.arange(
                            num_nodes_dict[dst_type],
                            device=edge_index.device,
                            dtype=edge_index.dtype,
                        ).view(1, -1)
                    )

                edge_index_dict[edge_type] = torch.cat(
                    (
                        edge_index_dict[edge_type],
                        torch.cat(
                            (torch.cat(rows, dim=-1), torch.cat(cols, dim=-1)),
                            dim=0,
                        ),
                    ),
                    dim=-1,
                )

                if src_type == dst_type:
                    edge_index_dict[edge_type] = torch.cat(
                        (
                            edge_index_dict[edge_type],
                            torch.cat(
                                (torch.cat(cols, dim=-1), torch.cat(rows, dim=-1)),
                                dim=0,
                            ),
                        ),
                        dim=-1,
                    )

            batch.num_nodes += self.num_virtual_nodes

        batch.node_feature = h_dict[node_types[0]]
        batch.edge_index = edge_index_dict[edge_types[0]]

        for i in range(cfg.gt.layers):
            batch = self.convs[i](batch)

            if cfg.gt.jumping_knowledge:
                h_temp_dict = {node_types[0]: batch.node_feature}

                if self.num_virtual_nodes > 0:
                    h_temp_dict = {
                        node_type: h_val[:num_nodes_dict[node_type], :]
                        for node_type, h_val in h_temp_dict.items()
                    }

                for node_type in node_types:
                    interm[node_type].append(h_temp_dict[node_type])

        if self.num_virtual_nodes > 0:
            batch.node_feature = batch.node_feature[:num_nodes_dict[node_types[0]], :]
            batch.num_nodes -= self.num_virtual_nodes

        if cfg.gt.jumping_knowledge:
            batch.node_feature = torch.cat(interm[node_types[0]], dim=1)

        if cfg.gt.l2_norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)

        return self.post_gt(batch)


register_network('GTModel', GTModel)