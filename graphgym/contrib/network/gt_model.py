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
        act = register.act_dict[name]
        return act() if isinstance(act, type) else act
    raise KeyError(f'Unknown activation: {name}')


class FeatureEncoder(torch.nn.Module):
    """
    No-op feature encoder for homogeneous EPTN integration.
    """
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in

        if cfg.dataset.node_encoder:
            raise NotImplementedError("node_encoder is not supported in this homogeneous EPTN integration.")
        if cfg.dataset.edge_encoder:
            raise NotImplementedError("edge_encoder is not supported in this homogeneous EPTN integration.")

    def forward(self, batch):
        return batch


class GTModel(torch.nn.Module):
    """
    Homogeneous FraudGT-style model adapted for ROLAND / EPTN.

    Expects:
      batch.node_feature : [N, raw_node_dim]
      batch.edge_feature : [E, raw_edge_dim]
      batch.edge_index   : [2, E]
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()

        if cfg.dataset.is_hetero:
            raise NotImplementedError("This GTModel version only supports homogeneous EPTN graphs.")

        if cfg.gt.layer_type != 'SparseNodeTransformer':
            raise NotImplementedError(
                f"This GTModel version only supports SparseNodeTransformer, got {cfg.gt.layer_type}."
            )

        if cfg.gt.virtual_nodes > 0:
            raise NotImplementedError("virtual_nodes is not supported in this homogeneous EPTN integration.")

        self.is_hetero = False
        self.metadata = [("node_type",), (("node_type", "edge_type", "node_type"),)]
        self.node_type = self.metadata[0][0]
        self.edge_type = self.metadata[1][0]

        self.dim_h = cfg.gt.dim_hidden
        self.input_drop = nn.Dropout(cfg.gt.input_dropout)
        self.activation = get_activation(cfg.gt.act)
        self.batch_norm = cfg.gt.batch_norm
        self.layer_norm = cfg.gt.layer_norm
        self.l2_norm = cfg.gt.l2_norm

        GNNHead = head_dict[cfg.gt.head]

        self.encoder = FeatureEncoder(dim_in)
        self.raw_node_dim = self.encoder.dim_in

        # Raw EPTN dims -> transformer hidden dim
        self.input_proj = nn.Linear(self.raw_node_dim, self.dim_h)
        self.edge_input_proj = nn.Linear(cfg.dataset.edge_dim, self.dim_h)

        # Optional pre-GT stack works on hidden-size features
        if cfg.gt.layers_pre_gt > 0:
            self.pre_gt = GTPreNN(
                self.dim_h,
                self.dim_h,
                has_bn=self.batch_norm,
                has_ln=self.layer_norm,
                has_l2norm=self.l2_norm,
            )
        else:
            self.pre_gt = None

        local_gnn_type, global_model_type = 'None', cfg.gt.layer_type

        self.convs = nn.ModuleList()
        dim_h_total = self.dim_h

        current_dim = self.dim_h
        for i in range(cfg.gt.layers):
            conv = GTLayer(
                current_dim,
                current_dim,
                current_dim,
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
                current_dim *= 2

            if cfg.gt.jumping_knowledge:
                dim_h_total += current_dim
            else:
                dim_h_total = current_dim

        # Some GraphGym heads accept positional args, some named args.
        try:
            self.post_gt = GNNHead(dim_in=dim_h_total, dim_out=dim_out)
        except TypeError:
            self.post_gt = GNNHead(dim_h_total, dim_out)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        nn.init.xavier_uniform_(self.edge_input_proj.weight)
        nn.init.zeros_(self.edge_input_proj.bias)

    def forward(self, batch):
        batch = self.encoder(batch)

        # Project raw features to hidden size expected by FraudGT layer
        x = self.input_proj(batch.node_feature)
        x = self.input_drop(x)

        batch.node_feature = x

        if hasattr(batch, 'edge_feature') and batch.edge_feature is not None:
            batch.edge_feature = self.edge_input_proj(batch.edge_feature)

        if self.pre_gt is not None:
            batch.node_feature = self.pre_gt(batch.node_feature)

        intermediates = [batch.node_feature]

        for conv in self.convs:
            batch = conv(batch)
            if cfg.gt.jumping_knowledge:
                intermediates.append(batch.node_feature)

        if cfg.gt.jumping_knowledge:
            batch.node_feature = torch.cat(intermediates, dim=1)

        if cfg.gt.l2_norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)

        return self.post_gt(batch)


register_network('GTModel', GTModel)
