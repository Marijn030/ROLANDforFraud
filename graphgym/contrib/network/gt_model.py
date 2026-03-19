import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

import graphgym.register as register
from graphgym.config import cfg
from graphgym.register import register_network

from graphgym.contrib.layer.gt_layer import GTLayer
from graphgym.contrib.network.utils import GTPreNN

from graphgym.models.head import head_dict

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
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, is_hetero=False):
        super(FeatureEncoder, self).__init__()
        self.is_hetero = is_hetero
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            raise NotImplementedError("node_encoder requires dataset access in this integration.")
        if cfg.dataset.edge_encoder:
            raise NotImplementedError("edge_encoder requires dataset access in this integration.")

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GTModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.is_hetero  = False
        self.metadata = [("node_type",), (("node_type", "edge_type", "node_type"), )]
        self.dim_h      = cfg.gt.dim_hidden
        self.input_drop = nn.Dropout(cfg.gt.input_dropout)
        self.activation = get_activation(cfg.gt.act)
        self.batch_norm = cfg.gt.batch_norm
        self.layer_norm = cfg.gt.layer_norm
        self.l2_norm    = cfg.gt.l2_norm
        GNNHead         = head_dict[cfg.gt.head]

        self.encoder = FeatureEncoder(dim_in, is_hetero=self.is_hetero)
        self.dim_in = self.encoder.dim_in

        if not self.is_hetero:
            self.input_proj = nn.Linear(self.dim_in, self.dim_h)
            self.edge_input_proj = nn.Linear(cfg.dataset.edge_dim, self.dim_h)
        else:
            self.input_proj = None
            self.edge_input_proj = None

        if cfg.gt.layers_pre_gt > 0:
            if not self.is_hetero:
                self.dim_in = {self.metadata[0][0]: self.dim_in}
            self.pre_gt_dict = torch.nn.ModuleDict()
            for node_type in self.metadata[0]:
                self.pre_gt_dict[node_type] = GTPreNN(
                    self.dim_in[node_type], self.dim_h,
                    has_bn=self.batch_norm, has_ln=self.layer_norm,
                    has_l2norm=self.l2_norm
                )
        
        
        try:
            layer_type = cfg.gt.layer_type
            if layer_type in ['TorchTransformer', 'SparseNodeTransformer']:
                local_gnn_type, global_model_type = 'None', layer_type
            else:
                local_gnn_type, global_model_type = layer_type, 'None'
        except:
            raise ValueError(f"Unexpected layer type: {layer_type}")

        self.num_virtual_nodes = cfg.gt.virtual_nodes
        if self.num_virtual_nodes > 0:
            self.virtual_nodes = nn.ParameterDict()
            for node_type in dim_in:
                self.virtual_nodes[node_type] = nn.Parameter(torch.empty((self.num_virtual_nodes, self.dim_h)))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim_h_total = self.dim_h
        for i in range(cfg.gt.layers):
            conv = GTLayer(self.dim_h, self.dim_h, self.dim_h, self.metadata,
                    local_gnn_type, global_model_type, i,
                    cfg.gt.attn_heads, 
                    # layer_norm=False,
                    # batch_norm=False)
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    return_attention=False)
            self.convs.append(conv)

            if self.layer_norm or self.batch_norm:
                self.norms.append(nn.ModuleDict())
                for node_type in self.metadata[0]:
                    if self.layer_norm:
                        self.norms[-1][node_type] = nn.LayerNorm(self.dim_h)
                    elif self.batch_norm:
                        self.norms[-1][node_type] = nn.BatchNorm1d(self.dim_h)
            
            if cfg.gt.residual == 'Concat':
                self.dim_h *= 2
            if cfg.gt.jumping_knowledge:
                dim_h_total += self.dim_h
            else:
                dim_h_total = self.dim_h

        self.post_gt = GNNHead(dim_h_total, dim_out)
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_virtual_nodes > 0:
            for node_type in self.virtual_nodes:
                torch.nn.init.normal_(self.virtual_nodes[node_type])

    def forward(self, batch):
        batch = self.encoder(batch)
        if isinstance(batch, HeteroData):
            node_types = batch.node_types
            edge_types = batch.edge_types
            h_dict, edge_index_dict = batch.collect('x'), batch.collect('edge_index')
        else:
            node_types = [self.metadata[0][0]]
            edge_types = [self.metadata[1][0]]
            h_dict = {self.metadata[0][0]: batch.node_feature}
            edge_index_dict = {self.metadata[1][0]: batch.edge_index}

        if not self.is_hetero:
            h_dict[self.metadata[0][0]] = self.input_proj(h_dict[self.metadata[0][0]])
            if hasattr(batch, 'edge_feature') and batch.edge_feature is not None:
                batch.edge_feature = self.edge_input_proj(batch.edge_feature)

        h_dict = {
            node_type: self.input_drop(h_dict[node_type]) for node_type in h_dict
        }

        if cfg.gt.layers_pre_gt > 0:
            h_dict = {
                node_type: self.pre_gt_dict[node_type](h_dict[node_type]) for node_type in h_dict
            }

        interm = {node_type: [h_dict[node_type]] for node_type in h_dict}
        num_nodes_dict = None
        if self.num_virtual_nodes > 0:
            # Concat global virtual nodes to the end, so the edge_index leaves untouched
            h_dict = {
                node_type: torch.cat((h, self.virtual_nodes[node_type]), dim=0)
                for node_type, h in h_dict.items()
            }
            # Connect global virtual nodes to every node
            num_nodes_dict = batch.num_nodes_dict
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                rows, cols = [], []
                for i in range(self.num_virtual_nodes):
                    rows.append(torch.full((1, num_nodes_dict[dst_type]), num_nodes_dict[src_type] + i, device=edge_index_dict[edge_type].device))
                    cols.append(torch.arange(num_nodes_dict[dst_type], device=edge_index_dict[edge_type].device).view(1, -1))

                edge_index_dict[edge_type] = torch.cat((edge_index_dict[edge_type], torch.cat((torch.cat(rows, dim=-1), torch.cat(cols, dim=-1)))), dim=-1)
                if src_type == dst_type:
                    edge_index_dict[edge_type] = torch.cat((edge_index_dict[edge_type], torch.cat((torch.cat(cols, dim=-1), torch.cat(rows, dim=-1)))), dim=-1)

            for node_type in node_types:
                batch[node_type].num_nodes += self.num_virtual_nodes

        # Write back for conv layer
        if isinstance(batch, HeteroData):
            for node_type in node_types:
                batch[node_type].x = h_dict[node_type]
        else:
            batch.node_feature = h_dict[self.metadata[0][0]]
        for i in range(cfg.gt.layers):
            batch = self.convs[i](batch)
            # batch = self.convs[i](batch)
            # batch.saved_scores = saved_scores
            if cfg.gt.jumping_knowledge:
                h_temp_dict = batch.collect('x')
                if self.num_virtual_nodes > 0:
                    # Remove the virtual nodes
                    h_temp_dict = {
                        node_type: h[:num_nodes_dict[node_type], :] for node_type, h in h_dict.items()
                    }
                for node_type in h_dict:
                    interm[node_type] = interm[node_type] + [h_temp_dict[node_type]]

        if self.num_virtual_nodes > 0:
            # Remove the virtual nodes
            for node_type in node_types:
                batch[node_type].x = batch[node_type].x[:num_nodes_dict[node_type], :]
                batch[node_type].num_nodes -= self.num_virtual_nodes

        # Jumping knowledge.
        if cfg.gt.jumping_knowledge:
            for node_type in node_types:
                batch[node_type].x = torch.cat(interm[node_type], dim=1)

        # Output L2 norm
        if cfg.gt.l2_norm:
            for node_type in node_types:
                batch[node_type].x = F.normalize(batch[node_type].x, p=2, dim=-1) 

        return self.post_gt(batch)

register_network('GTModel', GTModel)
