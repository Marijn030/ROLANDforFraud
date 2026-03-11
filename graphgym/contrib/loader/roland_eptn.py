# graphgym/contrib/loader/roland_eptn.py

import os
import pickle
import torch
import networkx as nx
from deepsnap.graph import Graph
from graphgym.register import register_loader
from graphgym.config import cfg


def load_eptn_pkl(path):

    with open(path, "rb") as f:
        G = pickle.load(f)

    node_list = list(G.nodes())
    node_map = {n: i for i, n in enumerate(node_list)}

    N = len(node_list)

    # node labels
    node_label = torch.zeros(N).long()

    for n in node_list:
        node_label[node_map[n]] = G.nodes[n]['isp']

    # node features (use constant feature if none exist)
    node_feature = torch.ones(N, 1)

    edges = []
    edge_feat = []
    edge_time = []

    for u, v, k, data in G.edges(keys=True, data=True):

        edges.append([node_map[u], node_map[v]])

        edge_feat.append([
            data.get("amount", 0),
        ])

        edge_time.append(data.get("timestamp", 0))

    edge_index = torch.tensor(edges).t().long()
    edge_feature = torch.tensor(edge_feat).float()
    edge_time = torch.tensor(edge_time).float()

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        node_label=node_label,
        directed=True
    )

    graph.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
    graph.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
    graph.node_degree_existing = torch.zeros(N)

    return [graph]


def load_eptn_dataset(format, name, dataset_dir):

    if format == "eptn":
        path = os.path.join(dataset_dir, name)
        return load_eptn_pkl(path)


register_loader("roland_eptn", load_eptn_dataset)
