# graphgym/contrib/loader/roland_eptn.py

import os
import pickle
import math
import torch
from deepsnap.graph import Graph
from graphgym.register import register_loader
from graphgym.config import cfg


def load_eptn_pkl(path):
    cache_path = path + ".pt"

    if os.path.exists(cache_path):
        print(f"[EPTN] Loading cached processed graph from {cache_path}")
        return torch.load(cache_path)

    print(f"[EPTN] Loading raw pickle from {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"[EPTN] Raw graph loaded: {num_nodes} nodes, {num_edges} edges")

    print("[EPTN] Building node index map...")
    node_list = list(G.nodes())
    node_map = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    print("[EPTN] Initializing node labels and node statistics...")
    node_label = torch.zeros(N, dtype=torch.long)

    in_deg = torch.zeros(N, dtype=torch.float32)
    out_deg = torch.zeros(N, dtype=torch.float32)

    in_amt_sum = torch.zeros(N, dtype=torch.float32)
    out_amt_sum = torch.zeros(N, dtype=torch.float32)

    in_tx_count = torch.zeros(N, dtype=torch.float32)
    out_tx_count = torch.zeros(N, dtype=torch.float32)

    first_ts = torch.full((N,), float("inf"), dtype=torch.float32)
    last_ts = torch.zeros(N, dtype=torch.float32)

    print("[EPTN] Reading node labels...")
    for idx, n in enumerate(node_list):
        node_label[node_map[n]] = int(G.nodes[n].get("isp", 0))
        if idx % 500_000 == 0 and idx > 0:
            print(f"[EPTN] Processed {idx}/{N} nodes")

    print("[EPTN] Preallocating edge tensors...")
    E = num_edges
    edge_index = torch.empty((2, E), dtype=torch.long)
    edge_feature = torch.empty((E, 2), dtype=torch.float32)  # [log_amount, norm_timestamp]
    edge_time = torch.empty(E, dtype=torch.float32)

    print("[EPTN] Processing edges...")
    for idx, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
        u_idx = node_map[u]
        v_idx = node_map[v]

        amount = float(data.get("amount", 0.0))
        timestamp = float(data.get("timestamp", 0.0))

        edge_index[0, idx] = u_idx
        edge_index[1, idx] = v_idx

        edge_feature[idx, 0] = math.log1p(max(amount, 0.0))
        edge_feature[idx, 1] = timestamp
        edge_time[idx] = timestamp

        out_deg[u_idx] += 1.0
        in_deg[v_idx] += 1.0

        out_tx_count[u_idx] += 1.0
        in_tx_count[v_idx] += 1.0

        out_amt_sum[u_idx] += amount
        in_amt_sum[v_idx] += amount

        if timestamp > 0:
            if timestamp < first_ts[u_idx]:
                first_ts[u_idx] = timestamp
            if timestamp < first_ts[v_idx]:
                first_ts[v_idx] = timestamp
            if timestamp > last_ts[u_idx]:
                last_ts[u_idx] = timestamp
            if timestamp > last_ts[v_idx]:
                last_ts[v_idx] = timestamp

        if idx % 1_000_000 == 0 and idx > 0:
            print(f"[EPTN] Processed {idx}/{E} edges")

    print("[EPTN] Finalizing node features...")
    first_ts[first_ts == float("inf")] = 0.0

    avg_in_amt = in_amt_sum / torch.clamp(in_tx_count, min=1.0)
    avg_out_amt = out_amt_sum / torch.clamp(out_tx_count, min=1.0)
    active_span = torch.clamp(last_ts - first_ts, min=0.0)

    node_feature = torch.stack(
        [
            torch.log1p(in_deg),
            torch.log1p(out_deg),
            torch.log1p(in_amt_sum),
            torch.log1p(out_amt_sum),
            torch.log1p(avg_in_amt),
            torch.log1p(avg_out_amt),
            torch.log1p(in_tx_count),
            torch.log1p(out_tx_count),
            torch.log1p(active_span),
        ],
        dim=1,
    )

    print("[EPTN] Normalizing timestamp edge feature...")
    if edge_time.numel() > 0:
        t_min = edge_time.min()
        t_max = edge_time.max()
        if t_max > t_min:
            edge_feature[:, 1] = (edge_feature[:, 1] - t_min) / (t_max - t_min)
        else:
            edge_feature[:, 1] = 0.0

    print("[EPTN] Creating DeepSNAP graph...")
    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        node_label=node_label,
        directed=True,
    )

    graph.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
    graph.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
    graph.node_degree_existing = in_deg + out_deg

    graphs = [graph]

    print(f"[EPTN] Saving processed cache to {cache_path}")
    torch.save(graphs, cache_path)

    print("[EPTN] Done")
    return graphs


def load_eptn_dataset(format, name, dataset_dir):
    if format == "eptn":
        path = os.path.join(dataset_dir, name)
        return load_eptn_pkl(path)


register_loader("roland_eptn", load_eptn_dataset)