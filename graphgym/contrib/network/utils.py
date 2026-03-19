import torch
import torch.nn as nn
import torch.nn.functional as F

import graphgym.register as register
from graphgym.config import cfg
from graphgym.models.layer import GeneralMultiLayer

def GTPreNN(dim_in, dim_out, **kwargs):
    """
    Wrapper for NN layer before Graph Transformer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    return GeneralMultiLayer('linear',
                             cfg.gt.layers_pre_gt,
                             dim_in,
                             dim_out,
                             dim_inner=dim_out,
                             final_act=True,
                             **kwargs)