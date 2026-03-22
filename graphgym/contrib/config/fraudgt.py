from yacs.config import CfgNode as CN
from graphgym.register import register_config


def set_cfg_fraudgt(cfg):
    # FraudGT-specific config namespace
    cfg.gt = CN()

    # architecture
    cfg.gt.head = 'default'
    cfg.gt.layer_type = 'TorchTransformer'
    cfg.gt.dim_hidden = 64
    cfg.gt.layers = 2
    cfg.gt.layers_pre_gt = 0

    # normalization / activation
    cfg.gt.act = 'relu'
    cfg.gt.batch_norm = False
    cfg.gt.layer_norm = True
    cfg.gt.l2_norm = False

    # dropout
    cfg.gt.input_dropout = 0.0
    cfg.gt.dropout = 0.0
    cfg.gt.attn_dropout = 0.0

    # attention
    cfg.gt.attn_heads = 1
    cfg.gt.attn_mask = 'Edge'   # Edge, kHop, Bias, none
    cfg.gt.hops = 1
    cfg.gt.edge_weight = False

    # residual / FFN
    cfg.gt.residual = 'Fixed'   # Fixed, Learn, Concat, none
    cfg.gt.ffn = 'Single'       # Single, Type, none

    # extras
    cfg.gt.virtual_nodes = 0
    cfg.gt.jumping_knowledge = False


register_config('fraudgt', set_cfg_fraudgt)