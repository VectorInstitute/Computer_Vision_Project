#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.config import CfgNode

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.TRAIN.CHECKPOINT_TRANSFER = CfgNode()

    _C.TRAIN.CHECKPOINT_TRANSFER.ENABLE = False
    _C.TRAIN.CHECKPOINT_TRANSFER.PATH_MAPPING = "/h/rgoyal/code/SlowFast/assets/class_mappings/dict_similarity_1.pkl"

