# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn

class Parameter(nn.Parameter):
    """
    FSDP Parameter class that holds sharded tensor and metadata for reconstruction.
    """
    def __new__(cls, data=None, requires_grad=True):
        t = nn.Parameter.__new__(cls, data, requires_grad)
        t.full_shape = None
        t.shard_metadata = None  # Store shard information for this rank
        t.bwd_sync = True  # For compatibility with error_handling
        return t 