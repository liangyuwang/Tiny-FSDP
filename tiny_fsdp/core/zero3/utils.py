# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch.nn as nn

class Parameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, fwd_sync=True, bwd_sync=True, rank_id=None, cached_size=None, cached_dtype=None):
        t = nn.Parameter.__new__(cls, data, requires_grad)
        t.fwd_sync = fwd_sync
        t.bwd_sync = bwd_sync
        t.rank_id = rank_id
        t.cached_size = cached_size
        t.cached_dtype = cached_dtype
        return t

