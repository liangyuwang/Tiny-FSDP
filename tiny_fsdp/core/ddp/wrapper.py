# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn

from .module import (
    Linear,
    LayerNorm,
    Embedding,
)
from ..utils.wrapper import wrap_layers, error_handling

class DDP(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        wrap_layers(model, 
                    _supported_modules,
                    auto_tune=False)
        error_handling(model)
        self.module = model
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        if self.require_backward_grad_sync:
            self.enable_grad_sync()
        self.require_backward_grad_sync = False
        return self.module(*args, **kwargs)

    def enable_grad_sync(self):
        for param in self.module.parameters():
            setattr(param, 'bwd_sync', True)


_supported_modules = {
    nn.Linear: Linear,
    nn.LayerNorm: LayerNorm,
    nn.Embedding: Embedding,
}
