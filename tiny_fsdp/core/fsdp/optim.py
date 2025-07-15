# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
from collections import OrderedDict
import torch.distributed as dist

from ..optim import sgd, adamw

def _step_fn(self):
    """Common step function for FSDP optimizers."""
    # Increment time step for optimizers that need it (like AdamW)
    if hasattr(self, 't'):
        self.t += 1
        
    for name, param in self.parameters.items():
        if param.grad is None:
            continue
            
        # Update parameter using sharded gradient
        # All ranks update their local shards independently
        param = self.one_step(name, param)
        
        # Clear gradients
        self._zero_grad(param)


class SGD(sgd.SGD):
    def __init__(self, parameters, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False):
        super().__init__(parameters, lr, momentum, dampening, weight_decay, nesterov, maximize)
    
    def _init_opt(self):
        # Initialize velocity for the local shards
        if self.momentum != 0:
            self.velocities = OrderedDict()
            for name, param in self.parameters.items():
                # Initialize velocity tensor with the same shape as the local shard
                device = param.device
                self.velocities[name] = torch.zeros_like(param, device=device)
    
    def step(self):
        _step_fn(self)


class AdamW(adamw.AdamW):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(parameters, lr, betas, eps, weight_decay, amsgrad)

    def _init_opt(self):
        # Initialize moment estimates for the local shards
        self.moments = OrderedDict()
        self.velocities = OrderedDict()
        if self.amsgrad:
            self.max_squared = OrderedDict()
            
        for name, param in self.parameters.items():
            device = param.device
            # Initialize moment tensors with the same shape as the local shard
            self.moments[name] = torch.zeros_like(param, device=device)
            self.velocities[name] = torch.zeros_like(param, device=device)
            if self.amsgrad:
                self.max_squared[name] = torch.zeros_like(param, device=device)
        
    def step(self):
        _step_fn(self) 