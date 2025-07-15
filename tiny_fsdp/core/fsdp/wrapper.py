# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict

from .module import (
    Linear,
    LayerNorm,
    Embedding,
)
from .partition import fsdp_partition_tensors
from ..utils.wrapper import target_modules, error_handling, get_init_args

class FSDP(nn.Module):
    def __init__(self, model: nn.Module, world_size: int = None, rank: int = None):
        super().__init__()
        self.module = model
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        
        # Partition the model parameters using FSDP strategy
        with torch.device('meta'):
            meta_tensors = OrderedDict(model.named_parameters())
        
        self.sharded_tensors, self.full_shapes = fsdp_partition_tensors(
            meta_tensors, 
            world_size=self.world_size,
            rank=self.rank,
            verbose=True if self.rank == 0 else False
        )
        
        # Wrap layers with FSDP-aware modules
        self.wrap_layers(self.module, _supported_modules, auto_tune=False)
        error_handling(self.module)
        
        # Initialize parameters and set metadata
        self.reinit()
        self.set_param_metadata()
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def reinit(self):
        """Reinitialize all wrapped modules with FSDP-aware parameter initialization."""
        def _replace_module_recursive(model, path=''):
            for child_name, child in model.named_children():
                full_name = f"{path}.{child_name}" if path else child_name
                if hasattr(child, "reinit_parameters"):
                    child.reinit_parameters()
                else:
                    _replace_module_recursive(child, full_name)
        _replace_module_recursive(self.module)
    
    def set_param_metadata(self):
        """Set metadata for all parameters including full shapes for reconstruction."""
        for name, param in self.module.named_parameters():
            if name in self.full_shapes:
                # Set full shape metadata for FSDP reconstruction
                param.full_shape = self.full_shapes[name]
                param.world_size = self.world_size
                param.rank = self.rank
    
    def wrap_layers(self, model: nn.Module, supported_modules: dict, auto_tune: bool = False):
        """
        Wrap the selected layers with FSDP-aware modules.
        
        Args:
            model (nn.Module): The original model to modify.
            supported_modules (dict): Mapping from original module types to FSDP modules.
            auto_tune (bool): If using auto tuning or not.
        """
        def _replace_module_recursive(model, path=''):
            for child_name, child in model.named_children():
                full_name = f"{path}.{child_name}" if path else child_name
                if isinstance(child, tuple(target_modules)):
                    module_class = supported_modules[type(child)]
                    child_init_args = get_init_args(child)
                    new_module = module_class(**child_init_args, auto_tune=auto_tune)
                    new_module.load_state_dict(child.state_dict())
                    new_module.train(child.training)
                    setattr(model, child_name, new_module)
                elif not isinstance(child, tuple(target_modules)):
                    _replace_module_recursive(child, full_name)
        _replace_module_recursive(model)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Override to return parameters with FSDP metadata."""
        return self.module.named_parameters(prefix, recurse)
    
    def parameters(self, recurse: bool = True):
        """Override to return parameters with FSDP metadata."""
        return self.module.parameters(recurse)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to gather full parameters from all ranks.
        This ensures that the saved model contains complete parameters.
        """
        from .module import all_gather_param, clear_param_cache
        
        # Temporarily gather all parameters to their full size
        gathered_params = {}
        for name, param in self.module.named_parameters():
            if hasattr(param, 'full_shape'):
                # Gather the full parameter
                full_param = all_gather_param(param, param.full_shape)
                gathered_params[name] = param.data
                param.data = full_param
        
        # Get the state dict with full parameters
        state_dict = self.module.state_dict(destination, prefix, keep_vars)
        
        # Restore original sharded parameters
        for name, param in self.module.named_parameters():
            if name in gathered_params:
                param.data = gathered_params[name]
                clear_param_cache(param)
        
        return state_dict
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Override load_state_dict to properly load full parameters and shard them.
        """
        from .module import init_shard_from_full
        
        # First load the full state dict
        # Create a temporary model to load full parameters
        temp_state_dict = {}
        for name, param in state_dict.items():
            temp_state_dict[name] = param
        
        # Load the full parameters first
        missing_keys, unexpected_keys = self.module.load_state_dict(temp_state_dict, strict=False)
        
        # Now shard all the loaded parameters
        for name, param in self.module.named_parameters():
            if name in state_dict and hasattr(param, 'full_shape'):
                # Get the full parameter that was just loaded
                full_param = param.data
                # Shard it for this rank
                sharded_param = init_shard_from_full(full_param, self.world_size, self.rank)
                param.data = sharded_param
        
        if strict and (missing_keys or unexpected_keys):
            error_msg = []
            if missing_keys:
                error_msg.append(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                error_msg.append(f"Unexpected keys: {unexpected_keys}")
            raise RuntimeError(f"Error loading state dict: {', '.join(error_msg)}")
        
        return missing_keys, unexpected_keys


# Mapping from original PyTorch modules to FSDP-aware modules
_supported_modules = {
    nn.Linear: Linear,
    nn.LayerNorm: LayerNorm,
    nn.Embedding: Embedding,
} 