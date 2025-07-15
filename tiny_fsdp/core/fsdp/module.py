# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.distributed as dist
import math
import os

from ..module import (
    ops,
    linear, 
    normalization, 
    embedding,
)
from .utils import Parameter
from .partition import gather_tensor, scatter_tensor

def all_gather_param(param, full_shape, async_op=False):
    """Gather sharded parameter to reconstruct full tensor for computation."""
    if hasattr(param, '_fsdp_full_tensor_cache'):
        # Return cached full tensor if available
        return param._fsdp_full_tensor_cache
        
    full_tensor = gather_tensor(param.data, full_shape)
    
    # Cache the full tensor to avoid repeated all-gather in the same forward/backward pass
    param._fsdp_full_tensor_cache = full_tensor
    return full_tensor

def reduce_scatter_grad(grad, param):
    """Reduce-scatter gradient to get local shard."""
    if grad is None:
        return None
    
    # Reduce-scatter the gradient
    local_grad_shard = scatter_tensor(grad)
    return local_grad_shard

def clear_param_cache(param):
    """Clear the cached full tensor to free memory."""
    if hasattr(param, '_fsdp_full_tensor_cache'):
        delattr(param, '_fsdp_full_tensor_cache')

def init_shard_from_full(full_tensor, world_size=None, rank=None):
    """Initialize a parameter shard from full tensor."""
    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()
        
    if len(full_tensor.shape) == 0:  # scalar
        return full_tensor
        
    # Calculate shard size and indices
    dim0_size = full_tensor.shape[0]
    shard_size = (dim0_size + world_size - 1) // world_size
    start_idx = rank * shard_size
    end_idx = min(start_idx + shard_size, dim0_size)
    
    if start_idx >= dim0_size:
        # Empty shard
        shard_shape = (0,) + full_tensor.shape[1:]
        return torch.empty(shard_shape, dtype=full_tensor.dtype, device=full_tensor.device)
    
    return full_tensor[start_idx:end_idx].clone()


class Linear(linear.Linear):
    def _init_parameters(self):
        with torch.device('meta'):  # Fake init
            self.weight = Parameter(torch.empty((self.out_features, self.in_features), **self.factory_kwargs))
            if self.use_bias:
                self.bias = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
            else:
                self.register_parameter('bias', None)
    
    def reinit_parameters(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f'cuda:{local_rank}'
        
        # Create full tensors and then shard them
        if dist.get_rank() == 0:
            # Initialize full tensors on rank 0
            weight_full = torch.empty((self.out_features, self.in_features), 
                                    dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                    device=device)
            nn.init.kaiming_uniform_(weight_full, a=math.sqrt(5))
            
            if self.use_bias:
                bias_full = torch.empty(self.out_features, 
                                      dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                      device=device)
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias_full, -bound, bound)
        else:
            # Create dummy tensors on other ranks
            weight_full = torch.empty((self.out_features, self.in_features), 
                                    dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                    device=device)
            if self.use_bias:
                bias_full = torch.empty(self.out_features, 
                                      dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                      device=device)
        
        # Broadcast full tensors to all ranks
        dist.broadcast(weight_full, src=0)
        if self.use_bias:
            dist.broadcast(bias_full, src=0)
        
        # Shard the tensors
        weight_shard = init_shard_from_full(weight_full)
        self.weight = Parameter(weight_shard)
        self.weight.full_shape = (self.out_features, self.in_features)
        
        if self.use_bias:
            bias_shard = init_shard_from_full(bias_full)
            self.bias = Parameter(bias_shard)
            self.bias.full_shape = (self.out_features,)
        else:
            self.register_parameter('bias', None)
    
    def forward_callback(self, ctx, input, weight, bias, runtime_tuner):
        # Gather full weight tensor
        weight_full = all_gather_param(weight, weight.full_shape)
        bias_full = None
        if bias is not None:
            bias_full = all_gather_param(bias, bias.full_shape)
        
        ctx.save_for_backward(input, weight_full, bias_full)
        ctx.weight_param = weight  # Save reference to sharded parameter
        ctx.bias_param = bias if bias is not None else None
        
        # Forward computation with full tensors
        output = ops.linear_forward(input, weight_full, bias_full, runtime_tuner)
        
        return ctx, output

    def backward_callback(self, ctx, grad_output, runtime_tuner):
        input, weight_full, bias_full = ctx.saved_tensors

        # Compute gradients with full tensors
        grad_weight_full = ops.linear_weight_grad(grad_output, input, weight_full, runtime_tuner) if ctx.needs_input_grad[1] else None
        grad_bias_full = ops.linear_bias_grad(grad_output, input, weight_full, runtime_tuner) if bias_full is not None and ctx.needs_input_grad[2] else None
        grad_input = ops.linear_input_grad(grad_output, input, weight_full, runtime_tuner) if ctx.needs_input_grad[0] else None

        # Check if the grad shape is correct
        if grad_input is not None and grad_input.shape != input.shape:
            raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
        if grad_weight_full is not None and grad_weight_full.shape != weight_full.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight_full.shape} is not equal to weight shape {weight_full.shape}")
        if grad_bias_full is not None and grad_bias_full.shape != bias_full.shape:
            raise RuntimeError(f"grad_bias shape {grad_bias_full.shape} is not equal to bias shape {bias_full.shape}")
        
        # Reduce-scatter gradients to get local shards
        grad_weight = reduce_scatter_grad(grad_weight_full, ctx.weight_param) if grad_weight_full is not None else None
        grad_bias = reduce_scatter_grad(grad_bias_full, ctx.bias_param) if grad_bias_full is not None else None
        
        # Clear cached full tensors to free memory
        clear_param_cache(ctx.weight_param)
        if ctx.bias_param is not None:
            clear_param_cache(ctx.bias_param)

        return grad_input, grad_weight, grad_bias


class LayerNorm(normalization.LayerNorm):
    def _init_parameters(self):
        with torch.device('meta'):  # Fake init
            if self.elementwise_affine:
                self.weight = Parameter(torch.empty(self.normalized_shape, **self.factory_kwargs))
                if self.use_bias:
                    self.bias = Parameter(torch.empty(self.normalized_shape, **self.factory_kwargs))
                else:
                    self.register_parameter('bias', None)
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
    
    def reinit_parameters(self):
        if self.elementwise_affine:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f'cuda:{local_rank}'
            
            # For LayerNorm, parameters are usually small, we can replicate them across all ranks
            # Or shard them if they are large enough
            if dist.get_rank() == 0:
                weight_full = torch.ones(self.normalized_shape, 
                                       dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                       device=device)
                if self.use_bias:
                    bias_full = torch.zeros(self.normalized_shape, 
                                          dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                          device=device)
            else:
                weight_full = torch.empty(self.normalized_shape, 
                                        dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                        device=device)
                if self.use_bias:
                    bias_full = torch.empty(self.normalized_shape, 
                                          dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                          device=device)
            
            # Broadcast to all ranks
            dist.broadcast(weight_full, src=0)
            if self.use_bias:
                dist.broadcast(bias_full, src=0)
            
            # For small tensors like LayerNorm, we might just replicate
            # But for consistency with FSDP, we shard them
            weight_shard = init_shard_from_full(weight_full)
            self.weight = Parameter(weight_shard)
            self.weight.full_shape = self.normalized_shape
            
            if self.use_bias:
                bias_shard = init_shard_from_full(bias_full)
                self.bias = Parameter(bias_shard)
                self.bias.full_shape = self.normalized_shape
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward_callback(self, ctx, input, weight, bias, eps, runtime_tuner):
        # Gather full tensors
        weight_full = all_gather_param(weight, weight.full_shape) if weight is not None else None
        bias_full = all_gather_param(bias, bias.full_shape) if bias is not None else None
        
        output, mean, rstd, args = ops.layernorm_fwd(input, weight_full, bias_full, eps, runtime_tuner)
        
        ctx.save_for_backward(input, weight_full, bias_full, mean, rstd)
        ctx.args = args
        ctx.weight_param = weight
        ctx.bias_param = bias
        
        return ctx, output

    def backward_callback(self, ctx, grad_output, eps, runtime_tuner):
        input, weight_full, bias_full, mean, rstd = ctx.saved_tensors
        
        args = {
            'BLOCK_SIZE': ctx.args['BLOCK_SIZE'],
            'num_warps': ctx.args['num_warps'],
            'eps': eps,
        }
        dx, dw_full, db_full, args = ops.layernorm_dx(grad_output, input, weight_full, bias_full, mean, rstd, args, runtime_tuner)
        dw_full, db_full = ops.layernorm_dwdb(weight_full, bias_full, dw_full, db_full, args, runtime_tuner)
        
        # Check if the grad shape is correct
        if dx is not None and dx.shape != input.shape:
            raise RuntimeError(f"grad_input shape {dx.shape} is not equal to input shape {input.shape}")
        if dw_full is not None and dw_full.shape != weight_full.shape:
            raise RuntimeError(f"grad_weight shape {dw_full.shape} is not equal to weight shape {weight_full.shape}")
        if db_full is not None and db_full.shape != bias_full.shape:
            raise RuntimeError(f"grad_bias shape {db_full.shape} is not equal to bias shape {bias_full.shape}")

        # Reduce-scatter gradients
        dw = reduce_scatter_grad(dw_full, ctx.weight_param) if dw_full is not None else None
        db = reduce_scatter_grad(db_full, ctx.bias_param) if db_full is not None else None
        
        # Clear cached full tensors
        if ctx.weight_param is not None:
            clear_param_cache(ctx.weight_param)
        if ctx.bias_param is not None:
            clear_param_cache(ctx.bias_param)

        return dx, dw, db


class Embedding(embedding.Embedding):
    def _init_parameters(self):
        with torch.device('meta'):  # Fake init
            if self._weight is None:
                self.weight = Parameter(torch.empty((self.num_embeddings, self.embedding_dim), **self.factory_kwargs),
                                        requires_grad=not self._freeze)
            else:
                raise NotImplementedError("Pretrained Embedding weight is not supported yet.")

    def reinit_parameters(self):
        if self._weight is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f'cuda:{local_rank}'
            
            if dist.get_rank() == 0:
                # Initialize full embedding on rank 0
                weight_full = torch.empty((self.num_embeddings, self.embedding_dim), 
                                        dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                        device=device)
                nn.init.normal_(weight_full)
            else:
                weight_full = torch.empty((self.num_embeddings, self.embedding_dim), 
                                        dtype=self.factory_kwargs.get('dtype', torch.float32), 
                                        device=device)
            
            # Broadcast to all ranks
            dist.broadcast(weight_full, src=0)
            
            # Shard the embedding weight
            weight_shard = init_shard_from_full(weight_full)
            self.weight = Parameter(weight_shard, requires_grad=not self._freeze)
            self.weight.full_shape = (self.num_embeddings, self.embedding_dim)
        else:
            raise NotImplementedError("Pretrained Embedding weight is not supported yet.")

    def forward_callback(self, ctx, input, weight, padding_idx, max_norm, norm_type, runtime_tuner):
        # Gather full weight tensor
        weight_full = all_gather_param(weight, weight.full_shape)
        
        ctx.save_for_backward(input, weight_full)
        ctx.weight_param = weight
        
        output = ops.embedding_forward(input, weight_full, padding_idx, max_norm, norm_type, runtime_tuner)
        
        return ctx, output

    def backward_callback(self, ctx, grad_output, padding_idx, max_norm, norm_type, runtime_tuner):
        input, weight_full = ctx.saved_tensors

        # Compute gradients
        grad_weight_full = ops.embedding_weight_grad(grad_output, input, weight_full, runtime_tuner) if ctx.needs_input_grad[1] else None

        # Check if the grad shape is correct
        if grad_weight_full is not None and grad_weight_full.shape != weight_full.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight_full.shape} is not equal to weight shape {weight_full.shape}")

        # Reduce-scatter gradients
        grad_weight = reduce_scatter_grad(grad_weight_full, ctx.weight_param) if grad_weight_full is not None else None
        
        # Clear cached full tensors
        clear_param_cache(ctx.weight_param)

        return grad_weight 