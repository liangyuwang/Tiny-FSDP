# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
from collections import OrderedDict
import torch.distributed as dist

def fsdp_partition_tensors(
        tensors_dict: OrderedDict, 
        world_size: int = None,
        rank: int = None,
        verbose: bool = False,
    ):
    """
    Partition the tensors of a model using FSDP strategy (intra-tensor sharding).
    Each tensor is split along dim-0 and distributed across all ranks.
    
    Args:
        tensors_dict: OrderedDict, the dict of model tensors to partition.
        world_size: int, the number of ranks in the distributed setup.
        rank: int, the current rank.
        verbose: bool, whether to print partition information.
    
    Returns:
        sharded_tensors_dict: OrderedDict, tensors sharded for the current rank
        full_shapes: dict, original shapes of all tensors for reconstruction
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()
    
    sharded_tensors_dict = OrderedDict()
    full_shapes = {}
    
    for name, tensor in tensors_dict.items():
        original_shape = tensor.shape
        full_shapes[name] = original_shape
        
        if len(original_shape) == 0:  # scalar tensor
            # For scalar tensors, just replicate on each rank
            sharded_tensors_dict[name] = tensor
            if verbose:
                print(f"Scalar tensor {name} replicated on rank {rank}")
            continue
            
        # Calculate shard size along dim-0
        dim0_size = original_shape[0]
        shard_size = (dim0_size + world_size - 1) // world_size  # Ceiling division
        
        # Calculate start and end indices for this rank
        start_idx = rank * shard_size
        end_idx = min(start_idx + shard_size, dim0_size)
        
        if start_idx >= dim0_size:
            # This rank gets an empty shard
            shard_shape = (0,) + original_shape[1:]
            sharded_tensor = torch.empty(shard_shape, dtype=tensor.dtype, device=tensor.device)
        else:
            # Extract the shard for this rank
            if tensor.device.type == "meta":
                # For meta tensors, create the appropriate shard shape
                shard_shape = (end_idx - start_idx,) + original_shape[1:]
                sharded_tensor = torch.empty(shard_shape, dtype=tensor.dtype, device=tensor.device)
            else:
                sharded_tensor = tensor[start_idx:end_idx]
        
        sharded_tensors_dict[name] = sharded_tensor
        
        if verbose:
            print(f"Rank {rank}: {name} {original_shape} -> shard [{start_idx}:{end_idx}] {sharded_tensor.shape}")
    
    return sharded_tensors_dict, full_shapes


def gather_tensor(sharded_tensor, full_shape, world_size=None, rank=None):
    """
    Gather a sharded tensor from all ranks to reconstruct the full tensor.
    
    Args:
        sharded_tensor: torch.Tensor, the local shard
        full_shape: tuple, the original full tensor shape
        world_size: int, number of ranks
        rank: int, current rank
    
    Returns:
        full_tensor: torch.Tensor, the reconstructed full tensor
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()
        
    if len(full_shape) == 0:  # scalar tensor
        return sharded_tensor
    
    # Calculate shard sizes for all ranks
    dim0_size = full_shape[0]
    shard_size = (dim0_size + world_size - 1) // world_size
    
    # Prepare list to collect all shards
    all_shards = []
    for r in range(world_size):
        start_idx = r * shard_size
        end_idx = min(start_idx + shard_size, dim0_size)
        if start_idx >= dim0_size:
            shard_shape = (0,) + full_shape[1:]
        else:
            shard_shape = (end_idx - start_idx,) + full_shape[1:]
        
        # Create tensor for this rank's shard
        shard = torch.empty(shard_shape, dtype=sharded_tensor.dtype, device=sharded_tensor.device)
        all_shards.append(shard)
    
    # Perform all-gather
    dist.all_gather(all_shards, sharded_tensor)
    
    # Concatenate shards to reconstruct full tensor
    # Filter out empty shards
    valid_shards = [shard for shard in all_shards if shard.numel() > 0]
    if not valid_shards:
        return torch.empty(full_shape, dtype=sharded_tensor.dtype, device=sharded_tensor.device)
    
    full_tensor = torch.cat(valid_shards, dim=0)
    
    # Ensure the shape matches exactly (handle padding)
    if full_tensor.shape[0] > dim0_size:
        full_tensor = full_tensor[:dim0_size]
    
    return full_tensor


def scatter_tensor(full_tensor, world_size=None, rank=None):
    """
    Scatter a full tensor to get the local shard for the current rank.
    Used during gradient reduce-scatter operations.
    
    Args:
        full_tensor: torch.Tensor, the full tensor to scatter
        world_size: int, number of ranks  
        rank: int, current rank
    
    Returns:
        sharded_tensor: torch.Tensor, the local shard
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()
        
    if len(full_tensor.shape) == 0:  # scalar tensor
        # For scalar, reduce to all ranks
        dist.all_reduce(full_tensor, op=dist.ReduceOp.SUM)
        return full_tensor / world_size
    
    # Calculate shard size for current rank
    dim0_size = full_tensor.shape[0]
    shard_size = (dim0_size + world_size - 1) // world_size
    start_idx = rank * shard_size
    end_idx = min(start_idx + shard_size, dim0_size)
    
    if start_idx >= dim0_size:
        # Empty shard for this rank
        shard_shape = (0,) + full_tensor.shape[1:]
        return torch.empty(shard_shape, dtype=full_tensor.dtype, device=full_tensor.device)
    
    # Extract local shard
    local_shard = full_tensor[start_idx:end_idx].clone()
    
    # Prepare shards for all ranks
    all_shards = []
    for r in range(world_size):
        r_start_idx = r * shard_size
        r_end_idx = min(r_start_idx + shard_size, dim0_size)
        if r_start_idx >= dim0_size:
            shard_shape = (0,) + full_tensor.shape[1:]
            shard = torch.empty(shard_shape, dtype=full_tensor.dtype, device=full_tensor.device)
        else:
            shard = full_tensor[r_start_idx:r_end_idx]
        all_shards.append(shard)
    
    # Perform reduce-scatter: sum gradients and scatter to corresponding ranks
    dist.reduce_scatter(local_shard, all_shards, op=dist.ReduceOp.SUM)
    
    return local_shard


if __name__ == "__main__":
    # Test the partition function
    from torchvision.models import vit_b_16
    
    # Simulate distributed environment
    print("Testing FSDP partition logic...")
    
    with torch.device('meta'):
        model = vit_b_16()
    
    # Test with 4 ranks
    world_size = 4
    for rank in range(world_size):
        print(f"\n--- Rank {rank} ---")
        sharded_tensors, full_shapes = fsdp_partition_tensors(
            OrderedDict(model.named_parameters()), 
            world_size=world_size,
            rank=rank,
            verbose=True
        ) 