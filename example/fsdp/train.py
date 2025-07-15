# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
from collections import OrderedDict

from example.model import GPTConfigs, GPT2Model
from tiny_fsdp.core import FSDPAdamW, FSDP

# init distributed
rank = int(os.getenv('LOCAL_RANK', '0'))
torch.manual_seed(rank)
world_size = int(os.getenv('WORLD_SIZE', '1'))
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(rank)

config = GPTConfigs.gpt2

# Create model and wrap with FSDP
input = torch.randint(0, config.vocab_size, (1, config.block_size)).to(rank)
target = torch.randint(0, config.vocab_size, (1, config.block_size)).to(rank)

# Create model on CPU first, then wrap with FSDP
model = GPT2Model(config)
model = FSDP(model, world_size=world_size, rank=rank)

# Create FSDP optimizer - it will work with sharded parameters
optimizer = FSDPAdamW(model.module.named_parameters(), lr=1e-5, weight_decay=1e-1)

if rank == 0:
    print(f"FSDP training with {world_size} ranks")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters per rank (sharded)")

for i in tqdm(range(100), disable=rank!=0):
    # Forward pass
    _, loss = model(input, target)
    
    # Backward pass - gradients are automatically reduce-scattered
    loss.backward()
    
    # Update parameters on sharded gradients
    optimizer.step()
    
    # All-reduce loss for logging
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    
    if rank == 0: 
        tqdm.write(f"iter {i} loss: {loss.item():.4f}")

if rank == 0:
    print("Training completed!")

dist.destroy_process_group() 