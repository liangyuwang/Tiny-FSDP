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
from tiny_fsdp.core import Zero3SGD, Zero3AdamW, Zero3
from tiny_fsdp.core import zero3_partition_tensors

# init distributed
rank = int(os.getenv('LOCAL_RANK', '0'))
torch.manual_seed(rank)
world_size = int(os.getenv('WORLD_SIZE', '1'))
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(rank)

config = GPTConfigs.gpt2
ranks_map = [f"cuda:{i}" for i in range(world_size)]
with torch.device('meta'):
    model = GPT2Model(config)
    parts, _ = zero3_partition_tensors(OrderedDict(model.named_parameters()), 
                                    ranks_map=ranks_map,
                                    evenness_priority=0,
                                    verbose=True)

input = torch.randint(0, config.vocab_size, (1, config.block_size)).to(rank)
target = torch.randint(0, config.vocab_size, (1, config.block_size)).to(rank)
model = GPT2Model(config)  # Create model on CPU first
model = Zero3(model, parts)  # Then wrap with Zero3
optimizer = Zero3AdamW(model.module.named_parameters(), lr=1e-5, weight_decay=1e-1, param_part_table=parts, ranks_map=ranks_map)

for i in tqdm(range(100), disable=rank!=0):
    model.require_backward_grad_sync = True # set to True when need grad all reduce
    _, loss = model(input, target)
    loss.backward()
    optimizer.step()
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    if rank==0: tqdm.write(f"iter {i} loss: {loss.item():.4f}")

dist.destroy_process_group()
