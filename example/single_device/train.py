# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import argparse

from example.model import GPTConfigs, GPT2Model
from tiny_fsdp.core import SGD, AdamW

# Parse arguments
parser = argparse.ArgumentParser(description='Single Device Training')
parser.add_argument('--model', choices=['gpt2', 'gpt2_medium', 'gpt2_large', 'gpt2_xl'], default='gpt2', help='Model size')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--steps', type=int, default=100, help='Training steps')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay')
args = parser.parse_args()

# init single device
torch.manual_seed(0)
torch.cuda.set_device(0)
device = torch.device("cuda:0")

config = getattr(GPTConfigs, args.model)
input = torch.randint(0, config.vocab_size, (1, config.block_size)).to(device)
target = torch.randint(0, config.vocab_size, (1, config.block_size)).to(device)
model = GPT2Model(config).to(device)
optimizer = AdamW(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)

print(f"Single device training - Model: {args.model}, LR: {args.lr}, Steps: {args.steps}")

for i in range(args.steps):
    _, loss = model(input, target)
    loss.backward()
    optimizer.step()
    print(f"iter {i} loss: {loss.item():.4f}")

