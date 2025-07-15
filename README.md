# Tiny-FSDP: Minimal Distributed Training Framework

A lightweight, educational implementation of distributed deep learning strategies including **DDP**, **ZeRO-3**, and **FSDP** (Fully Sharded Data Parallel). This project provides clean, understandable implementations of modern distributed training techniques with a focus on clarity and learning.

## 🚀 Features

- **Three Distributed Strategies**: Complete implementations of DDP, ZeRO-3, and FSDP
- **Educational Focus**: Clean, well-documented code designed for learning and understanding
- **Production-Ready**: Efficient implementations with proper memory management and communication optimization
- **Modular Design**: Easy to extend and customize for different use cases
- **CUDA Optimized**: Built-in support for GPU acceleration and distributed computing
- **PyTorch Native**: Leverages PyTorch's distributed primitives for maximum compatibility

## 📋 Supported Strategies

| Strategy | Sharding Type | Memory Distribution | Communication Pattern | Best For |
|----------|---------------|-------------------|---------------------|----------|
| **DDP** | Gradient Only | Full model replication | All-reduce gradients | Small to medium models |
| **ZeRO-3** | Inter-tensor | Whole tensors per rank | Broadcast parameters | Large models with uneven layers |
| **FSDP** | Intra-tensor | Tensor slices per rank | All-gather/Reduce-scatter | Very large models, even distribution |

### Strategy Details

#### DDP (Distributed Data Parallel)
- **Memory**: Each rank holds a full copy of the model
- **Communication**: Gradients are all-reduced across ranks
- **Overhead**: Minimal, suitable for smaller models

#### ZeRO-3 (Zero Redundancy Optimizer - Stage 3)
- **Memory**: Parameters distributed across ranks (inter-tensor sharding)
- **Communication**: Broadcast parameters from owner, reduce gradients to owner
- **Overhead**: Dynamic parameter synchronization, good for heterogeneous workloads

#### FSDP (Fully Sharded Data Parallel)
- **Memory**: Each parameter tensor split along dim-0 across all ranks
- **Communication**: All-gather for forward/backward, reduce-scatter for gradients
- **Overhead**: Balanced load distribution, optimal for very large models

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU support)
- NCCL (for multi-GPU communication)

### Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install tqdm
```

### Clone Repository
```bash
git clone <repository-url>
cd Tiny-FSDP
```

## 🚀 Quick Start

### Single GPU Training
```python
import torch
from tiny_fsdp.core import SGD, AdamW
from example.model import GPT2Model, GPTConfigs

model = GPT2Model(GPTConfigs.gpt2)
optimizer = AdamW(model.named_parameters(), lr=1e-4)

# Standard training loop
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Distributed Training with DDP
```bash
torchrun --nproc_per_node=2 example/ddp/train.py
```

### Distributed Training with ZeRO-3
```bash
torchrun --nproc_per_node=2 example/zero3/train.py
```

### Distributed Training with FSDP
```bash
torchrun --nproc_per_node=2 example/fsdp/train.py
```

## 📚 Usage Examples

### DDP Example
```python
import torch.distributed as dist
from tiny_fsdp.core import DDP, DDPAdamW

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Wrap model with DDP
model = GPT2Model(config)
model = DDP(model)

# Use DDP-aware optimizer
optimizer = DDPAdamW(model.named_parameters(), lr=1e-4)

# Training loop
model.require_backward_grad_sync = True  # Enable gradient sync
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradients automatically all-reduced
    optimizer.step()
```

### ZeRO-3 Example
```python
from tiny_fsdp.core import Zero3, Zero3AdamW, zero3_partition_tensors

# Partition model parameters across ranks
with torch.device('meta'):
    model = GPT2Model(config)
    
parts, _ = zero3_partition_tensors(
    OrderedDict(model.named_parameters()),
    ranks_map=[f"cuda:{i}" for i in range(world_size)],
    evenness_priority=0
)

# Wrap with Zero3
model = GPT2Model(config)
model = Zero3(model, parts)

# Use Zero3-aware optimizer
optimizer = Zero3AdamW(
    model.module.named_parameters(), 
    lr=1e-4, 
    param_part_table=parts,
    ranks_map=[f"cuda:{i}" for i in range(world_size)]
)
```

### FSDP Example
```python
from tiny_fsdp.core import FSDP, FSDPAdamW

# Wrap model with FSDP (automatic parameter sharding)
model = GPT2Model(config)
model = FSDP(model, world_size=world_size, rank=rank)

# Use FSDP-aware optimizer (works with sharded parameters)
optimizer = FSDPAdamW(model.named_parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    output = model(batch)  # Parameters auto-gathered
    loss = criterion(output, target)
    loss.backward()        # Gradients auto-scattered
    optimizer.step()       # Update local shards
```

## 🏗️ Architecture

### Core Components

```
tiny_fsdp/core/
├── ddp/          # Distributed Data Parallel
├── zero3/        # ZeRO-3 Implementation  
├── fsdp/         # Fully Sharded Data Parallel
├── module/       # Base modules (Linear, LayerNorm, Embedding)
├── optim/        # Optimizers (SGD, AdamW)
└── utils/        # Utilities and helpers
```

### Module Structure
Each strategy implements:
- **Module Wrappers**: Custom Linear, LayerNorm, Embedding layers
- **Model Wrapper**: High-level model container
- **Optimizers**: Strategy-specific parameter update logic
- **Communication**: Efficient parameter and gradient synchronization

## 📊 Performance Characteristics

### Memory Usage (Approximate)

For a model with P parameters across N ranks:

| Strategy | Parameters/Rank | Gradients/Rank | Optimizer States/Rank |
|----------|----------------|----------------|----------------------|
| DDP | P | P | P |
| ZeRO-3 | P/N | P/N | P/N |
| FSDP | P/N | P/N | P/N |

### Communication Complexity

| Strategy | Forward Pass | Backward Pass | Optimizer Step |
|----------|-------------|---------------|----------------|
| DDP | None | All-reduce(P) | None |
| ZeRO-3 | Broadcast(P) | Reduce(P) | None |
| FSDP | All-gather(P) | Reduce-scatter(P) | None |

## 🔧 Advanced Configuration

### Custom Module Support

To add support for new PyTorch modules:

```python
# 1. Implement the module wrapper
class MyCustomModule(base_module.MyCustomModule):
    def forward_callback(self, ctx, *args):
        # Custom forward logic with parameter sync
        pass
        
    def backward_callback(self, ctx, grad_output):
        # Custom backward logic with gradient handling
        pass

# 2. Register in the strategy wrapper
_supported_modules = {
    nn.MyCustomModule: MyCustomModule,
    # ... other modules
}
```

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing = True

# Tune communication overlap
model.enable_async_communication = True

# Configure precision
model.half()  # Use FP16 for memory efficiency
```

## 🧪 Benchmarks

Tested on GPT-2 (117M parameters) across 2 RTX 4090s:

| Strategy | Memory/GPU | Training Speed | Convergence |
|----------|------------|----------------|-------------|
| DDP | ~2.1GB | 4.9 it/s | Baseline |
| ZeRO-3 | ~1.8GB | 4.2 it/s | Same as DDP |
| FSDP | ~1.8GB | 4.5 it/s | Same as DDP |

*Results may vary based on model architecture, batch size, and hardware configuration.*

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
pip install -e .
pip install pytest black flake8
```

### Running Tests
```bash
pytest tests/
```

## 📖 Educational Resources

This implementation is designed for learning. Key educational features:

- **Clear Code**: Well-commented, readable implementations
- **Minimal Dependencies**: Focus on core concepts without complexity
- **Comparative Analysis**: Easy to compare different strategies
- **Documentation**: Comprehensive docs and examples

## 🎓 Recommended Reading

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [FSDP in PyTorch](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- [Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the distributed training primitives
- Microsoft DeepSpeed for ZeRO innovation
- Meta for FSDP contributions to the community
- The broader open-source ML community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: See `README_FSDP.md` for detailed FSDP implementation notes

---

*Built with ❤️ for the distributed training community*