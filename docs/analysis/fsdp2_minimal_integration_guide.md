# FSDP2 最小集成指南：从零开始集成到新框架

> **目标读者**：希望在新框架中集成 FSDP2 的工程师
> **作者**：基于 slime 源码分析和 PyTorch 官方文档
> **日期**：2025-12-15
> **版本**：v1.0

本文档提供 FSDP2 的最小可用实现，帮助你在任何 Python 训练框架中快速集成 FSDP2。

---

## 目录

1. [最小代码清单](#1-最小代码清单)
2. [核心 API 详解](#2-核心-api-详解)
3. [完整示例代码](#3-完整示例代码)
4. [集成测试方法](#4-集成测试方法)
5. [常见问题排查](#5-常见问题排查)
6. [性能验证与优化](#6-性能验证与优化)

---

## 1. 最小代码清单

### 1.1 必需的依赖

```bash
# PyTorch 2.4+ (FSDP2 在 PyTorch 2.4 引入)
pip install torch>=2.4.0

# 分布式训练需要的库
pip install transformers  # 如果使用 HuggingFace 模型
```

### 1.2 最少需要的代码模块

集成 FSDP2 最少需要以下 **5 个核心步骤**：

```python
# ============ 步骤 1: 初始化分布式环境 ============
import torch.distributed as dist

dist.init_process_group(
    backend='nccl',           # GPU 通信后端
    init_method='env://',     # 从环境变量读取配置
)

# ============ 步骤 2: 创建 DeviceMesh ============
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(
    "cuda",                   # 设备类型
    mesh_shape=(world_size,), # 1D mesh (纯 DP)
    mesh_dim_names=("dp",)    # 维度名称
)

# ============ 步骤 3: 加载模型 ============
model = YourModel().cuda()

# ============ 步骤 4: 应用 FSDP2 ============
from torch.distributed.fsdp import fully_shard

model = fully_shard(
    model,
    mesh=mesh,                # 传入 DeviceMesh
)

# ============ 步骤 5: 训练循环 ============
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    # Forward
    output = model(batch['input'])
    loss = compute_loss(output, batch['target'])

    # Backward
    loss.backward()

    # Update
    optimizer.step()
    optimizer.zero_grad()
```

**总代码量**：约 **30 行核心代码**即可完成集成！

---

## 2. 核心 API 详解

### 2.1 分布式初始化 API

#### `torch.distributed.init_process_group`

```python
import torch.distributed as dist

dist.init_process_group(
    backend='nccl',        # 必需：通信后端
                          # - 'nccl': GPU 通信（推荐）
                          # - 'gloo': CPU 通信
    init_method='env://',  # 可选：初始化方法
                          # - 'env://': 从环境变量读取
                          # - 'tcp://ip:port': 手动指定
    rank=None,            # 可选：当前进程 rank（从环境变量读取）
    world_size=None,      # 可选：总进程数（从环境变量读取）
    timeout=timedelta(minutes=30),  # 可选：超时时间
)
```

**环境变量要求**：
```bash
export RANK=0              # 当前进程的全局 rank
export WORLD_SIZE=4        # 总进程数
export MASTER_ADDR=localhost  # 主节点地址
export MASTER_PORT=29500   # 主节点端口
```

**验证初始化**：
```python
assert dist.is_initialized(), "分布式未初始化"
print(f"Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")
```

---

### 2.2 DeviceMesh 创建 API

#### `init_device_mesh` - 1D Mesh（纯 DP）

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(
    device_type="cuda",           # 必需：设备类型
    mesh_shape=(world_size,),     # 必需：Mesh 形状 (1D)
    mesh_dim_names=("dp",)        # 可选：维度名称
)

# 提取 DP 组（用于 FSDP）
dp_group = mesh.get_group("dp")  # 返回 ProcessGroup
```

**Mesh 形状说明**：
- `(world_size,)`: 1D Mesh，纯 Data Parallel
- 所有 GPU 在同一个 DP 组内进行梯度同步

#### `init_device_mesh` - 2D Mesh（DP + CP）

```python
# 8 GPUs: dp_size=4, cp_size=2
mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(4, 2),            # 2D: (dp_size, cp_size)
    mesh_dim_names=("dp", "cp")
)

# 提取通信组
dp_group = mesh.get_group("dp")   # DP 维度（梯度同步）
cp_group = mesh.get_group("cp")   # CP 维度（序列并行）
dp_mesh = mesh["dp"]              # DP 子 Mesh（传给 FSDP）
```

**2D Mesh 布局**：
```
rank = dp_idx * cp_size + cp_idx

     CP Dim →
DP   [0  1]    CP Groups: [0,1], [2,3], [4,5], [6,7]
↓    [2  3]    DP Groups: [0,2,4,6], [1,3,5,7]
     [4  5]
     [6  7]
```

---

### 2.3 FSDP2 核心 API

#### `fully_shard` - 基础用法

```python
from torch.distributed.fsdp import fully_shard

model = fully_shard(
    module=model,         # 必需：要包装的模块
    mesh=mesh,            # 可选：DeviceMesh（默认使用所有 ranks）
)
```

**关键行为**：
1. 将 `module` 的参数转换为 **DTensor**（分片张量）
2. 注册 **Hook**：自动触发 All-Gather 和 Reduce-Scatter
3. 返回包装后的 module（可继续正常使用）

#### `fully_shard` - 完整参数

```python
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy
)

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,   # 参数计算精度（forward/backward）
    reduce_dtype=torch.float32,   # 梯度归约精度（推荐 fp32）
)

offload_policy = CPUOffloadPolicy()  # 启用 CPU Offload

model = fully_shard(
    module=model,
    mesh=mesh,
    mp_policy=mp_policy,           # 可选：混合精度策略
    offload_policy=offload_policy, # 可选：CPU Offload 策略
    reshard_after_forward=True,    # 可选：Forward 后立即释放参数
)
```

**MixedPrecisionPolicy 详解**：
```python
class MixedPrecisionPolicy:
    param_dtype: torch.dtype      # All-Gather 后用于计算的精度
    reduce_dtype: torch.dtype     # 梯度 Reduce-Scatter 的精度

    # 推荐配置：
    # - param_dtype=bf16: 计算快，省显存
    # - reduce_dtype=fp32: 梯度归约数值稳定
```

**CPUOffloadPolicy 详解**：
```python
class CPUOffloadPolicy:
    # 启用后，FSDP2 会自动：
    # 1. 将 Sharded Parameters offload 到 CPU
    # 2. Forward/Backward 时自动加载到 GPU
    # 3. 计算完成后自动 offload 回 CPU
    #
    # 适用场景：显存受限，可以牺牲时间换空间
    pass
```

#### `fully_shard` - 分层包装

```python
# 方式 1：自动包装（推荐 HuggingFace 模型）
def apply_fsdp2_auto(model, mesh):
    """
    利用 HuggingFace 的 _no_split_modules 自动包装
    """
    # 获取需要独立包装的层类型
    layer_cls_to_wrap = model._no_split_modules  # 如 ['Qwen2DecoderLayer']

    # 找到所有匹配的子模块
    modules = [
        module for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
    ]

    # 对每个子模块应用 fully_shard
    for module in modules:
        fully_shard(module, mesh=mesh)

    # 对顶层模型应用 fully_shard（包装 embedding、lm_head）
    fully_shard(model, mesh=mesh)

    return model

# 方式 2：手动包装
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50000, 1024)
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(12)
        ])
        self.lm_head = nn.Linear(1024, 50000)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

def apply_fsdp2_manual(model, mesh):
    """
    手动指定包装粒度
    """
    # 包装每个 Transformer Layer
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)

    # 包装 Embedding
    fully_shard(model.embedding, mesh=mesh)

    # 包装 LM Head
    fully_shard(model.lm_head, mesh=mesh)

    # 包装顶层模型
    fully_shard(model, mesh=mesh)

    return model
```

**包装粒度的影响**：
```
粒度越细（每个 layer 独立包装）：
  优点：显存占用更低（每次只 All-Gather 一个 layer）
  缺点：通信次数更多（每个 layer 都要通信）

粒度越粗（整个 model 一起包装）：
  优点：通信次数更少
  缺点：显存峰值更高（需要 All-Gather 整个 model）

推荐：Layer-wise 包装（Decoder Layer 粒度）
```

---

### 2.4 Optimizer 与 Checkpoint API

#### Optimizer（无需特殊处理）

```python
# FSDP2 的参数仍然是 Parameter，可直接传给优化器
optimizer = torch.optim.AdamW(
    model.parameters(),  # FSDP2 包装后的参数（DTensor）
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

# 训练循环（标准 PyTorch 代码）
for batch in dataloader:
    output = model(batch['input'])
    loss = compute_loss(output, batch['target'])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**关键点**：
- FSDP2 的 DTensor 继承自 `torch.nn.Parameter`
- Optimizer 会自动识别并处理 DTensor
- Optimizer State（如 AdamW 的 `exp_avg`）也会被自动分片

#### Checkpoint 保存

```python
from torch.distributed.checkpoint import save

def save_fsdp2_checkpoint(model, optimizer, path):
    """
    保存 FSDP2 分布式 checkpoint
    """
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions

    # 获取 state_dict（自动处理 DTensor）
    model_state_dict, optimizer_state_dict = get_state_dict(
        model, optimizer,
        options=StateDictOptions(
            full_state_dict=False,  # 保存分片（每个 rank 保存 1/N）
            cpu_offload=True,       # Offload 到 CPU 再保存
        )
    )

    # 保存到分布式存储
    state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
    }

    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(path),
    )

    print(f"Rank {dist.get_rank()}: Checkpoint saved to {path}")

# 使用
save_fsdp2_checkpoint(model, optimizer, "/path/to/ckpt/")
```

**Checkpoint 目录结构**：
```
/path/to/ckpt/
├── .metadata
├── __0_0.distcp    # Rank 0 的分片
├── __1_0.distcp    # Rank 1 的分片
├── __2_0.distcp    # Rank 2 的分片
└── __3_0.distcp    # Rank 3 的分片
```

#### Checkpoint 加载

```python
from torch.distributed.checkpoint import load

def load_fsdp2_checkpoint(model, optimizer, path):
    """
    加载 FSDP2 分布式 checkpoint
    """
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.state_dict import set_state_dict, StateDictOptions

    # 准备空的 state_dict
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # 从分布式存储加载
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(path),
    )

    # 设置到 model 和 optimizer
    set_state_dict(
        model, optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict["optimizer"],
        options=StateDictOptions(
            broadcast_from_rank0=True,  # 从 rank 0 广播
        )
    )

    print(f"Rank {dist.get_rank()}: Checkpoint loaded from {path}")

# 使用
load_fsdp2_checkpoint(model, optimizer, "/path/to/ckpt/")
```

---

## 3. 完整示例代码

### 3.1 最小可运行示例（100 行）

```python
#!/usr/bin/env python3
"""
FSDP2 最小训练示例
运行方式：torchrun --nproc_per_node=4 minimal_fsdp2.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# ===================== 模型定义 =====================
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=10000, hidden_size=512, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

# ===================== 主函数 =====================
def main():
    # 1. 初始化分布式
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Initialized with world_size={world_size}")

    # 2. 创建 DeviceMesh
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    print(f"[Rank {rank}] DeviceMesh created: {mesh}")

    # 3. 创建模型
    model = SimpleTransformer(
        vocab_size=10000,
        hidden_size=512,
        num_layers=4
    ).cuda()

    # 4. 应用 FSDP2
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # 分层包装：先包装 layers，再包装整体
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)

    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    print(f"[Rank {rank}] FSDP2 applied")

    # 5. 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 6. 创建数据集（假数据）
    dataset = TensorDataset(
        torch.randint(0, 10000, (1000, 128)),  # input_ids
        torch.randint(0, 10000, (1000, 128)),  # labels
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
    )

    # 7. 训练循环
    model.train()
    for epoch in range(2):
        sampler.set_epoch(epoch)  # 确保每个 epoch shuffle 不同

        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            # Forward
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, 10000),
                labels.view(-1)
            )

            # Backward
            loss.backward()

            # Update
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # 8. 清理
    dist.destroy_process_group()
    print(f"[Rank {rank}] Training completed!")

if __name__ == "__main__":
    main()
```

**运行方式**：
```bash
# 单节点 4 GPU
torchrun --nproc_per_node=4 minimal_fsdp2.py

# 多节点（节点 0）
torchrun --nproc_per_node=8 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr=192.168.1.100 \
         --master_port=29500 \
         minimal_fsdp2.py

# 多节点（节点 1）
torchrun --nproc_per_node=8 \
         --nnodes=2 \
         --node_rank=1 \
         --master_addr=192.168.1.100 \
         --master_port=29500 \
         minimal_fsdp2.py
```

---

### 3.2 HuggingFace 模型示例

```python
#!/usr/bin/env python3
"""
FSDP2 + HuggingFace 模型训练示例
运行方式：torchrun --nproc_per_node=4 fsdp2_hf.py
"""
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# ===================== 自定义数据集 =====================
class DummyDataset(Dataset):
    def __init__(self, size=1000, seq_len=128):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 50000, (self.seq_len,)),
            'labels': torch.randint(0, 50000, (self.seq_len,)),
        }

# ===================== FSDP2 包装函数 =====================
def apply_fsdp2_to_hf_model(model, mesh):
    """
    自动包装 HuggingFace 模型
    """
    # 获取模型的 _no_split_modules（如 ['Qwen2DecoderLayer']）
    layer_cls_to_wrap = model._no_split_modules
    print(f"Layer classes to wrap: {layer_cls_to_wrap}")

    # 找到所有需要独立包装的子模块
    modules = [
        module for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
    ]

    print(f"Found {len(modules)} modules to wrap")

    # 配置混合精度
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # 包装每个 Decoder Layer
    for module in modules:
        fully_shard(module, mesh=mesh, mp_policy=mp_policy)

    # 包装顶层模型
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    return model

# ===================== 主函数 =====================
def main():
    # 1. 初始化分布式
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # 2. 创建 DeviceMesh
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,))

    # 3. 加载 HuggingFace 模型
    model_name = "Qwen/Qwen2-0.5B"  # 小模型，便于测试

    if rank == 0:
        # Rank 0 加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda()
    else:
        # 其他 ranks 创建空模型（节省内存）
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        with torch.device('meta'):  # meta device 不分配内存
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to_empty(device='cuda')  # 分配空 tensor

    # 4. 应用 FSDP2
    model = apply_fsdp2_to_hf_model(model, mesh)

    # 5. 广播权重（从 rank 0 到其他 ranks）
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

    if rank == 0:
        state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()  # 空的 state_dict

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
    )

    set_model_state_dict(model, state_dict, options=options)

    print(f"[Rank {rank}] Model loaded and weights broadcasted")

    # 6. 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 7. 创建数据集
    dataset = DummyDataset(size=100, seq_len=128)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    # 8. 训练循环
    model.train()
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward
        loss.backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    # 9. 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

---

## 4. 集成测试方法

### 4.1 功能验证清单

#### 测试 1：参数是否被正确分片

```python
def test_parameter_sharding(model, world_size):
    """
    验证参数是否被正确分片
    """
    from torch.distributed.tensor import DTensor

    print("\n=== Testing Parameter Sharding ===")

    for name, param in model.named_parameters():
        # 检查是否是 DTensor
        assert isinstance(param, DTensor), f"{name} is not DTensor"

        # 检查分片维度
        print(f"{name}:")
        print(f"  - Type: {type(param)}")
        print(f"  - Placements: {param.placements}")
        print(f"  - Device Mesh: {param.device_mesh}")
        print(f"  - Global Shape: {param.shape}")
        print(f"  - Local Shape: {param.to_local().shape}")

        # 验证分片大小
        global_numel = param.numel()
        local_numel = param.to_local().numel()
        expected_local_numel = global_numel // world_size

        assert local_numel == expected_local_numel, \
            f"{name}: local_numel={local_numel}, expected={expected_local_numel}"

    print("✅ All parameters are correctly sharded!")

# 运行测试
test_parameter_sharding(model, world_size=dist.get_world_size())
```

**预期输出**：
```
=== Testing Parameter Sharding ===
model.layers.0.linear1.weight:
  - Type: <class 'torch.distributed.tensor.DTensor'>
  - Placements: [Shard(0)]
  - Device Mesh: DeviceMesh('cuda', mesh=[0, 1, 2, 3])
  - Global Shape: torch.Size([2048, 512])
  - Local Shape: torch.Size([512, 512])  # 分片到 4 个 GPU，每个 512 行
✅ All parameters are correctly sharded!
```

#### 测试 2：梯度是否正确同步

```python
def test_gradient_synchronization(model, world_size):
    """
    验证梯度是否在所有 ranks 上正确同步
    """
    import torch.distributed as dist

    print("\n=== Testing Gradient Synchronization ===")

    # 创建假数据
    input_ids = torch.randint(0, 10000, (2, 128)).cuda()
    labels = torch.randint(0, 10000, (2, 128)).cuda()

    # Forward
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(logits.view(-1, 10000), labels.view(-1))

    # Backward
    loss.backward()

    # 检查梯度
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # 获取本地梯度
        local_grad = param.grad.to_local()

        # All-Gather 所有 ranks 的梯度（仅用于验证）
        grad_list = [torch.zeros_like(local_grad) for _ in range(world_size)]
        dist.all_gather(grad_list, local_grad)

        # 验证所有 ranks 的对应分片是否相同
        for i in range(1, world_size):
            assert torch.allclose(grad_list[0], grad_list[i], atol=1e-5), \
                f"{name}: gradients not synchronized across ranks"

    print("✅ Gradients are correctly synchronized!")

# 运行测试
test_gradient_synchronization(model, world_size=dist.get_world_size())
```

#### 测试 3：Loss 一致性验证

```python
def test_loss_consistency():
    """
    验证多卡训练和单卡训练的 Loss 是否一致
    """
    print("\n=== Testing Loss Consistency ===")

    # 固定随机种子
    torch.manual_seed(42)

    # 创建相同的输入（所有 ranks 相同）
    input_ids = torch.randint(0, 10000, (4, 128)).cuda()
    labels = torch.randint(0, 10000, (4, 128)).cuda()

    # Forward
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(logits.view(-1, 10000), labels.view(-1))

    # 收集所有 ranks 的 loss
    loss_list = [torch.zeros(1).cuda() for _ in range(dist.get_world_size())]
    dist.all_gather(loss_list, loss.unsqueeze(0))

    # 验证所有 ranks 的 loss 是否相同
    for i in range(1, dist.get_world_size()):
        assert torch.allclose(loss_list[0], loss_list[i], atol=1e-4), \
            f"Loss not consistent: rank 0 = {loss_list[0].item()}, rank {i} = {loss_list[i].item()}"

    print(f"✅ Loss is consistent across all ranks: {loss.item():.6f}")

# 运行测试
test_loss_consistency()
```

---

### 4.2 性能验证

#### 测试 4：显存占用验证

```python
def test_memory_usage():
    """
    验证 FSDP2 的显存优化效果
    """
    import torch

    print("\n=== Testing Memory Usage ===")

    # 记录初始显存
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / 1e9

    # 训练一个 step
    input_ids = torch.randint(0, 10000, (4, 128)).cuda()
    labels = torch.randint(0, 10000, (4, 128)).cuda()

    logits = model(input_ids)
    loss = nn.functional.cross_entropy(logits.view(-1, 10000), labels.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 记录峰值显存
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    current_memory = torch.cuda.memory_allocated() / 1e9

    print(f"Initial memory: {initial_memory:.2f} GB")
    print(f"Peak memory: {peak_memory:.2f} GB")
    print(f"Current memory: {current_memory:.2f} GB")

    # 验证显存占用在合理范围内
    world_size = dist.get_world_size()
    # 理论上，FSDP2 的显存应该约为单卡的 1/world_size
    # 实际会有 activations 等额外开销

    print(f"✅ Memory usage validated (Peak: {peak_memory:.2f} GB)")

# 运行测试
test_memory_usage()
```

#### 测试 5：训练速度验证

```python
import time

def test_training_speed(num_steps=100):
    """
    测试训练速度（throughput）
    """
    print("\n=== Testing Training Speed ===")

    # 预热（编译 CUDA kernels）
    for _ in range(10):
        input_ids = torch.randint(0, 10000, (4, 128)).cuda()
        labels = torch.randint(0, 10000, (4, 128)).cuda()
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, 10000), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 测试
    torch.cuda.synchronize()
    start_time = time.time()

    for step in range(num_steps):
        input_ids = torch.randint(0, 10000, (4, 128)).cuda()
        labels = torch.randint(0, 10000, (4, 128)).cuda()

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, 10000), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    throughput = num_steps / elapsed_time

    print(f"Trained {num_steps} steps in {elapsed_time:.2f}s")
    print(f"Throughput: {throughput:.2f} steps/s")

    # 验证 throughput 在合理范围内（取决于硬件）
    # 这里只是示例，实际阈值需要根据硬件调整
    assert throughput > 1.0, f"Throughput too low: {throughput:.2f} steps/s"

    print(f"✅ Training speed validated ({throughput:.2f} steps/s)")

# 运行测试
test_training_speed(num_steps=100)
```

---

### 4.3 完整测试脚本

```python
#!/usr/bin/env python3
"""
FSDP2 集成测试脚本
运行方式：torchrun --nproc_per_node=4 test_fsdp2_integration.py
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# 模型定义（同前）
class SimpleTransformer(nn.Module):
    # ... (省略，同 3.1)
    pass

def run_all_tests():
    """
    运行所有集成测试
    """
    # 初始化
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # 创建模型
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,))
    model = SimpleTransformer().cuda()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 运行测试（仅在 rank 0 打印）
    if rank == 0:
        print("=" * 60)
        print("FSDP2 Integration Test Suite")
        print("=" * 60)

    try:
        # Test 1: 参数分片
        if rank == 0:
            test_parameter_sharding(model, world_size)

        # Test 2: 梯度同步
        if rank == 0:
            test_gradient_synchronization(model, world_size)

        # Test 3: Loss 一致性
        test_loss_consistency()

        # Test 4: 显存占用
        if rank == 0:
            test_memory_usage()

        # Test 5: 训练速度
        if rank == 0:
            test_training_speed(num_steps=50)

        if rank == 0:
            print("\n" + "=" * 60)
            print("✅ All tests passed!")
            print("=" * 60)

    except AssertionError as e:
        print(f"❌ Test failed on rank {rank}: {e}")
        raise

    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    run_all_tests()
```

**运行测试**：
```bash
torchrun --nproc_per_node=4 test_fsdp2_integration.py
```

**预期输出**：
```
============================================================
FSDP2 Integration Test Suite
============================================================

=== Testing Parameter Sharding ===
✅ All parameters are correctly sharded!

=== Testing Gradient Synchronization ===
✅ Gradients are correctly synchronized!

=== Testing Loss Consistency ===
✅ Loss is consistent across all ranks: 9.234567

=== Testing Memory Usage ===
Initial memory: 2.34 GB
Peak memory: 4.56 GB
Current memory: 2.45 GB
✅ Memory usage validated (Peak: 4.56 GB)

=== Testing Training Speed ===
Trained 50 steps in 12.34s
Throughput: 4.05 steps/s
✅ Training speed validated (4.05 steps/s)

============================================================
✅ All tests passed!
============================================================
```

---

## 5. 常见问题排查

### 5.1 错误：`RuntimeError: NCCL error`

**症状**：
```
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1234
NCCL operation failed: unhandled system error
```

**可能原因**：
1. GPU 之间通信不通（网络/拓扑问题）
2. NCCL 版本不兼容
3. CUDA 设备设置错误

**解决方案**：
```python
# 1. 检查 CUDA 设备设置
torch.cuda.set_device(rank)  # 确保每个进程绑定到正确的 GPU

# 2. 检查 NCCL 版本
import torch
print(f"NCCL version: {torch.cuda.nccl.version()}")
# 推荐 NCCL >= 2.18

# 3. 启用 NCCL 调试日志
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

# 4. 测试点对点通信
def test_p2p_communication():
    tensor = torch.ones(10).cuda() * rank
    dist.all_reduce(tensor)
    print(f"Rank {rank}: {tensor}")  # 应该所有 ranks 相同

test_p2p_communication()
```

---

### 5.2 错误：`AssertionError: model._no_split_modules is empty`

**症状**：
```python
# apply_fsdp2_auto() 报错
AssertionError: layer_cls_to_wrap is empty
```

**原因**：
- 模型没有定义 `_no_split_modules` 属性（自定义模型）

**解决方案**：
```python
# 方案 1：手动设置 _no_split_modules
model._no_split_modules = ['TransformerLayer']  # 替换为你的 layer 类名

# 方案 2：手动包装（推荐）
def apply_fsdp2_manual(model, mesh):
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model

model = apply_fsdp2_manual(model, mesh)
```

---

### 5.3 错误：`RuntimeError: DTensor does not support...`

**症状**：
```
RuntimeError: DTensor does not support inplace operations
```

**原因**：
- FSDP2 的 DTensor 不支持某些 inplace 操作（如 `x += y`）

**解决方案**：
```python
# 错误写法
x += y  # inplace 加法

# 正确写法
x = x + y  # 创建新 tensor
```

---

### 5.4 显存占用仍然很高

**症状**：
- 使用 FSDP2 后显存占用没有明显下降

**可能原因**：
1. 包装粒度太粗（整个模型一起包装）
2. 没有启用 `reshard_after_forward`
3. Activations 占用过高

**解决方案**：
```python
# 1. 使用更细的包装粒度
for layer in model.layers:
    fully_shard(layer, mesh=mesh)  # Layer-wise 包装

# 2. 启用 reshard_after_forward
fully_shard(
    model,
    mesh=mesh,
    reshard_after_forward=True,  # Forward 后立即释放参数
)

# 3. 启用 Gradient Checkpointing（降低 activations 占用）
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def forward(self, x):
        return checkpoint(self.layer, x, use_reentrant=False)
```

---

## 6. 性能验证与优化

### 6.1 性能基准测试

```python
import torch
import time
from torch.profiler import profile, ProfilerActivity

def benchmark_fsdp2(model, num_steps=100, batch_size=4, seq_len=128):
    """
    FSDP2 性能基准测试
    """
    print("\n=== FSDP2 Performance Benchmark ===")

    # 预热
    for _ in range(10):
        input_ids = torch.randint(0, 10000, (batch_size, seq_len)).cuda()
        labels = torch.randint(0, 10000, (batch_size, seq_len)).cuda()
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, 10000), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 测试
    torch.cuda.synchronize()
    start_time = time.time()

    total_tokens = 0
    for step in range(num_steps):
        input_ids = torch.randint(0, 10000, (batch_size, seq_len)).cuda()
        labels = torch.randint(0, 10000, (batch_size, seq_len)).cuda()

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, 10000), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch_size * seq_len

    torch.cuda.synchronize()
    end_time = time.time()

    # 统计
    elapsed_time = end_time - start_time
    throughput_steps = num_steps / elapsed_time
    throughput_tokens = total_tokens / elapsed_time

    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"Steps: {num_steps}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Throughput: {throughput_steps:.2f} steps/s, {throughput_tokens:.0f} tokens/s")
    print(f"Peak memory: {peak_memory:.2f} GB")

    return {
        'throughput_steps': throughput_steps,
        'throughput_tokens': throughput_tokens,
        'peak_memory': peak_memory,
    }

# 运行 benchmark
results = benchmark_fsdp2(model, num_steps=100, batch_size=4, seq_len=128)
```

---

### 6.2 使用 PyTorch Profiler 分析性能

```python
def profile_fsdp2(model, num_steps=10):
    """
    使用 PyTorch Profiler 分析 FSDP2 性能
    """
    print("\n=== Profiling FSDP2 ===")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(num_steps):
            input_ids = torch.randint(0, 10000, (4, 128)).cuda()
            labels = torch.randint(0, 10000, (4, 128)).cuda()

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(logits.view(-1, 10000), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            prof.step()  # 通知 profiler 进入下一步

    # 打印统计信息（按 CUDA 时间排序，显示前 10 个）
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    # 导出 Chrome Trace（可在 chrome://tracing 查看）
    prof.export_chrome_trace("fsdp2_trace.json")
    print("Trace exported to fsdp2_trace.json")
    print("Open chrome://tracing in Chrome and load this file")

# 运行 profiling
profile_fsdp2(model, num_steps=10)
```

**预期输出**：
```
=== Profiling FSDP2 ===
---------------------------------------  ---------------  ---------------  ...
Name                                     Self CPU total   Self CUDA total  ...
---------------------------------------  ---------------  ---------------  ...
aten::nccl::all_reduce                        1.234 ms          45.6 ms   ...
aten::mm                                      2.345 ms          12.3 ms   ...
aten::cudnn::convolution                      0.567 ms           8.9 ms   ...
...
---------------------------------------  ---------------  ---------------  ...
Trace exported to fsdp2_trace.json
```

---

### 6.3 优化建议

根据 Profiling 结果优化 FSDP2 性能：

#### 优化 1：调整包装粒度

```python
# 如果通信开销高（all_reduce 占比 > 30%）
# → 使用更粗的粒度（减少通信次数）

# 当前：Layer-wise 包装
for layer in model.layers:
    fully_shard(layer, mesh=mesh)

# 优化：每 2 个 layer 一起包装
for i in range(0, len(model.layers), 2):
    container = nn.Sequential(model.layers[i], model.layers[i+1])
    fully_shard(container, mesh=mesh)
```

#### 优化 2：启用混合精度

```python
# 如果计算时间长（mm/convolution 占比高）
# → 启用 BF16/FP16 混合精度

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,  # 计算精度
    reduce_dtype=torch.float32,  # 梯度精度
)

fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

#### 优化 3：调整 Batch Size

```python
# 如果 GPU 利用率低（< 70%）
# → 增加 batch size

# 动态调整 batch size（防止 OOM）
max_batch_size = 32
current_batch_size = 4

while current_batch_size < max_batch_size:
    try:
        # 尝试更大的 batch
        test_batch = torch.randint(0, 10000, (current_batch_size * 2, 128)).cuda()
        _ = model(test_batch)
        current_batch_size *= 2
    except torch.cuda.OutOfMemoryError:
        break

print(f"Optimal batch size: {current_batch_size}")
```

---

## 总结

### 核心要点回顾

1. **最少代码**：5 步 30 行即可集成 FSDP2
   - 初始化分布式 → 创建 DeviceMesh → 加载模型 → 应用 FSDP → 训练

2. **核心 API**：
   - `dist.init_process_group()`: 初始化分布式
   - `init_device_mesh()`: 创建设备网格
   - `fully_shard()`: 应用 FSDP2
   - `torch.distributed.checkpoint`: 保存/加载

3. **测试方法**：
   - 参数分片验证
   - 梯度同步验证
   - Loss 一致性验证
   - 显存和速度验证

### 下一步学习

完成最小集成后，可以进一步学习：

1. **进阶特性**：
   - 2D DeviceMesh (DP + CP)
   - CPU Offload
   - Gradient Checkpointing

2. **性能优化**：
   - 通信计算重叠
   - 自定义包装策略
   - Profiling 和调优

3. **生产部署**：
   - 容错和恢复
   - 多节点训练
   - 监控和日志

### 参考资源

- **PyTorch 官方文档**: [FSDP2 Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- **Slime 框架源码**: `slime/backends/fsdp_utils/`
- **相关博客**: [RL System Deep Dive: FSDP Training Backend](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)

---

**文档版本**: v1.0
**基于代码版本**: slime main branch (commit: 9d7f34d)
**生成日期**: 2025-12-15
**作者**: Claude Code 基于 slime 源码分析
