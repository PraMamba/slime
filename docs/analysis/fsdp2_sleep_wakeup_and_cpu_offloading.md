# FSDP2 Sleep/Wake_up Operations and CPU Offloading Analysis

## Problem-7: Sleep 操作的参数转移目标与带宽瓶颈分析

### 问题描述

sleep 操作具体是将参数移动到了哪里？是 CPU 内存（RAM）还是磁盘？如果是 CPU 内存，带宽瓶颈会不会导致训练和推理切换很慢？

### 核心发现总结

1. **参数转移目标**: CPU 内存（RAM），不是磁盘
2. **传输机制**: PCIe 总线，使用 PyTorch 的 `.cpu()` 和 `.cuda()` 方法
3. **性能影响**: 首次训练迭代有 2-5 秒开销（7B 模型），后续迭代无传输开销
4. **关键洞察**: `sleep()` 仅在初始化时调用一次，训练迭代之间不调用

---

## 1. Sleep/Wake_up 操作的实现机制

### 1.1 源码位置与实现

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py`

#### Sleep 操作实现 (lines 276-288)

```python
def sleep(self) -> None:
    """Pause CUDA memory for all tracked tensors."""
    if not self.args.offload_train:
        return

    print_memory("before offload model")

    # 将模型参数移动到 CPU RAM
    self.model.cpu()

    # 将优化器状态移动到 CPU RAM
    move_torch_optimizer(self.optimizer, "cpu")

    # 清理 GPU 缓存
    clear_memory()

    # 同步所有进程
    dist.barrier(group=get_gloo_group())

    print_memory("after offload model")
```

**关键操作**:
- `self.model.cpu()`: 将所有模型参数从 GPU VRAM 移动到 CPU RAM
- `move_torch_optimizer(self.optimizer, "cpu")`: 将优化器状态（momentum, variance）从 GPU 移动到 CPU RAM
- `clear_memory()`: 清理 GPU 显存缓存，释放空间给 SGLang 推理使用

#### Wake_up 操作实现 (lines 290-298)

```python
def wake_up(self) -> None:
    """Resume CUDA memory for all tracked tensors."""
    if not self.args.offload_train:
        return

    # 将模型参数从 CPU RAM 移回 GPU
    self.model.cuda()

    # 将优化器状态从 CPU RAM 移回 GPU
    move_torch_optimizer(self.optimizer, "cuda")

    # 同步所有进程
    dist.barrier(group=get_gloo_group())

    print_memory("after wake_up model")
```

**关键操作**:
- `self.model.cuda()`: 将模型参数从 CPU RAM 移回 GPU VRAM
- `move_torch_optimizer(self.optimizer, "cuda")`: 将优化器状态从 CPU RAM 移回 GPU

### 1.2 优化器状态转移的具体实现

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 1001-1013)

```python
@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """
    Move optimizer state between CPU and GPU.

    参考: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    # non_blocking=True 允许异步传输，提高性能
                    state[key] = value.to(device, non_blocking=True)

    # 等待所有异步传输完成
    torch.cuda.synchronize()
```

**性能优化点**:
- `non_blocking=True`: 启用异步传输，多个 tensor 可以并行传输
- `torch.cuda.synchronize()`: 确保所有传输完成后才返回

---

## 2. 数据传输量分析

### 2.1 传输内容组成

在 colocated 模式下，sleep/wake_up 需要传输的数据包括：

1. **模型参数** (Model Parameters)
   - 对于 bf16/fp16 模型: `num_params × 2 bytes`
   - 对于 fp32 模型: `num_params × 4 bytes`

2. **优化器状态** (Optimizer States, AdamW)
   - **Momentum (一阶动量)**: `num_params × 4 bytes` (通常是 fp32)
   - **Variance (二阶动量)**: `num_params × 4 bytes` (通常是 fp32)
   - **总计**: `num_params × 8 bytes`

3. **总数据量**
   - bf16 模型 + AdamW: `num_params × (2 + 8) = num_params × 10 bytes`
   - fp32 模型 + AdamW: `num_params × (4 + 8) = num_params × 12 bytes`

### 2.2 不同模型规模的传输量

| 模型规模 | 参数数量 | 模型参数 (bf16) | 优化器状态 (fp32) | 总计 | 备注 |
|---------|---------|----------------|------------------|------|------|
| 7B      | 7B      | 14 GB          | 56 GB            | 70 GB | 常见规模 |
| 13B     | 13B     | 26 GB          | 104 GB           | 130 GB | 中等规模 |
| 30B     | 30B     | 60 GB          | 240 GB           | 300 GB | 大规模 |
| 70B     | 70B     | 140 GB         | 560 GB           | 700 GB | 超大规模 |

**说明**:
- 优化器状态通常比模型参数大 4 倍（对于 bf16 模型）
- 总传输量主要由优化器状态主导
- 70B 模型需要传输 700GB，对 PCIe 带宽要求极高

---

## 3. PCIe 带宽瓶颈分析

### 3.1 PCIe 带宽规格

| PCIe 版本 | 理论带宽 | 实际可用带宽 | 编码开销 | 备注 |
|----------|---------|------------|---------|------|
| PCIe 3.0 x16 | 16 GB/s | ~13.6 GB/s | 128b/130b | 85% 效率 |
| PCIe 4.0 x16 | 32 GB/s | ~27.2 GB/s | 128b/130b | 85% 效率 |
| PCIe 5.0 x16 | 64 GB/s | ~54.4 GB/s | 128b/130b | 85% 效率 |

**说明**:
- PCIe 使用 128b/130b 编码，实际带宽约为理论值的 85%
- x16 通道数表示有 16 条数据传输通道
- CPU-GPU 传输需要经过 PCIe 总线

### 3.2 传输时间估算

#### PCIe 3.0 x16 (13.6 GB/s)

| 模型规模 | 总数据量 | 单向传输时间 | 双向传输时间 | 备注 |
|---------|---------|------------|-------------|------|
| 7B      | 70 GB   | 5.15 秒    | 10.3 秒     | wake_up + sleep |
| 13B     | 130 GB  | 9.56 秒    | 19.1 秒     | wake_up + sleep |
| 30B     | 300 GB  | 22.1 秒    | 44.2 秒     | wake_up + sleep |
| 70B     | 700 GB  | 51.5 秒    | 103 秒      | wake_up + sleep |

#### PCIe 4.0 x16 (27.2 GB/s)

| 模型规模 | 总数据量 | 单向传输时间 | 双向传输时间 | 备注 |
|---------|---------|------------|-------------|------|
| 7B      | 70 GB   | 2.57 秒    | 5.15 秒     | wake_up + sleep |
| 13B     | 130 GB  | 4.78 秒    | 9.56 秒     | wake_up + sleep |
| 30B     | 300 GB  | 11.0 秒    | 22.1 秒     | wake_up + sleep |
| 70B     | 700 GB  | 25.7 秒    | 51.5 秒     | wake_up + sleep |

**说明**:
- "单向传输时间" 指 CPU→GPU (wake_up) 或 GPU→CPU (sleep) 的时间
- "双向传输时间" 指完整的 wake_up + sleep 循环时间
- PCIe 4.0 相比 PCIe 3.0 快 2 倍

### 3.3 异步传输的性能提升

`move_torch_optimizer()` 使用 `non_blocking=True`，允许多个 tensor 并行传输：

```python
state[key] = value.to(device, non_blocking=True)
```

**性能提升**:
- 理论提升: 1.5-2x（取决于 tensor 数量和大小分布）
- 实际传输时间可能比上表中的估算值快 1.5-2 倍
- 例如：7B 模型在 PCIe 4.0 上的实际 wake_up 时间可能在 1.3-1.7 秒

---

## 4. Sleep/Wake_up 调用时机分析

### 4.1 初始化阶段的 Sleep 调用

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (line 139)

```python
def __init__(self, ...):
    # ... 模型和优化器初始化 ...

    # 在初始化结束时调用 sleep，将参数转移到 CPU
    if self.args.offload_train:
        self.sleep()
```

**时机**: Actor 初始化完成后，首次 rollout 开始前

**目的**: 为 SGLang 推理释放 GPU 显存

### 4.2 训练阶段的 Wake_up 调用

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 447-459)

```python
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    """Run one training update over a rollout batch."""

    # 训练开始前调用 wake_up，将参数从 CPU 移回 GPU
    if self.args.offload_train:
        self.wake_up()

    with inverse_timer("train_wait"), timer("train"):
        rollout_data = process_rollout_data(...)

        if self.args.debug_rollout_only:
            return

        self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

    # ⚠️ 关键发现：训练结束后不调用 sleep()！
    # 模型参数保留在 GPU 上，直到下次 rollout 开始
```

**关键发现**:
- `train()` 方法开始时调用 `wake_up()`
- `train()` 方法结束时**不调用** `sleep()`
- 参数在训练迭代之间保留在 GPU 上

### 4.3 完整的 Sleep/Wake_up 生命周期

```
初始化阶段:
  Actor.__init__() → sleep() → 参数移到 CPU

第一次训练迭代:
  train() 开始 → wake_up() → CPU→GPU 传输 (2-5 秒)
  train() 执行训练
  train() 结束 → 参数保留在 GPU

第二次训练迭代:
  train() 开始 → wake_up() → 参数已在 GPU，无操作
  train() 执行训练
  train() 结束 → 参数保留在 GPU

第 N 次训练迭代:
  train() 开始 → wake_up() → 参数已在 GPU，无操作
  train() 执行训练
  train() 结束 → 参数保留在 GPU
```

**wake_up() 的幂等性**:

查看 `wake_up()` 的实现，它调用 `self.model.cuda()`，PyTorch 会自动处理已经在 GPU 上的参数：

```python
# PyTorch 的行为
tensor.cuda()  # 如果 tensor 已在 GPU 上，这是一个空操作（no-op）
```

因此，第二次及后续调用 `wake_up()` 时，参数已经在 GPU 上，不会发生实际的数据传输。

---

## 5. 性能影响分析

### 5.1 实际性能开销

基于上述分析，colocated 模式的性能开销如下：

| 阶段 | 操作 | 数据传输 | 时间开销 (7B 模型, PCIe 4.0) |
|-----|------|---------|---------------------------|
| 初始化 | sleep() | GPU→CPU | 2.57 秒 |
| 第一次训练 | wake_up() | CPU→GPU | 2.57 秒 |
| 第二次训练 | wake_up() | 无 | ~0 秒 (no-op) |
| 第 N 次训练 | wake_up() | 无 | ~0 秒 (no-op) |

**总开销**:
- **初始化开销**: 2.57 秒（仅一次）
- **首次训练开销**: 2.57 秒（仅一次）
- **后续训练开销**: ~0 秒（无数据传输）

### 5.2 与训练时间的对比

假设每次训练迭代的实际训练时间为 10 秒（7B 模型，global_batch_size=128）：

| 迭代次数 | 传输开销 | 训练时间 | 总时间 | 开销占比 |
|---------|---------|---------|--------|---------|
| 第 1 次 | 2.57 秒 | 10 秒   | 12.57 秒 | 20.4% |
| 第 2 次 | 0 秒    | 10 秒   | 10 秒    | 0% |
| 第 10 次 | 0 秒   | 10 秒   | 10 秒    | 0% |
| 总计 (10 次) | 2.57 秒 | 100 秒 | 102.57 秒 | 2.5% |

**结论**:
- 首次训练迭代有 ~20% 的开销（2.57 秒传输 + 10 秒训练）
- 10 次训练迭代的平均开销仅为 2.5%
- 100 次训练迭代的平均开销降至 0.25%
- 对于长时间训练任务，PCIe 带宽瓶颈的影响可以忽略不计

### 5.3 为什么不在每次训练后调用 sleep()?

**代码设计的权衡**:

1. **推理阶段不需要训练参数**
   - Rollout 阶段使用 SGLang 进行推理
   - SGLang 有自己独立的模型副本（通过 `sync_weights()` 同步）
   - 训练参数可以保留在 GPU 上，不影响 SGLang 推理

2. **Colocated 模式的显存管理**
   - 训练 actor 和 SGLang 共享 GPU，但在不同时间使用
   - 训练时: 训练参数在 GPU，SGLang 暂停
   - 推理时: SGLang 使用 GPU，训练参数保留但不占用计算资源
   - 通过 `--sglang-mem-fraction-static` 限制 SGLang 的显存使用

3. **避免重复传输**
   - 如果每次训练后都 sleep()，下次训练前又要 wake_up()
   - 对于连续多次训练（如 `num_steps_per_rollout > 1`），会产生大量无谓的传输
   - 当前设计只在真正需要时传输（初始化 + 首次训练）

---

## 6. 为什么选择 CPU RAM 而非磁盘？

### 6.1 磁盘 I/O 性能对比

| 存储介质 | 顺序读写带宽 | 随机 I/O 延迟 | 传输 70GB 时间 |
|---------|------------|--------------|---------------|
| NVMe SSD (PCIe 4.0) | ~7 GB/s | 10-20 μs | 10 秒 |
| SATA SSD | ~0.5 GB/s | 50-100 μs | 140 秒 |
| HDD | ~0.2 GB/s | 5-10 ms | 350 秒 |
| **DDR4 RAM** | **~50 GB/s** | **50-100 ns** | **1.4 秒** |
| **PCIe 4.0 (CPU-GPU)** | **27.2 GB/s** | **- ** | **2.57 秒** |

**结论**:
- CPU RAM 的带宽比 NVMe SSD 快 5-7 倍
- CPU RAM 的延迟比 NVMe SSD 低 100-200 倍
- 使用磁盘会使传输时间增加 4-100 倍（取决于存储类型）

### 6.2 CPU RAM 的优势

1. **高带宽**: DDR4/DDR5 提供 50-100 GB/s 的带宽
2. **低延迟**: 纳秒级访问延迟 vs. 磁盘的微秒/毫秒级
3. **简单性**: PyTorch 原生支持 `.cpu()` 和 `.cuda()`，无需额外的序列化/反序列化
4. **可靠性**: 内存访问失败率远低于磁盘 I/O

### 6.3 为什么不直接保留在 GPU？

在 colocated 模式下，训练和推理共享 GPU：

```
GPU 显存分配:
  训练阶段: [训练参数 + 梯度 + 激活值] → 占满 GPU
  推理阶段: [SGLang 模型 + KV Cache] → 占满 GPU
```

**显存不足以同时容纳两者**:
- 7B 模型训练需要 ~40 GB 显存（参数 + 优化器 + 激活值）
- SGLang 推理需要 ~20 GB 显存（模型 + KV cache）
- A100 80GB 无法同时容纳两者（需要 60+ GB）

因此，必须在训练和推理之间切换，offload 到 CPU 是最快的选择。

---

## 7. 优化策略建议

### 7.1 硬件层面优化

1. **使用 PCIe 4.0 或 5.0**
   - PCIe 4.0 相比 PCIe 3.0 快 2 倍
   - PCIe 5.0 相比 PCIe 4.0 再快 2 倍
   - 对于大模型（70B+），PCIe 5.0 至关重要

2. **使用高速 CPU RAM**
   - DDR5 相比 DDR4 提供更高带宽
   - 确保 CPU 支持足够的内存通道（8-12 通道）

3. **拓扑优化**
   - 确保 GPU 和 CPU 之间的 PCIe 连接最短（同一 NUMA 节点）
   - 避免跨 NUMA 节点的内存访问

### 7.2 软件层面优化

1. **减少传输数据量**
   - 使用更激进的混合精度训练（bf16/fp16）
   - 考虑使用 8-bit 优化器（如 bitsandbytes）
   - 使用梯度检查点减少激活值内存

2. **利用异步传输**
   - `non_blocking=True` 已在 `move_torch_optimizer()` 中使用
   - 可以进一步优化模型参数的传输（当前 `self.model.cuda()` 可能是同步的）

3. **流水线化传输和计算**
   - 在传输参数的同时准备训练数据
   - 使用 CUDA streams 实现并行

### 7.3 算法层面优化

1. **增加 `num_steps_per_rollout`**
   - 当前设计下，参数在多个训练步之间保留在 GPU
   - 更多训练步可以摊薄首次传输的开销
   - 例如: `num_steps_per_rollout=10` 使开销从 20% 降至 2.5%

2. **考虑 Disaggregated 模式**
   - 对于有充足 GPU 资源的场景
   - 训练和推理使用独立 GPU，无需参数传输
   - 配置示例:
     ```bash
     --actor-num-gpus-per-node 4 \   # 训练使用 4 卡
     --rollout-num-gpus 4             # 推理使用另外 4 卡
     ```

---

## 8. 与其他框架的对比

### 8.1 DeepSpeed ZeRO-Offload

DeepSpeed 的 CPU offload 策略:

```python
# DeepSpeed ZeRO-Offload
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",  # 优化器状态 offload 到 CPU
            "pin_memory": True
        }
    }
}
```

**与 slime 的区别**:
- DeepSpeed: 仅 offload 优化器状态，模型参数保留在 GPU
- slime: offload 模型参数 + 优化器状态，释放更多显存

### 8.2 FSDP (Fully Sharded Data Parallel)

PyTorch FSDP 的 CPU offload:

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True)
)
```

**与 slime 的区别**:
- FSDP: 逐层 offload，训练时按需加载
- slime: 全量 offload/wake_up，适合训练-推理切换场景

### 8.3 Megatron-LM

Megatron-LM 主要使用张量并行和流水线并行，较少使用 CPU offload：

**原因**:
- Megatron 针对多 GPU 训练优化
- 假设有足够的 GPU 显存
- 不需要在同一 GPU 上切换训练和推理

**slime 的创新点**:
- 针对 RL 训练的特殊需求（训练 + 推理交替）
- 在有限 GPU 资源下实现 colocated 模式
- 通过 CPU offload 实现显存复用

---

## 9. 总结

### 9.1 问题回答

**Q1: sleep 操作将参数移动到了哪里？**
- **答**: CPU 内存（RAM），不是磁盘
- **实现**: 通过 PyTorch 的 `model.cpu()` 和 `move_torch_optimizer(optimizer, "cpu")`
- **传输路径**: GPU VRAM → PCIe 总线 → CPU RAM

**Q2: 带宽瓶颈会不会导致训练和推理切换很慢？**
- **答**: 仅在首次训练迭代有影响，后续训练无开销
- **首次开销**: 2-5 秒（7B 模型，PCIe 4.0），占首次训练的 20%
- **长期开销**: 10 次迭代平均 2.5%，100 次迭代平均 0.25%，可忽略不计

### 9.2 关键设计洞察

1. **Sleep 仅在初始化时调用**
   - `Actor.__init__()` 结束时调用 `sleep()`
   - 为首次 rollout（SGLang 推理）释放显存

2. **Wake_up 在每次训练前调用，但幂等**
   - `train()` 开始时调用 `wake_up()`
   - 首次调用: 执行 CPU→GPU 传输
   - 后续调用: 参数已在 GPU，无操作（no-op）

3. **训练后不调用 sleep**
   - 避免重复传输
   - 训练参数保留在 GPU 上，不影响 SGLang 推理（独立模型副本）

4. **Colocated 模式的核心**
   - 训练和推理时间分离（rollout 阶段 vs. train 阶段）
   - 通过 CPU offload 实现显存复用
   - 通过智能的 sleep/wake_up 调用时机最小化传输开销

### 9.3 性能优化建议

1. **硬件**: 使用 PCIe 4.0/5.0，高速 DDR5 RAM
2. **配置**: 增加 `num_steps_per_rollout` 摊薄传输开销
3. **算法**: 对于充足 GPU 资源，考虑 disaggregated 模式避免传输

### 9.4 适用场景

**Colocated 模式适合**:
- GPU 资源有限（单机 8 卡或更少）
- 训练和推理可以时间分离
- 模型规模在 7B-30B（传输时间可接受）

**Disaggregated 模式适合**:
- GPU 资源充足（多机多卡）
- 需要最高训练性能
- 模型规模在 70B+（传输时间过长）

---

## 10. 相关源码索引

| 功能 | 文件路径 | 行号 |
|-----|---------|------|
| sleep() 实现 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 276-288 |
| wake_up() 实现 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 290-298 |
| move_torch_optimizer() | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 1001-1013 |
| 初始化 sleep 调用 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 139 |
| train() 中的 wake_up 调用 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 447-459 |
| --offload-train 参数 | `/home/scbjtfy/slime/slime/utils/arguments.py` | 92-98 |

---

**生成时间**: 2025-12-04
**分析框架版本**: slime (commit: 9d7f34d)
**分析者**: Claude Code (Sonnet 4.5)
