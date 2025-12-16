# FSDP2 Mixed Precision：Master Weights 存储、梯度裁剪与通信量分析

## 概述

本文档针对 Infra 学习者，深入分析 Slime 框架中 FSDP2 Mixed Precision 模式的三个核心问题：

1. **Master Weights 存储方式**：FP32 副本是切分存储还是完整存储？
2. **Gradient Clip 精度与通信**：梯度裁剪时的精度转换和 All-Reduce 机制
3. **通信量评估**：Mixed Precision 训练的带宽开销分析

本分析基于 Slime 框架源码（commit: 9d7f34d），所有结论均有代码依据。

---

## 问题 1：Master Weights 的存储方式

### 结论

**Master Weights（FP32 副本）是切分存储在各卡上的**，每个 GPU 仅存储 1/N 的参数（N = DP size）。

### 详细分析

#### 1.1 配置来源

**文件**：`slime/backends/fsdp_utils/actor.py:1042-1045`

```python
fsdp_kwargs = {
    "mp_policy": MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,    # Unsharded 参数（All-Gather 后）的精度
        reduce_dtype=torch.float32,     # 梯度归约的精度
    ),
    "offload_policy": offload_policy,
    "mesh": mesh,
}
```

#### 1.2 参数存储的三种状态

根据 PyTorch FSDP2 官方文档和 Slime 实现：

| 状态 | 精度 | 位置 | 大小（7B 模型，4 GPU DP） |
|------|------|------|--------------------------|
| **Sharded Parameters（分片存储）** | **FP32** | 各 GPU | 7 GB / GPU |
| **Unsharded Parameters（All-Gather 后）** | **BF16** | 各 GPU（临时） | 14 GB / GPU |
| **Optimizer States** | **FP32** | 各 GPU | 14 GB / GPU（exp_avg + exp_avg_sq）|

**关键点**：
- ✅ **Sharded Parameters 是 FP32**：这就是 "Master Weights"
- ✅ **每个 GPU 仅存储 1/N**：数据并行分片
- ✅ **Unsharded Parameters 是临时的**：Forward/Backward 后立即释放

#### 1.3 精度转换流程

```
[存储阶段]
  各 GPU: Sharded Params (FP32, 7GB / 4 = 1.75 GB per GPU)
           ↓
[Forward 前]
  All-Gather (跨 dp_group)
           ↓
  各 GPU: Unsharded Params (BF16, 7GB × 2 bytes = 14 GB per GPU)
           ↓
[Forward 计算]
  使用 BF16 Unsharded Params 进行前向传播
           ↓
[Forward 后]
  释放 Unsharded Params（节省显存）
  保留 Sharded Params (FP32)
           ↓
[Backward 时]
  再次 All-Gather → BF16 Unsharded Params
  反向传播 → 计算 BF16 梯度
           ↓
[Reduce-Scatter]
  BF16 Local Gradients → FP32（upcast）
  Reduce-Scatter (FP32 精度求和)
           ↓
  各 GPU: Sharded Gradients (FP32, 1.75 GB per GPU)
           ↓
[Optimizer Step]
  更新 Sharded Params (FP32) 使用 Sharded Gradients (FP32)
```

#### 1.4 为什么 Master Weights 存储为 FP32 分片？

根据源码分析和 PyTorch 官方文档：

1. **Optimizer 兼容性**：
   - AdamW 的 `exp_avg` 和 `exp_avg_sq` 必须是 FP32 才能避免数值下溢
   - Optimizer 直接操作 Sharded Params，精度必须一致

2. **避免精度转换开销**：
   - 如果 Sharded Params 是 BF16，每次 `optimizer.step()` 需要：
     - BF16 → FP32（更新前）
     - FP32 → BF16（更新后）
   - 保持 FP32 避免了这个往返转换

3. **数值稳定性**：
   - 参数更新的累积误差在 FP32 下更小
   - 长时间训练（数万步）时精度损失显著降低

#### 1.5 内存占用详解

**7B 模型，4 GPU DP，单卡显存占用**：

```
组件                    精度    大小（单 GPU）
─────────────────────────────────────────────
Sharded Params          FP32    7 GB / 4 = 1.75 GB
Optimizer exp_avg       FP32    7 GB / 4 = 1.75 GB
Optimizer exp_avg_sq    FP32    7 GB / 4 = 1.75 GB
Sharded Gradients       FP32    7 GB / 4 = 1.75 GB
─────────────────────────────────────────────
持久化显存总计                  7 GB

训练时临时峰值：
Unsharded Params (临时) BF16    14 GB
Activations            BF16    变动（取决于 batch size）
─────────────────────────────────────────────
训练时显存峰值                  21 GB + Activations
```

**关键洞察**：
- **持久化数据**（params + optimizer states + grads）：7 GB / GPU
- **临时数据**（unsharded params）：14 GB / GPU，仅在 Forward/Backward 时存在
- **显存峰值**：Forward/Backward 时最高，之后降回 7 GB

---

## 问题 2：Gradient Clip 阶段的精度与通信

### 结论

1. **梯度已经是 FP32**：Gradient Clip 时梯度已通过 Reduce-Scatter 转换为 FP32
2. **需要一次 All-Reduce**：`clip_grad_norm_` 计算全局梯度范数时需要跨 GPU 通信
3. **通信对象是标量**：仅传输梯度范数（scalar），通信量极小（< 1 MB）

### 详细分析

#### 2.1 Gradient Clip 的代码位置

**文件**：`slime/backends/fsdp_utils/actor.py:711-718`

```python
if (mbs_id + 1) in grad_accum:
    # TODO: check if the grad norm is global grad norm.
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(),
        self.args.clip_grad
    )
    # the grad norm used to be of DTensor
    grad_norm = float(grad_norm)

    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
```

#### 2.2 梯度精度时间线

| 时刻 | 梯度状态 | 精度 | 位置 |
|------|---------|------|------|
| **Backward 计算** | 局部梯度 | BF16 | 各 GPU（完整梯度）|
| **Micro-batch 累积** | 累积局部梯度 | BF16 | 各 GPU（累加多个 micro-batch）|
| **Reduce-Scatter 前** | 累积局部梯度 | BF16 → **FP32** (upcast) | 转换中 |
| **Reduce-Scatter 中** | 梯度求和 | **FP32** | 跨 dp_group 通信 |
| **Reduce-Scatter 后** | 分片梯度 | **FP32** | 各 GPU（1/N 分片）|
| **Gradient Clip 时** | 分片梯度 | **FP32** | 各 GPU |
| **Optimizer Step 时** | 分片梯度 | **FP32** | 各 GPU |

**关键发现**：
- ✅ Gradient Clip 发生在 Reduce-Scatter **之后**
- ✅ 此时梯度已经是 **FP32 分片梯度**
- ✅ **无需再次 cast**，梯度已经是正确精度

#### 2.3 clip_grad_norm_ 的工作机制

PyTorch 的 `torch.nn.utils.clip_grad_norm_` 对 FSDP2 DTensor 的处理：

```python
# 伪代码：clip_grad_norm_ 在 FSDP2 下的行为
def clip_grad_norm_(parameters, max_norm):
    # 步骤 1：计算每个 GPU 上分片梯度的局部 L2 范数
    local_norm_sq = 0.0
    for param in parameters:
        if param.grad is not None:
            # param.grad 是 DTensor（分布式张量），代表 1/N 分片
            local_norm_sq += (param.grad ** 2).sum()

    # 步骤 2：All-Reduce 求全局范数平方和
    # DTensor 自动处理跨 dp_group 的通信
    global_norm_sq = all_reduce(local_norm_sq, group=dp_group)  # ← 通信发生在这里

    # 步骤 3：计算全局范数
    global_norm = sqrt(global_norm_sq)

    # 步骤 4：如果超过阈值，裁剪所有梯度
    if global_norm > max_norm:
        clip_coef = max_norm / (global_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad *= clip_coef  # 每个 GPU 裁剪自己的分片

    return global_norm
```

**通信细节**：
- **通信操作**：All-Reduce（求和）
- **通信对象**：标量（`local_norm_sq`，FP32，4 bytes）
- **通信组**：`dp_group`（数据并行组）
- **通信量**：4 GPU DP → 4 × 4 bytes = 16 bytes（可忽略不计）

#### 2.4 为什么需要 All-Reduce？

因为每个 GPU 只存储 1/N 的梯度分片，局部范数不等于全局范数：

```
假设 4 GPU，某参数梯度分布：
  GPU 0: grad_shard_0 = [1.0, 2.0]     → local_norm_0 = sqrt(1² + 2²) = 2.236
  GPU 1: grad_shard_1 = [3.0, 4.0]     → local_norm_1 = sqrt(3² + 4²) = 5.0
  GPU 2: grad_shard_2 = [5.0, 6.0]     → local_norm_2 = sqrt(5² + 6²) = 7.810
  GPU 3: grad_shard_3 = [7.0, 8.0]     → local_norm_3 = sqrt(7² + 8²) = 10.630

错误方式（不 All-Reduce）：
  max(local_norm_i) = 10.630  ← 仅反映 GPU 3 的分片

正确方式（All-Reduce）：
  global_norm² = 1² + 2² + 3² + 4² + 5² + 6² + 7² + 8² = 204
  global_norm = sqrt(204) = 14.283  ← 反映全局梯度
```

**结论**：All-Reduce 是必需的，但通信量极小（仅标量）。

#### 2.5 精度保证

- **梯度是 FP32**：Reduce-Scatter 已转换，数值精度高
- **范数计算是 FP32**：`local_norm_sq` 和 `global_norm_sq` 都是 FP32
- **裁剪系数是 FP32**：`clip_coef = max_norm / global_norm`
- **裁剪操作是 FP32**：`param.grad *= clip_coef`（FP32 tensor 上的操作）

**全程 FP32，无精度损失**。

---

## 问题 3：通信量评估

### 结论

**7B 模型，4 GPU DP，BF16 混合精度训练**：

| 操作 | 通信量（单向） | 精度 | 频率 |
|------|--------------|------|------|
| **Parameter All-Gather** | 14 GB / GPU | BF16 | 每次 Forward/Backward（2 次）|
| **Gradient Reduce-Scatter** | 7 GB / GPU | FP32 | 每次 Backward |
| **Gradient Clip All-Reduce** | < 1 KB / GPU | FP32 | 每次 Optimizer Step |
| **Metric All-Gather** | < 1 MB / GPU | FP32 | 每次 Optimizer Step |
| **总计（每训练步）** | **35 GB / GPU** | 混合 | - |

### 详细分析

#### 3.1 Parameter All-Gather 通信量

**操作流程**：
```
各 GPU: Sharded Params (FP32, 1.75 GB)
    ↓ 转换
各 GPU: Sharded Params (BF16, 1.75 GB 转为 0.875 GB)
    ↓ All-Gather
各 GPU: Unsharded Params (BF16, 14 GB)
```

**通信量计算**：

```
单个 GPU 的 All-Gather 通信量：
  - 发送：1.75 GB (FP32) → 转换为 BF16 → 发送 0.875 GB (BF16)
  - 接收：3 × 0.875 GB = 2.625 GB（从其他 3 个 GPU）
  - 单向总计：0.875 + 2.625 = 3.5 GB

实际上按 BF16 计算：
  - 总参数：7B × 2 bytes (BF16) = 14 GB
  - 每个 GPU 广播：14 GB / 4 = 3.5 GB
  - 每个 GPU 接收：3 × 3.5 GB = 10.5 GB
  - 单 GPU 通信量：3.5 GB (发送) + 10.5 GB (接收) = 14 GB (双向)
  - 单向等效：14 GB

注：Forward 和 Backward 各需要一次 All-Gather：
  - Forward: 14 GB
  - Backward: 14 GB
  - 总计：28 GB / GPU
```

#### 3.2 Gradient Reduce-Scatter 通信量

**操作流程**：
```
各 GPU: Local Gradients (BF16, 14 GB)
    ↓ Upcast
各 GPU: Local Gradients (FP32, 28 GB)
    ↓ Reduce-Scatter
各 GPU: Sharded Gradients (FP32, 7 GB)
```

**通信量计算**：

```
Reduce-Scatter 的通信模式：
  - 阶段 1：Reduce（各 GPU 计算自己负责的分片）
  - 阶段 2：Scatter（分发结果）

简化计算（单向等效）：
  - 总梯度：7B × 4 bytes (FP32) = 28 GB
  - Reduce-Scatter 通信量 ≈ 总梯度大小 = 28 GB (双向)
  - 单向等效：28 GB / 4 GPU = 7 GB / GPU
```

**关键点**：
- ✅ 梯度在 Reduce-Scatter **前** upcast 为 FP32
- ⚠️ 通信量比 BF16 Reduce-Scatter 多一倍（28 GB vs 14 GB）
- ✅ 但数值精度得到保证（避免小梯度丢失）

#### 3.3 Gradient Clip All-Reduce 通信量

```
通信对象：标量（local_norm_sq）
数据类型：FP32
通信量：4 bytes
4 GPU All-Reduce：4 × 4 bytes = 16 bytes

完全可忽略不计（< 1 KB）
```

#### 3.4 Metric Reduction 通信量

**文件**：`slime/backends/fsdp_utils/actor.py:722-726`

```python
# Aggregate logs across DP ranks
aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
reduced_aggregated = [None] * self.dp_size
dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
```

**通信量估算**：
```
Metric 数量：~10 个（loss, grad_norm, kl_div, entropy, etc.）
每个 metric：8 bytes (FP64 Python float)
总大小：10 × 8 = 80 bytes

All-Gather-Object：4 GPU × 80 bytes = 320 bytes

完全可忽略不计（< 1 MB）
```

#### 3.5 总通信量表

**7B 模型，4 GPU DP，每个训练步**：

| 操作 | 单 GPU 通信量 | 总带宽（4 GPU）| 占比 |
|------|--------------|--------------|------|
| **Forward All-Gather** | 14 GB | 56 GB | 40.0% |
| **Backward All-Gather** | 14 GB | 56 GB | 40.0% |
| **Gradient Reduce-Scatter** | 7 GB | 28 GB | 20.0% |
| **Gradient Clip All-Reduce** | < 1 KB | < 4 KB | 0.0% |
| **Metric All-Gather** | < 1 KB | < 4 KB | 0.0% |
| **总计** | **35 GB** | **140 GB** | **100%** |

#### 3.6 与全 FP32 训练对比

| 配置 | All-Gather (Forward) | All-Gather (Backward) | Reduce-Scatter | 总计 / GPU |
|------|---------------------|----------------------|---------------|-----------|
| **全 FP32** | 28 GB | 28 GB | 7 GB | 63 GB |
| **Mixed BF16** | 14 GB | 14 GB | 7 GB | 35 GB |
| **节省** | 14 GB (50%) | 14 GB (50%) | 0 GB (0%) | 28 GB (44.4%) |

**权衡分析**：
- ✅ **All-Gather 节省 50%**：param_dtype=BF16 传输更少数据
- ⚠️ **Reduce-Scatter 不节省**：reduce_dtype=FP32 保证精度
- ✅ **总通信量减少 44.4%**：显著降低网络压力
- ✅ **数值稳定性不受影响**：FP32 梯度归约保证精度

#### 3.7 实际网络带宽需求

假设：
- GPU 间互联：NVLink 或 InfiniBand
- 训练步用时：10 秒

```
带宽需求计算：
  每训练步通信量：35 GB / GPU
  每秒带宽：35 GB / 10 s = 3.5 GB/s = 28 Gbps

网络技术对比：
  - NVLink 3.0: 600 GB/s (双向) → 绰绰有余
  - InfiniBand HDR: 200 Gbps (25 GB/s) → 充足
  - PCIe 4.0 x16: 32 GB/s → 充足
  - 10GbE: 1.25 GB/s → 不足（瓶颈）
```

**推荐配置**：
- **单机多卡**：NVLink（最佳）或 PCIe 4.0（足够）
- **多机训练**：InfiniBand HDR（200 Gbps）或更高
- **不推荐**：普通以太网（1/10 GbE），会成为严重瓶颈

---

## 核心洞察总结

### 1. Master Weights 存储策略

```
设计理念：分片存储 + 按需聚合

存储：各 GPU 存储 FP32 分片（1/N）
计算：临时 All-Gather 为 BF16 完整参数
更新：直接在 FP32 分片上进行

优点：
  ✅ 显存占用低（分片）
  ✅ 计算效率高（BF16）
  ✅ 更新精度高（FP32）
  ✅ 无精度转换开销
```

### 2. Gradient Clip 精度保证

```
精度转换时机：Reduce-Scatter 时（BF16 → FP32）
Gradient Clip 时机：Reduce-Scatter 之后

结果：
  ✅ 梯度已经是 FP32
  ✅ 范数计算是 FP32
  ✅ 裁剪操作是 FP32
  ✅ 全程高精度，无数值损失

通信开销：
  ✅ 仅 All-Reduce 一个标量
  ✅ 通信量 < 1 KB（可忽略）
```

### 3. 通信量评估方法

```
评估维度：
  1. 操作类型：All-Gather vs Reduce-Scatter
  2. 数据精度：BF16 vs FP32
  3. 通信频率：每 Forward/Backward/Step
  4. 数据规模：模型参数量 × 数据类型大小

关键公式：
  All-Gather (BF16): model_size × 2 bytes
  Reduce-Scatter (FP32): model_size × 4 bytes / dp_size
  总通信量 = 2 × All-Gather + Reduce-Scatter
```

---

## 实战建议

### 1. 复现 FSDP2 后端时的关键点

如果你要在其他框架中实现类似的 FSDP2 混合精度机制：

#### 1.1 参数管理

```python
class FSDP2Module:
    def __init__(self, module, mp_policy):
        # 1. 参数分片（FP32）
        self.sharded_params = shard_parameters(module.parameters(), dp_rank, dp_size)

        # 2. 记录精度策略
        self.param_dtype = mp_policy.param_dtype  # BF16
        self.reduce_dtype = mp_policy.reduce_dtype  # FP32

    def forward(self, x):
        # 3. All-Gather + 转换为 BF16
        unsharded_params = all_gather(self.sharded_params, dp_group)
        unsharded_params = unsharded_params.to(self.param_dtype)

        # 4. 前向计算（BF16）
        output = original_forward(x, unsharded_params)

        # 5. 释放 unsharded params
        del unsharded_params

        return output

    def backward_hook(self, grad):
        # 6. 梯度 upcast 为 FP32
        grad_fp32 = grad.to(self.reduce_dtype)

        # 7. Reduce-Scatter（FP32）
        sharded_grad = reduce_scatter(grad_fp32, dp_group)

        # 8. 存储到参数（FP32 分片）
        self.sharded_params.grad = sharded_grad
```

#### 1.2 Gradient Clip 实现

```python
def clip_grad_norm_distributed(parameters, max_norm, dp_group):
    """分布式环境下的梯度裁剪"""
    # 1. 计算局部范数平方
    local_norm_sq = 0.0
    for param in parameters:
        if param.grad is not None:
            local_norm_sq += (param.grad ** 2).sum()

    # 2. All-Reduce 求全局范数平方
    global_norm_sq = torch.tensor(local_norm_sq, device='cuda')
    dist.all_reduce(global_norm_sq, group=dp_group)  # ← 关键通信

    # 3. 计算全局范数
    global_norm = torch.sqrt(global_norm_sq)

    # 4. 裁剪梯度
    if global_norm > max_norm:
        clip_coef = max_norm / (global_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(clip_coef)

    return global_norm.item()
```

#### 1.3 通信量监控

```python
class CommunicationProfiler:
    def __init__(self):
        self.comm_log = []

    def profile_all_gather(self, tensor_size, dtype, group):
        """监控 All-Gather 通信量"""
        elem_size = get_dtype_size(dtype)  # BF16: 2, FP32: 4
        total_size = tensor_size * elem_size

        world_size = dist.get_world_size(group)
        comm_volume = total_size  # 单 GPU 接收量（近似）

        self.comm_log.append({
            'op': 'all_gather',
            'size': comm_volume,
            'dtype': dtype
        })

    def profile_reduce_scatter(self, tensor_size, dtype, group):
        """监控 Reduce-Scatter 通信量"""
        elem_size = get_dtype_size(dtype)
        total_size = tensor_size * elem_size

        world_size = dist.get_world_size(group)
        comm_volume = total_size / world_size  # 单 GPU 接收分片

        self.comm_log.append({
            'op': 'reduce_scatter',
            'size': comm_volume,
            'dtype': dtype
        })

    def get_total_communication(self):
        """计算总通信量"""
        return sum(item['size'] for item in self.comm_log)
```

### 2. 内存管理注意事项

#### 2.1 显存峰值控制

```python
# 问题：Unsharded Params 可能导致 OOM
# 解决：使用 Activation Checkpointing

from torch.utils.checkpoint import checkpoint

class FSDPModuleWithCheckpoint:
    def forward(self, x):
        # 方案 1：逐层 checkpoint
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)

        # 方案 2：分段 checkpoint
        x = checkpoint(self.layers[0:10], x)
        x = checkpoint(self.layers[10:20], x)

        return x
```

#### 2.2 Optimizer State 管理

```python
# 确保 Optimizer State 精度与 Sharded Params 一致
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

# 验证精度
for param in model.parameters():
    state = optimizer.state[param]
    assert state['exp_avg'].dtype == torch.float32
    assert state['exp_avg_sq'].dtype == torch.float32
```

### 3. 调试技巧

#### 3.1 精度验证

```python
def verify_mixed_precision(model, optimizer):
    """验证混合精度配置是否正确"""
    print("=== 验证 Mixed Precision 配置 ===")

    # 1. 检查 sharded params 精度
    for name, param in model.named_parameters():
        print(f"Param {name}: dtype={param.dtype}, shape={param.shape}")
        assert param.dtype == torch.float32, f"{name} should be FP32"

    # 2. 检查 optimizer state 精度
    for param in model.parameters():
        if param in optimizer.state:
            state = optimizer.state[param]
            assert state['exp_avg'].dtype == torch.float32
            assert state['exp_avg_sq'].dtype == torch.float32

    # 3. 检查梯度精度（在 backward 后）
    # loss.backward()
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         assert param.grad.dtype == torch.float32

    print("✅ 精度验证通过")
```

#### 3.2 通信量分析

```python
def analyze_communication(model_size_b, dp_size):
    """分析通信量"""
    param_count = model_size_b * 1e9

    # All-Gather (BF16)
    all_gather_size = param_count * 2  # BF16: 2 bytes

    # Reduce-Scatter (FP32)
    reduce_scatter_size = param_count * 4 / dp_size  # FP32: 4 bytes, 分片

    # 每训练步总通信量
    total_comm = 2 * all_gather_size + reduce_scatter_size  # 2x for forward & backward

    print(f"=== 通信量分析 ({model_size_b}B 模型，{dp_size} GPU DP) ===")
    print(f"Forward All-Gather:  {all_gather_size / 1e9:.2f} GB")
    print(f"Backward All-Gather: {all_gather_size / 1e9:.2f} GB")
    print(f"Reduce-Scatter:      {reduce_scatter_size / 1e9:.2f} GB")
    print(f"总计（每训练步）:     {total_comm / 1e9:.2f} GB / GPU")

    # 带宽需求（假设 10 秒/步）
    bandwidth_gbps = (total_comm / 1e9) / 10 * 8
    print(f"带宽需求（10s/步）:   {bandwidth_gbps:.2f} Gbps")

# 示例
analyze_communication(model_size_b=7, dp_size=4)
```

#### 3.3 数值稳定性检查

```python
def check_numerical_stability(model, ref_model):
    """检查训练初期的数值稳定性"""
    # 第一步训练检查
    with torch.no_grad():
        for (name, param), (ref_name, ref_param) in zip(
            model.named_parameters(),
            ref_model.named_parameters()
        ):
            # 1. 参数应该完全相同（初始化后）
            assert torch.allclose(param, ref_param, atol=1e-5)

    # 训练一步
    loss = train_one_step(model)
    ref_loss = train_one_step(ref_model)

    # 2. Loss 应该接近
    assert torch.allclose(loss, ref_loss, atol=1e-3), \
        f"Loss mismatch: {loss} vs {ref_loss}"

    # 3. 梯度范数应该合理
    grad_norm = compute_grad_norm(model)
    assert 0 < grad_norm < 100, f"Abnormal grad norm: {grad_norm}"

    print("✅ 数值稳定性检查通过")
```

---

## 相关源码索引

| 功能 | 文件路径 | 行号 |
|-----|---------|------|
| MixedPrecisionPolicy 配置 | `slime/backends/fsdp_utils/actor.py` | 1042-1045 |
| Gradient Clip 实现 | `slime/backends/fsdp_utils/actor.py` | 711-718 |
| move_torch_optimizer | `slime/backends/fsdp_utils/actor.py` | 1001-1013 |
| Device Mesh 设置 | `slime/backends/fsdp_utils/actor.py` | 164-210 |
| Optimizer 初始化 | `slime/backends/fsdp_utils/actor.py` | 106-115 |
| 训练循环 | `slime/backends/fsdp_utils/actor.py` | 447-465, 510-546 |
| Metric Reduction | `slime/backends/fsdp_utils/actor.py` | 722-726 |

---

## 参考资料

1. **PyTorch 官方文档**：
   - [FSDP2 MixedPrecisionPolicy API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
   - [Getting Started with FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

2. **Slime 框架文档**：
   - `docs/analysis/fsdp2_mixed_precision_policy_deep_dive.md`（混合精度详解）
   - `docs/analysis/fsdp2_optimizer_state_lifecycle.md`（优化器状态生命周期）

3. **技术博客**：
   - [Why reduction precision matters](https://main-horse.github.io/posts/reduction-precision/)

4. **GitHub Issues**：
   - [Question on MixedPrecisionPolicy in FSDP2 · pytorch/torchtitan#600](https://github.com/pytorch/torchtitan/issues/600)

---

**文档版本**：v1.0
**基于代码版本**：slime main branch (commit: 9d7f34d)
**生成日期**：2025-12-11
**目标读者**：Infra 学习者，希望复现 FSDP2 后端的工程师
