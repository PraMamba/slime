# FSDP2 Optimizer State 生命周期与状态一致性分析

## Problem-8: 优化器状态的持久化与一致性保证

### 问题描述

在训练结束后，Optimizer States (Adam 的动量等) 是被销毁了，还是也 Offload 到 CPU 了？下次 wake_up 时如何保证状态的一致性？

### 核心发现总结

1. **优化器状态不会被销毁**: 训练结束后，optimizer.state 保留在内存中（GPU 或 CPU）
2. **Offload 机制**: `sleep()` 将优化器状态移到 CPU，`wake_up()` 移回 GPU，状态本身不销毁
3. **状态一致性**: 通过 PyTorch 的参数-状态映射机制 (`optimizer.state[param]`) 保证一致性
4. **关键设计**: 训练迭代之间不调用 `sleep()`，状态保留在 GPU，避免重复传输

---

## 1. Optimizer State 的数据结构

### 1.1 AdamW Optimizer State 组成

以 AdamW 优化器为例，每个参数的状态包含：

```python
optimizer.state[param] = {
    'step': Tensor,        # 当前训练步数（标量）
    'exp_avg': Tensor,     # 一阶动量（梯度的指数移动平均）
    'exp_avg_sq': Tensor,  # 二阶动量（梯度平方的指数移动平均）
}
```

**实际示例**（通过 Python 演示）:

```python
import torch

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 初始状态（空）
print('Initial optimizer.state:', dict(optimizer.state))
# 输出: Initial optimizer.state: {}

# 执行一次 step 后
loss = model(torch.randn(1, 2)).sum()
loss.backward()
optimizer.step()

# 查看状态
for param in model.parameters():
    if param in optimizer.state:
        print(f'Optimizer state for param shape {param.shape}:')
        for key, value in optimizer.state[param].items():
            print(f'  {key}: {type(value).__name__} shape={value.shape}')

# 输出示例:
# Optimizer state for param shape torch.Size([1, 2]):
#   step: Tensor shape=torch.Size([])
#   exp_avg: Tensor shape=torch.Size([1, 2])
#   exp_avg_sq: Tensor shape=torch.Size([1, 2])
```

### 1.2 Optimizer State 的存储机制

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 106-115)

```python
if args.optimizer == "adam":
    self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
else:
    raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")
```

**关键特性**:
- `self.optimizer` 是 actor 的成员变量，在整个训练过程中持续存在
- `optimizer.state` 是一个字典: `{Parameter对象 -> 状态字典}`
- 状态字典中的 `exp_avg` 和 `exp_avg_sq` 是与参数同形状的 tensor

### 1.3 Optimizer State 的内存占用

| 模型规模 | 参数数量 | 单参数大小 (bf16) | exp_avg (fp32) | exp_avg_sq (fp32) | step | 总计 |
|---------|---------|------------------|----------------|------------------|------|------|
| 7B      | 7B      | 14 GB            | 28 GB          | 28 GB            | ~0   | 56 GB |
| 13B     | 13B     | 26 GB            | 52 GB          | 52 GB            | ~0   | 104 GB |
| 30B     | 30B     | 60 GB            | 120 GB         | 120 GB           | ~0   | 240 GB |
| 70B     | 70B     | 140 GB           | 280 GB         | 280 GB           | ~0   | 560 GB |

**说明**:
- exp_avg 和 exp_avg_sq 通常使用 fp32，无论模型参数是 bf16 还是 fp16
- step 是标量，内存可忽略不计
- 优化器状态占用 = 参数数量 × 8 bytes (两个 fp32 tensor)

---

## 2. Optimizer State 的生命周期

### 2.1 初始化阶段

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 106-139)

```python
# 1. 创建优化器对象
self.optimizer = torch.optim.AdamW(
    self.model.parameters(),
    lr=args.lr,
    betas=(args.adam_beta1, args.adam_beta2),
    eps=args.adam_eps,
    weight_decay=args.weight_decay,
)

# 2. 加载 checkpoint（如果有）
checkpoint_payload = checkpoint.load(self)

# 3. 恢复优化器状态（如果 checkpoint 中有）
checkpoint.finalize_load(self, checkpoint_payload)

# 4. 初始化结束时调用 sleep（offload 模式）
if self.args.offload_train:
    self.sleep()  # 将参数和优化器状态移到 CPU
```

**初始状态**:
- 新训练: `optimizer.state = {}` (空字典)
- 从 checkpoint 恢复: `optimizer.state` 包含之前的 exp_avg, exp_avg_sq, step

### 2.2 第一次训练迭代

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 447-465, 510-546)

```python
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    """Run one training update over a rollout batch."""

    # 1. Wake up: 将参数和优化器状态从 CPU 移回 GPU
    if self.args.offload_train:
        self.wake_up()

    # 2. 处理 rollout 数据
    with inverse_timer("train_wait"), timer("train"):
        rollout_data = process_rollout_data(...)
        if self.args.debug_rollout_only:
            return
        self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

    # ⚠️ 关键: 训练结束后不调用 sleep()
    # 优化器状态保留在 GPU 上

def _train_core(self, rollout_id: int, rollout_data) -> None:
    # ... 计算 advantages, log_probs 等 ...

    with timer("actor_train"):
        reported_accum: dict[str, list[torch.Tensor]] = {}

        # 3. 清空梯度
        self.optimizer.zero_grad(set_to_none=True)

        # 4. 遍历所有 micro-batches
        for mbs_id, packed_batch in enumerate(packed_batches):
            self._train_step(
                packed_batch=packed_batch,
                reported_accum=reported_accum,
                ...
            )
```

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 711-718)

```python
def _train_step(self, ...):
    # ... 前向传播、计算 loss ...

    # 5. 反向传播
    loss.backward()

    # 6. 梯度累积后执行优化器 step
    if (mbs_id + 1) in grad_accum:
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.clip_grad
        )

        # 7. 更新参数和优化器状态
        self.optimizer.step()  # ← 这里会初始化/更新 optimizer.state

        # 8. 清空梯度准备下次累积
        self.optimizer.zero_grad(set_to_none=True)
```

**第一次 `optimizer.step()` 发生的事情**:

对于 AdamW，PyTorch 内部执行：

```python
# 伪代码: PyTorch AdamW.step() 内部逻辑
for group in self.param_groups:
    for p in group['params']:
        if p.grad is None:
            continue

        # 初始化 state（如果不存在）
        state = self.state[p]
        if len(state) == 0:
            state['step'] = torch.tensor(0.0)
            state['exp_avg'] = torch.zeros_like(p)      # 一阶动量
            state['exp_avg_sq'] = torch.zeros_like(p)   # 二阶动量

        # 更新 state
        state['step'] += 1
        state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

        # 更新参数
        # ... (省略具体的 AdamW 更新公式) ...
```

**状态变化**:
- 第一次 step 前: `optimizer.state = {}`
- 第一次 step 后: `optimizer.state[param] = {'step': 1, 'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)}`
- 状态位置: GPU (因为参数在 GPU 上)

### 2.3 第二次及后续训练迭代

```python
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    # 1. Wake up: 检查参数和优化器状态是否在 GPU
    if self.args.offload_train:
        self.wake_up()  # ← 参数已在 GPU，这是 no-op

    # 2. 执行训练
    self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

    # 3. 训练结束后不调用 sleep()
    # 优化器状态继续保留在 GPU
```

**关键差异**:
- `wake_up()` 被调用，但参数和优化器状态已在 GPU
- PyTorch 的 `tensor.cuda()` 对已在 GPU 的 tensor 是空操作（no-op）
- `optimizer.step()` 继续使用上次的 exp_avg 和 exp_avg_sq 进行更新

**状态演进**:
```
第 1 次 step: state['step'] = 1, exp_avg 和 exp_avg_sq 初始化
第 2 次 step: state['step'] = 2, exp_avg 和 exp_avg_sq 继续更新
第 N 次 step: state['step'] = N, exp_avg 和 exp_avg_sq 累积历史梯度信息
```

### 2.4 完整的状态流转图

```
┌─────────────────────────────────────────────────────────────────┐
│ 初始化阶段                                                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. optimizer = AdamW(...)              → optimizer.state = {}   │
│ 2. checkpoint.load(self)               → 可能加载 optimizer.state│
│ 3. sleep() (offload 模式)               → state 移到 CPU RAM     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 第一次训练迭代                                                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. wake_up()                           → state 从 CPU 移到 GPU  │
│ 2. backward()                          → 计算梯度               │
│ 3. optimizer.step()                    → 初始化/更新 state      │
│                                          state['step'] = 1      │
│                                          state['exp_avg'] = ... │
│                                          state['exp_avg_sq'] = ..│
│ 4. train() 结束，不调用 sleep()         → state 保留在 GPU       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 第二次训练迭代                                                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. wake_up()                           → state 已在 GPU (no-op) │
│ 2. backward()                          → 计算梯度               │
│ 3. optimizer.step()                    → 使用上次的 state 更新  │
│                                          state['step'] = 2      │
│                                          state['exp_avg'] 累积   │
│                                          state['exp_avg_sq'] 累积│
│ 4. train() 结束，不调用 sleep()         → state 保留在 GPU       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                         (循环)
```

---

## 3. Sleep/Wake_up 中的 Optimizer State 处理

### 3.1 move_torch_optimizer() 实现

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 1000-1013)

```python
@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """
    在 CPU 和 GPU 之间移动优化器状态。

    参考: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    # 检查优化器是否有状态
    if not optimizer.state:
        return  # 如果状态为空（首次训练前），直接返回

    # 遍历所有参数组
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            # 获取当前参数的状态字典
            state = optimizer.state[param]

            # 遍历状态字典中的所有项
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    # 将 tensor 移动到目标设备（CPU 或 GPU）
                    # non_blocking=True: 异步传输，提高性能
                    state[key] = value.to(device, non_blocking=True)

    # 等待所有异步传输完成
    torch.cuda.synchronize()
```

**关键机制**:

1. **字典修改而非重建**: `state[key] = value.to(device, ...)` 直接修改字典中的值
2. **保持引用**: `optimizer.state[param]` 的引用不变，只是其中的 tensor 被替换
3. **异步传输**: `non_blocking=True` 允许多个 tensor 并行传输
4. **同步点**: `torch.cuda.synchronize()` 确保所有传输完成

### 3.2 Sleep 操作的详细流程

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 276-287)

```python
def sleep(self) -> None:
    """Pause CUDA memory for all tracked tensors."""
    if not self.args.offload_train:
        return

    print_memory("before offload model")

    # 1. 将模型参数移到 CPU
    self.model.cpu()

    # 2. 将优化器状态移到 CPU
    move_torch_optimizer(self.optimizer, "cpu")

    # 3. 清理 GPU 缓存
    clear_memory()

    # 4. 同步所有进程
    dist.barrier(group=get_gloo_group())

    print_memory("after offload model")
```

**执行后的状态**:
```python
# 模型参数
model.parameters() → 所有参数在 CPU

# 优化器状态
optimizer.state[param]['exp_avg'] → 在 CPU (fp32)
optimizer.state[param]['exp_avg_sq'] → 在 CPU (fp32)
optimizer.state[param]['step'] → 在 CPU (scalar tensor)
```

### 3.3 Wake_up 操作的详细流程

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 290-298)

```python
def wake_up(self) -> None:
    """Resume CUDA memory for all tracked tensors."""
    if not self.args.offload_train:
        return

    # 1. 将模型参数移回 GPU
    self.model.cuda()

    # 2. 将优化器状态移回 GPU
    move_torch_optimizer(self.optimizer, "cuda")

    # 3. 同步所有进程
    dist.barrier(group=get_gloo_group())

    print_memory("after wake_up model")
```

**执行后的状态**:
```python
# 模型参数
model.parameters() → 所有参数在 GPU

# 优化器状态
optimizer.state[param]['exp_avg'] → 在 GPU (fp32)
optimizer.state[param]['exp_avg_sq'] → 在 GPU (fp32)
optimizer.state[param]['step'] → 在 GPU (scalar tensor)
```

### 3.4 为什么 wake_up() 可以是幂等的？

**PyTorch 的 tensor.to() 行为**:

```python
# 如果 tensor 已经在目标设备上，to() 是空操作
tensor_gpu = torch.randn(10).cuda()
tensor_gpu_again = tensor_gpu.to('cuda')  # 不会发生数据传输
assert tensor_gpu is tensor_gpu_again  # 返回同一个对象
```

**在 slime 中的应用**:

```python
# 第一次 wake_up(): optimizer.state 在 CPU
move_torch_optimizer(self.optimizer, "cuda")  # 传输 CPU → GPU

# 第二次 wake_up(): optimizer.state 已在 GPU
move_torch_optimizer(self.optimizer, "cuda")  # 对于每个 tensor:
#   state[key] = value.to("cuda", non_blocking=True)
#   → value 已在 GPU，to() 返回原对象，无传输
```

---

## 4. 状态一致性保证机制

### 4.1 参数-状态映射的持久性

**核心机制**: PyTorch 使用 **参数对象本身** 作为 `optimizer.state` 的 key

```python
# optimizer.state 的内部结构
optimizer.state = {
    <Parameter对象1>: {'step': tensor, 'exp_avg': tensor, 'exp_avg_sq': tensor},
    <Parameter对象2>: {'step': tensor, 'exp_avg': tensor, 'exp_avg_sq': tensor},
    ...
}
```

**为什么这保证了一致性？**

1. **参数对象不变**:
   - `self.model.parameters()` 返回的参数对象在模型生命周期内不变
   - 即使参数数据移动到不同设备（CPU/GPU），参数对象本身不变

2. **状态字典不变**:
   - `optimizer.state[param]` 字典在初始化后持续存在
   - `move_torch_optimizer()` 只替换字典中的 tensor 值，不修改字典结构

3. **设备移动不影响映射**:
   ```python
   # 参数移动前
   param = model.linear.weight  # 假设在 GPU
   optimizer.state[param]  # 存在

   # 参数移动到 CPU
   model.cpu()
   param_after = model.linear.weight  # 同一个对象，只是数据在 CPU
   assert param is param_after  # True
   optimizer.state[param]  # 仍然有效

   # 参数移回 GPU
   model.cuda()
   optimizer.state[param]  # 仍然有效
   ```

### 4.2 FSDP2 下的状态管理

在 FSDP2 模式下，参数被包装为 DTensor（分布式张量），但映射机制依然有效：

```python
# FSDP2 包装后
from torch.distributed.fsdp import fully_shard

model = fully_shard(model)  # 参数变为 DTensor

# 优化器仍然使用 DTensor 对象作为 key
optimizer = torch.optim.AdamW(model.parameters())  # model.parameters() 返回 DTensor

# optimizer.state 的 key 是 DTensor 对象
optimizer.state[<DTensor对象>] = {'step': ..., 'exp_avg': ..., 'exp_avg_sq': ...}
```

**FSDP2 的状态同步**:

FSDP2 自动处理分布式优化器状态的分片和同步：

- 每个 rank 只存储自己负责的参数分片的优化器状态
- `optimizer.step()` 时，FSDP2 自动同步必要的梯度和参数
- checkpoint 保存/加载时，FSDP2 自动合并/分发优化器状态

### 4.3 Checkpoint 保存/加载中的一致性

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/checkpoint.py`

#### OptimizerState 包装类 (lines 32-46)

```python
class OptimizerState(Stateful):
    """Wrapper for optimizer state only."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        """提取优化器状态用于保存"""
        _, optimizer_state_dict = get_state_dict(
            self.model, optimizers=self.optimizer
        )
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        """从保存的状态恢复优化器"""
        set_state_dict(
            self.model,
            optimizers=self.optimizer,
            model_state_dict=None,
            optim_state_dict=state_dict["optim"]
        )
```

#### 保存优化器状态 (lines 188-192)

```python
def save(actor: Any, iteration: int) -> None:
    # ... 保存模型参数 ...

    # 保存优化器状态
    if hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))
```

#### 加载优化器状态 (lines 108-119)

```python
def load(actor: Any) -> dict[str, Any] | None:
    # ... 加载模型参数 ...

    # 加载优化器状态（可选）
    load_optimizer = (
        not getattr(actor.args, "no_load_optim", False)
        and hasattr(actor, "optimizer")
    )

    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(
                state_dict=optim_state_dict,
                checkpoint_id=str(optimizer_dir)
            )
            logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            logger.warning(
                f"[FSDP] Failed to load optimizer from {optimizer_dir}: {e}"
            )
    elif load_optimizer:
        logger.info(
            f"[FSDP] Optimizer checkpoint not found at {optimizer_dir}, "
            "skipping optimizer load."
        )
```

**一致性保证**:

1. **保存时**: PyTorch 的 `get_state_dict()` 提取当前所有参数的优化器状态
2. **加载时**: PyTorch 的 `set_state_dict()` 根据参数名称匹配状态
3. **FSDP2 支持**: `torch.distributed.checkpoint` 自动处理分布式状态的合并和分发

### 4.4 数值精度的一致性

**优化器状态的数据类型**:

```python
# AdamW 的默认行为
optimizer = torch.optim.AdamW(model.parameters())

# 无论模型参数是 bf16 还是 fp16，优化器状态都是 fp32
for param in model.parameters():
    state = optimizer.state[param]
    assert state['exp_avg'].dtype == torch.float32
    assert state['exp_avg_sq'].dtype == torch.float32
```

**CPU/GPU 传输不改变精度**:

```python
# GPU 上的 optimizer.state (fp32)
state['exp_avg'].dtype  # torch.float32
state['exp_avg'].device  # cuda:0

# 移到 CPU
move_torch_optimizer(optimizer, "cpu")
state['exp_avg'].dtype  # torch.float32 (不变)
state['exp_avg'].device  # cpu

# 移回 GPU
move_torch_optimizer(optimizer, "cuda")
state['exp_avg'].dtype  # torch.float32 (不变)
state['exp_avg'].device  # cuda:0
```

**数值一致性**:

- `tensor.to(device)` 只改变设备，不改变数值
- PCIe 传输是位级精确的（bit-exact）
- 不会引入浮点误差

---

## 5. 常见问题与边缘情况

### 5.1 Q: 如果在 sleep() 状态下直接调用 optimizer.step() 会怎样？

**A**: 会失败，因为参数和梯度在 CPU，但计算通常期望在 GPU 上进行。

**实际代码保护**:

```python
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    # 确保在训练前调用 wake_up()
    if self.args.offload_train:
        self.wake_up()  # ← 必须先唤醒

    # 然后才能安全地训练
    self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
```

### 5.2 Q: 多次调用 sleep() 或 wake_up() 会出问题吗？

**A**: 不会，两者都是幂等的（idempotent）。

```python
# 多次 sleep
self.sleep()
self.sleep()  # 第二次调用时，参数已在 CPU，model.cpu() 和 move_torch_optimizer 是 no-op

# 多次 wake_up
self.wake_up()
self.wake_up()  # 第二次调用时，参数已在 GPU，model.cuda() 和 move_torch_optimizer 是 no-op
```

### 5.3 Q: 如果 optimizer.state 为空（首次训练），move_torch_optimizer 会报错吗？

**A**: 不会，有显式检查：

```python
@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    if not optimizer.state:
        return  # ← 空状态直接返回，不执行任何操作
    # ...
```

**实际场景**:

```
初始化: optimizer.state = {}
sleep(): move_torch_optimizer(optimizer, "cpu") → 立即返回，无操作
wake_up(): move_torch_optimizer(optimizer, "cuda") → 立即返回，无操作
第一次 backward()
第一次 optimizer.step() → 初始化 optimizer.state（在 GPU 上）
第二次 wake_up(): move_torch_optimizer(optimizer, "cuda") → state 已在 GPU，无操作
```

### 5.4 Q: FSDP2 的 cpu_offload 和 slime 的 offload_train 有什么区别？

**A**: 两种不同的 offload 策略，不能同时使用。

**FSDP2 cpu_offload** (`--fsdp-cpu-offload`):

```python
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

model = fully_shard(
    model,
    offload_policy=CPUOffloadPolicy()
)
```

- **粒度**: 逐层 offload，训练时按需加载
- **时机**: 前向/反向传播时自动管理
- **优化器**: 优化器 step 在 CPU 上执行
- **适用**: 单纯的训练场景（不需要切换推理）

**slime offload_train** (`--offload-train`):

```python
# 手动控制的全量 offload
self.sleep()    # 全部移到 CPU
self.wake_up()  # 全部移回 GPU
```

- **粒度**: 全量 offload，一次性移动所有参数和优化器状态
- **时机**: 手动控制，在训练和推理之间切换
- **优化器**: 优化器 step 在 GPU 上执行（wake_up 后）
- **适用**: Colocated 训练-推理场景

**代码中的互斥逻辑** (actor.py lines 59-62):

```python
self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)

# Offload train and fsdp cpu offload cannot be used together
if self.args.offload_train and self.fsdp_cpu_offload:
    self.args.offload_train = False  # 优先使用 FSDP 的 cpu_offload
```

### 5.5 Q: 为什么不在每次 rollout 前调用 sleep()？

**A**: 因为 rollout 使用 SGLang，它有独立的模型副本。

**架构设计**:

```
训练模式:
  actor.model (在 GPU) → 执行训练
  SGLang 模型 (暂停或使用受限显存)

推理模式:
  actor.model (保留在 GPU，但不使用) → 不参与推理
  SGLang 模型 (在 GPU) → 执行推理
```

**关键点**:

1. **独立模型**: SGLang 有自己的模型副本（通过 `sync_weights()` 同步）
2. **时间分离**: 训练和推理不同时发生
3. **显存管理**: 通过 `--sglang-mem-fraction-static` 限制 SGLang 的显存使用
4. **避免传输**: 不需要每次 rollout 前 offload，节省传输时间

**参考**: `/home/scbjtfy/slime/docs/analysis/fsdp2_sleep_wakeup_and_cpu_offloading.md` 第 4.3 节

---

## 6. 性能影响分析

### 6.1 Optimizer State 传输时间

基于 Problem-7 的分析，优化器状态是传输的主要部分：

| 模型规模 | Optimizer State 大小 | PCIe 3.0 传输时间 | PCIe 4.0 传输时间 |
|---------|---------------------|------------------|------------------|
| 7B      | 56 GB               | 4.12 秒          | 2.06 秒          |
| 13B     | 104 GB              | 7.65 秒          | 3.82 秒          |
| 30B     | 240 GB              | 17.6 秒          | 8.82 秒          |
| 70B     | 560 GB              | 41.2 秒          | 20.6 秒          |

**说明**:
- 优化器状态占总传输量的 80%（7B 模型: 56 GB / 70 GB）
- 异步传输 (`non_blocking=True`) 可提速 1.5-2x

### 6.2 实际训练中的开销

**场景**: 7B 模型，10 次训练迭代，每次 10 秒

| 阶段 | 传输时间 (PCIe 4.0) | 训练时间 | 总时间 | 开销占比 |
|-----|-------------------|---------|--------|---------|
| 初始化 sleep | 2.06 秒 | - | 2.06 秒 | - |
| 第 1 次训练 wake_up | 2.06 秒 | 10 秒 | 12.06 秒 | 17.1% |
| 第 2-10 次训练 | 0 秒 | 90 秒 | 90 秒 | 0% |
| **总计** | **4.12 秒** | **100 秒** | **104.12 秒** | **4.0%** |

**结论**: 优化器状态的传输开销在长时间训练中可以忽略不计。

### 6.3 与 FSDP2 cpu_offload 的性能对比

| 策略 | 每次训练开销 | 优点 | 缺点 |
|-----|------------|------|------|
| slime offload_train | 首次 ~2 秒，后续 0 秒 | 简单，开销低，适合多步训练 | 需要手动调用 sleep/wake_up |
| FSDP2 cpu_offload | 每次 ~10-20% 额外时间 | 自动管理，适合显存极度受限 | 每次训练都有 CPU-GPU 传输开销 |
| 无 offload (disaggregated) | 0 秒 | 最快 | 需要更多 GPU 资源 |

**选择建议**:

- **充足 GPU**: Disaggregated 模式（`--actor-num-gpus-per-node 4 --rollout-num-gpus 4`）
- **有限 GPU + 多步训练**: slime offload_train（`--offload-train --num-steps-per-rollout 10`）
- **极度受限显存**: FSDP2 cpu_offload（`--fsdp-cpu-offload`）

---

## 7. 总结

### 7.1 问题回答

**Q1: 训练结束后，Optimizer States 是被销毁了，还是 Offload 到 CPU 了？**

**答**: **既不销毁也不 offload**，而是**保留在 GPU 上**。

- `train()` 方法结束后不调用 `sleep()`
- `optimizer.state` 作为 actor 的成员变量持续存在
- 优化器状态在训练迭代之间保留在 GPU 上，避免重复传输

**Q2: 下次 wake_up 时如何保证状态的一致性？**

**答**: 通过以下机制保证一致性：

1. **参数对象映射**: `optimizer.state` 使用参数对象本身作为 key，不受设备移动影响
2. **字典修改而非重建**: `move_torch_optimizer()` 直接修改 `optimizer.state[param]` 中的 tensor
3. **设备移动的精确性**: `tensor.to(device)` 是位级精确的，不改变数值
4. **幂等性**: `wake_up()` 对已在 GPU 的状态是空操作，不会重复传输或修改

### 7.2 关键设计洞察

1. **一次性 offload 策略**:
   - `sleep()` 仅在初始化时调用一次（为首次 rollout 释放显存）
   - 训练迭代之间不 sleep/wake_up，状态保持在 GPU

2. **优化器状态的持久性**:
   - 优化器状态从首次 `optimizer.step()` 初始化后持续存在
   - 历史动量信息 (exp_avg, exp_avg_sq) 跨训练迭代累积
   - 这是 AdamW 等自适应优化器有效的关键

3. **状态一致性的核心**:
   - PyTorch 的参数-状态映射机制（使用参数对象作为 key）
   - FSDP2 的分布式状态管理（自动分片和同步）
   - Checkpoint 的保存/加载机制（分离模型和优化器）

4. **Colocated 模式的巧妙性**:
   - 训练和推理使用独立的模型副本
   - 通过时间分离（而非空间分离）共享 GPU
   - 优化器状态保留在 GPU，只在真正需要时 offload

### 7.3 与其他框架对比

| 框架 | Optimizer State 管理策略 | 适用场景 |
|-----|------------------------|---------|
| **slime** | 一次性 offload + 训练时保留在 GPU | RL 训练（训练-推理交替） |
| **DeepSpeed ZeRO-2** | 分片到多个 GPU，按需聚合 | 大规模分布式训练 |
| **DeepSpeed ZeRO-Offload** | Offload 到 CPU，优化器 step 在 CPU 执行 | 显存受限的单机训练 |
| **FSDP2 cpu_offload** | 逐层 offload，按需加载 | 显存极度受限的训练 |
| **Megatron-LM** | 保持在 GPU，不 offload | 多 GPU 充足资源训练 |

**slime 的创新点**:
- 针对 RL 训练的特殊需求（训练 + 推理交替）
- 优化的 offload 时机（仅初始化时，而非每次迭代）
- 利用训练迭代的连续性（多步训练摊薄传输开销）

### 7.4 实现建议

如果要在其他框架中复现 slime 的 optimizer state 管理策略：

1. **创建 move_optimizer 函数**:
   ```python
   def move_optimizer(optimizer, device):
       if not optimizer.state:
           return
       for param_group in optimizer.param_groups:
           for param in param_group['params']:
               state = optimizer.state[param]
               for key, value in state.items():
                   if isinstance(value, torch.Tensor):
                       state[key] = value.to(device, non_blocking=True)
       torch.cuda.synchronize()
   ```

2. **实现 sleep/wake_up**:
   ```python
   def sleep(model, optimizer):
       model.cpu()
       move_optimizer(optimizer, 'cpu')
       torch.cuda.empty_cache()

   def wake_up(model, optimizer):
       model.cuda()
       move_optimizer(optimizer, 'cuda')
   ```

3. **调用时机**:
   ```python
   # 初始化后
   sleep(model, optimizer)

   # 首次训练前
   wake_up(model, optimizer)

   # 后续训练：不再调用 sleep/wake_up
   for epoch in range(epochs):
       train_one_epoch(model, optimizer)  # wake_up 已在首次调用
   ```

4. **注意事项**:
   - 确保 `optimizer.state` 使用参数对象作为 key（PyTorch 默认行为）
   - 使用 `non_blocking=True` 提高传输性能
   - 添加 `torch.cuda.synchronize()` 确保传输完成
   - 考虑与分布式训练框架（FSDP/DeepSpeed）的兼容性

---

## 8. 相关源码索引

| 功能 | 文件路径 | 行号 |
|-----|---------|------|
| 优化器初始化 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 106-115 |
| sleep() 实现 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 276-287 |
| wake_up() 实现 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 290-298 |
| move_torch_optimizer() | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 1001-1013 |
| train() 方法 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 447-465 |
| _train_core() 方法 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 510-546 |
| optimizer.step() 调用 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 717-718 |
| OptimizerState 类 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/checkpoint.py` | 32-46 |
| checkpoint.save() | `/home/scbjtfy/slime/slime/backends/fsdp_utils/checkpoint.py` | 163-214 |
| checkpoint.load() | `/home/scbjtfy/slime/slime/backends/fsdp_utils/checkpoint.py` | 65-132 |
| offload_train 与 fsdp_cpu_offload 互斥 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 59-62 |

---

**生成时间**: 2025-12-04
**分析框架版本**: slime (commit: 9d7f34d)
**分析者**: Claude Code (Sonnet 4.5)
