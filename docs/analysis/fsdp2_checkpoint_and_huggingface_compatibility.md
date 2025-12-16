# FSDP2 Checkpoint 与 HuggingFace 格式兼容性深度解析

> **面向对象**: Infra 工程师，目标是在其他框架中复现/支持 FSDP2 后端
> **作者**: 基于 slime 源码分析
> **日期**: 2025-12-03
> **版本**: v1.0

本文档深入分析 FSDP2 中 **DTensor 的结构保持机制**、**Checkpoint 保存/加载流程**，以及**与 HuggingFace 格式的兼容性**。

---

## 问题-2 回答：DTensor 保持原始结构的便利性

### 核心答案

**是的！DTensor 保持原始 shape 意味着 state_dict 的 key 和 shape 完全不需要改变。**

这是 FSDP2 相对于 FSDP1 的**最大优势**之一。

---

## 1. FSDP1 vs FSDP2：关键区别

### 1.1 FSDP1 的问题：FlatParameter

**FSDP1 的实现方式**：

```python
# FSDP1 将一个 module 的所有参数摊平成一个巨大的 FlatParameter

# 原始模型（以一个 Transformer Layer 为例）：
layer = TransformerLayer(
    self_attn=MultiHeadAttention(
        q_proj=Linear(3584, 3584),      # weight: [3584, 3584]
        k_proj=Linear(3584, 3584),      # weight: [3584, 3584]
        v_proj=Linear(3584, 3584),      # weight: [3584, 3584]
        o_proj=Linear(3584, 3584),      # weight: [3584, 3584]
    ),
    mlp=MLP(
        gate_proj=Linear(3584, 18944),  # weight: [18944, 3584]
        up_proj=Linear(3584, 18944),    # weight: [18944, 3584]
        down_proj=Linear(18944, 3584),  # weight: [3584, 18944]
    )
)

# FSDP1 的处理：
# 将所有 weight 摊平成一个 1D 向量
flat_param = FlatParameter(
    data=torch.cat([
        q_proj.weight.flatten(),
        k_proj.weight.flatten(),
        v_proj.weight.flatten(),
        o_proj.weight.flatten(),
        gate_proj.weight.flatten(),
        up_proj.weight.flatten(),
        down_proj.weight.flatten(),
    ]),
    # metadata 记录如何还原
)

# state_dict 变成：
{
    "_fsdp_wrapped_module.flat_param": Tensor([...])  # 巨大的 1D 向量
}
```

**FSDP1 的缺点**：

1. **state_dict key 改变**：
   - 原始：`layer.self_attn.q_proj.weight`
   - FSDP1：`_fsdp_wrapped_module.flat_param`

2. **Shape 信息丢失**：
   - 原始：`[3584, 3584]`
   - FSDP1：需要 metadata 来还原 shape

3. **不兼容 HuggingFace**：
   - 无法直接 `model.save_pretrained()`
   - 需要复杂的转换逻辑

4. **Padding 复杂**：
   - 为了对齐，可能需要在 FlatParameter 中添加 padding
   - 增加内存浪费

5. **元数据易失**：
   - 如果 metadata 丢失或不匹配，无法正确还原参数

---

### 1.2 FSDP2 的改进：DTensor

**FSDP2 的实现方式**：

```python
# FSDP2 使用 DTensor，保持原始结构

# 原始模型（同上）
layer = TransformerLayer(...)

# FSDP2 的处理：
# 每个参数独立包装为 DTensor，保持 shape 不变
layer.self_attn.q_proj.weight = DTensor(
    local_data=torch.randn(3584 // dp_size, 3584),  # 本地分片
    device_mesh=mesh,
    placements=[Shard(0)],  # 在 dim=0 上切分
    # 关键：全局 shape 仍然是 [3584, 3584]
)

# state_dict 保持不变：
{
    "layer.self_attn.q_proj.weight": DTensor(global_shape=[3584, 3584]),
    "layer.self_attn.k_proj.weight": DTensor(global_shape=[3584, 3584]),
    "layer.self_attn.v_proj.weight": DTensor(global_shape=[3584, 3584]),
    "layer.self_attn.o_proj.weight": DTensor(global_shape=[3584, 3584]),
    "layer.mlp.gate_proj.weight": DTensor(global_shape=[18944, 3584]),
    "layer.mlp.up_proj.weight": DTensor(global_shape=[18944, 3584]),
    "layer.mlp.down_proj.weight": DTensor(global_shape=[3584, 18944]),
}
```

**FSDP2 的优势**：

1. ✅ **state_dict key 完全不变**：`layer.self_attn.q_proj.weight`
2. ✅ **Shape 完全不变**：`[3584, 3584]`
3. ✅ **兼容 HuggingFace**：可以直接保存/加载
4. ✅ **无需 metadata**：shape 和 stride 信息保留在 DTensor 中
5. ✅ **无 Padding**：每个参数独立切分，无需对齐

---

### 1.3 DTensor 如何保持 shape 和 stride？

**DTensor 的内部结构**：

```python
class DTensor:
    def __init__(self, local_data, device_mesh, placements):
        self._local_data = local_data           # 本地分片
        self._device_mesh = device_mesh         # DeviceMesh
        self._placements = placements           # [Shard(0)]

        # 关键：保存全局 shape 和 stride
        self._global_shape = self._compute_global_shape()
        self._global_stride = self._compute_global_stride()

    @property
    def shape(self):
        """返回全局 shape（不是本地 shape）"""
        return self._global_shape

    @property
    def stride(self):
        """返回全局 stride（不是本地 stride）"""
        return self._global_stride

    def to_local(self):
        """返回本地分片（本地 shape）"""
        return self._local_data

    def full_tensor(self):
        """All-gather，返回完整 tensor（全局 shape）"""
        # 触发通信，收集所有分片
        return self._all_gather()
```

**关键点**：

- **对外接口**（`tensor.shape`）：返回**全局 shape**
- **内部存储**（`tensor._local_data`）：只存储**本地分片**
- **通信触发**（`tensor.full_tensor()`）：按需 all-gather

**示例**：

```python
# 假设 Linear(3584, 3584)，DP=4
weight = DTensor(
    local_data=torch.randn(896, 3584),  # 本地 shape: [896, 3584]
    device_mesh=mesh,
    placements=[Shard(0)],
)

# 对外接口：
print(weight.shape)        # torch.Size([3584, 3584])  ← 全局 shape
print(weight.stride())     # (3584, 1)                 ← 全局 stride
print(weight.dtype)        # torch.bfloat16

# 内部存储：
print(weight.to_local().shape)  # torch.Size([896, 3584])  ← 本地 shape

# 通信：
full_weight = weight.full_tensor()  # All-gather
print(full_weight.shape)   # torch.Size([3584, 3584])  ← 完整 shape
```

---

## 2. Checkpoint 保存流程

### 2.1 slime 的实现

**代码位置**：`checkpoint.py:163-214`

```python
def save(actor: Any, iteration: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    """
    torch.cuda.synchronize()

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"

    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # ====== 关键 1: 包装 model 为 ModelState ======
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    # ====== 关键 2: 使用 dcp.save 保存 ======
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    # ====== 关键 3: 保存 optimizer ======
    if hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    # ====== 关键 4: 保存 metadata ======
    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        logger.info(f"[FSDP] Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
```

**关键类：`ModelState`**：

```python
class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        # ====== 关键：使用 get_state_dict 获取 FSDP2 的 state_dict ======
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        # ====== 关键：使用 set_state_dict 恢复 FSDP2 的 state_dict ======
        set_state_dict(
            self.model,
            optimizers=[],
            model_state_dict=state_dict["model"],
            optim_state_dict=None
        )
```

---

### 2.2 `get_state_dict` 的工作机制

**PyTorch 的 API**：

```python
from torch.distributed.checkpoint.state_dict import get_state_dict

# 获取 FSDP2 模型的 state_dict
model_state_dict, optimizer_state_dict = get_state_dict(
    model,
    optimizers=optimizer,
    options=StateDictOptions(...)
)
```

**内部流程**：

```python
# 伪代码（简化）
def get_state_dict(model, optimizers, options=None):
    state_dict = {}

    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            # ====== 关键 1: DTensor 转换为普通 Tensor ======
            if options.full_state_dict:
                # 选项 A: 全量 state_dict（默认）
                # 触发 all-gather，获取完整 tensor
                full_tensor = param.full_tensor()
                state_dict[name] = full_tensor  # key 不变，shape 是全局 shape

            else:
                # 选项 B: Sharded state_dict
                # 只保存本地分片 + metadata
                state_dict[name] = {
                    "local_data": param.to_local(),
                    "placements": param.placements,
                    "device_mesh": param.device_mesh,
                }
        else:
            # 普通 Tensor 直接保存
            state_dict[name] = param

    return state_dict, optimizer_state_dict
```

**关键点**：

1. **full_state_dict=True**（默认）：
   - 每个 DTensor 调用 `full_tensor()` 进行 all-gather
   - 返回**完整的 tensor**，shape 是**全局 shape**
   - **key 完全不变**：`layer.self_attn.q_proj.weight`

2. **full_state_dict=False**（Sharded）：
   - 每个 rank 只保存**本地分片** + metadata
   - 节省存储空间和通信
   - 加载时需要重建 DTensor

**slime 的选择**：

```python
# slime 使用 full_state_dict=True
# 在 checkpoint.py 中没有显式指定，使用默认值

# 原因：
# 1. 兼容性更好（与 HuggingFace 格式一致）
# 2. 易于调试（可以直接查看 checkpoint）
# 3. 可以跨不同的 DP size 加载
```

---

### 2.3 Checkpoint 的目录结构

**实际保存的文件**：

```
/root/shared_data/run_xxx/checkpoints/
├── latest_checkpointed_iteration.txt  # 最新的 iteration
├── iter_0000001/
│   ├── model/                          # 模型权重
│   │   ├── __0_0.distcp              # Rank 0 的数据（分布式格式）
│   │   ├── __1_0.distcp              # Rank 1 的数据
│   │   ├── ...
│   │   ├── .metadata                 # Metadata（描述如何组装）
│   ├── optimizer/                     # 优化器状态
│   │   ├── __0_0.distcp
│   │   ├── __1_0.distcp
│   │   ├── ...
│   │   ├── .metadata
│   ├── rng.pt                         # RNG 状态
│   └── meta.json                      # Checkpoint metadata
└── iter_0000002/
    └── ...
```

**`.distcp` 文件格式**：

```python
# PyTorch Distributed Checkpoint 的自定义格式
# 每个文件包含：
{
    "state_dict": {
        # 该 rank 负责的参数（全量或分片）
        "layer.0.self_attn.q_proj.weight": Tensor([3584, 3584]),
        "layer.0.self_attn.k_proj.weight": Tensor([3584, 3584]),
        ...
    },
    "metadata": {
        # 参数的元信息（placement, mesh, etc.）
    }
}
```

**`.metadata` 文件**：

```python
# 描述如何从各个 rank 的文件中组装完整的 state_dict
{
    "layer.0.self_attn.q_proj.weight": {
        "shard_metadata": [
            {"rank": 0, "file": "__0_0.distcp", "offset": 0, "shape": [896, 3584]},
            {"rank": 1, "file": "__1_0.distcp", "offset": 0, "shape": [896, 3584]},
            {"rank": 2, "file": "__2_0.distcp", "offset": 0, "shape": [896, 3584]},
            {"rank": 3, "file": "__3_0.distcp", "offset": 0, "shape": [896, 3584]},
        ],
        "global_shape": [3584, 3584],
    },
    ...
}
```

---

## 3. Checkpoint 加载流程

### 3.1 slime 的实现

**代码位置**：`checkpoint.py:65-132`

```python
def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk."""
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"[FSDP] Checkpoint directory {root_path} not found; skipping load.")
        return None

    # ====== 关键 1: 确定要加载的 iteration ======
    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"[FSDP] No tracker file at {tracker_file}; skipping load.")
            return None
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"

    if not model_dir.exists():
        logger.info(f"[FSDP] Model checkpoint {model_dir} not found; skipping load.")
        return None

    # ====== 关键 2: 加载模型权重 ======
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"[FSDP] Loaded model from {model_dir}")
    except Exception as e:
        logger.error(f"[FSDP] Failed to load model from {model_dir}: {e}")
        return None

    # ====== 关键 3: 加载优化器状态（可选）======
    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load optimizer from {optimizer_dir}: {e}")

    # ====== 关键 4: 加载 metadata ======
    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
    }
```

---

### 3.2 `set_state_dict` 的工作机制

**PyTorch 的 API**：

```python
from torch.distributed.checkpoint.state_dict import set_state_dict

# 恢复 FSDP2 模型的 state_dict
set_state_dict(
    model,
    optimizers=optimizer,
    model_state_dict=loaded_state_dict,
    optim_state_dict=loaded_optim_state_dict,
    options=StateDictOptions(...)
)
```

**内部流程**：

```python
# 伪代码（简化）
def set_state_dict(model, optimizers, model_state_dict, optim_state_dict, options):
    for name, param in model.named_parameters():
        if name in model_state_dict:
            loaded_tensor = model_state_dict[name]

            if isinstance(param, DTensor):
                # ====== 关键 1: 将普通 Tensor 转换为 DTensor ======
                if options.full_state_dict:
                    # 从完整 tensor 恢复
                    # 方式 1: Rank 0 有完整数据，broadcast 到其他 rank
                    if options.broadcast_from_rank0:
                        if dist.get_rank() == 0:
                            full_tensor = loaded_tensor
                        else:
                            full_tensor = torch.empty_like(param.to_local())
                        dist.broadcast(full_tensor, src=0)

                        # 切分到本地分片
                        local_shard = shard_tensor(full_tensor, param.placements, param.device_mesh)
                        param.data.copy_(local_shard)

                    # 方式 2: 每个 rank 都有完整数据（或从磁盘读取）
                    else:
                        local_shard = shard_tensor(loaded_tensor, param.placements, param.device_mesh)
                        param.data.copy_(local_shard)

                else:
                    # 从 sharded state_dict 恢复
                    # 每个 rank 直接加载自己的分片
                    local_data = loaded_tensor["local_data"]
                    param.data.copy_(local_data)

            else:
                # 普通 Tensor 直接复制
                param.data.copy_(loaded_tensor)
```

**关键点**：

1. **full_state_dict=True**：
   - 输入是**完整的 tensor**（全局 shape）
   - 自动切分到各个 rank 的本地分片
   - **key 完全不变**

2. **broadcast_from_rank0=True**：
   - Rank 0 从磁盘加载完整 tensor
   - 其他 rank 接收 broadcast
   - 减少磁盘 I/O（只有 rank 0 读取）

---

### 3.3 初始化时的加载流程

**代码位置**：`actor.py:82-143`

```python
# 在 init 方法中：

# 步骤 1: 加载原始 HuggingFace 模型
init_context = self._get_init_weight_context_manager()
with init_context():
    model = AutoModelForCausalLM.from_pretrained(
        self.args.hf_checkpoint,  # 从 HuggingFace checkpoint 加载
        trust_remote_code=True,
        attn_implementation=self.args.attn_implementation,
    )

model.train()
full_state = model.state_dict()  # Rank 0 有完整权重，其他 rank 为空

# 步骤 2: 应用 FSDP2 包装
model = apply_fsdp2(
    model,
    mesh=self.dp_mesh,
    cpu_offload=self.fsdp_cpu_offload
)
# 此时 model 的参数已经是 DTensor，但数据还没加载

# 步骤 3: 从 rank 0 广播权重到所有 rank
model = self._fsdp2_load_full_state_dict(
    model,
    full_state,  # Rank 0 有数据，其他 rank 为空 dict
    self.dp_mesh,
    cpu_offload=True if self.fsdp_cpu_offload else None
)
# 此时每个 rank 的 DTensor 都包含正确的本地分片

# 步骤 4: 从 FSDP checkpoint 恢复（如果有）
checkpoint_payload = checkpoint.load(self)
# 这会覆盖步骤 3 的权重（如果指定了 --load）
```

**`_fsdp2_load_full_state_dict` 的实现**：

```python
def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
    """Load full state dict into FSDP2 model with efficient broadcast from rank 0."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # Rank 0: move with weights, others: allocate empty tensors on device
    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        # to_empty creates tensors on device without initializing memory
        model = model.to_empty(device=torch.cuda.current_device())

    is_cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True,          # 输入是完整的 state_dict
        cpu_offload=is_cpu_offload,    # 是否 offload 到 CPU
        broadcast_from_rank0=True      # 从 rank 0 广播
    )

    # ====== 关键：调用 set_model_state_dict ======
    set_model_state_dict(model, full_state, options=options)

    # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
    for _name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if is_cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(torch.cuda.current_device())

    return model
```

**关键点**：

1. **输入格式**：
   - `full_state` 是标准的 HuggingFace state_dict
   - Key: `layer.0.self_attn.q_proj.weight`
   - Value: Tensor, shape=[3584, 3584]（完整）

2. **输出格式**：
   - Model 的参数变成 DTensor
   - 每个 rank 只存储本地分片
   - 但 key 和全局 shape 完全不变

3. **无需转换**：
   - 直接从 HuggingFace checkpoint 加载
   - `set_model_state_dict` 自动处理切分

---

## 4. 与 HuggingFace 格式的兼容性

### 4.1 完全兼容

**核心结论**：

✅ **state_dict 的 key 和 shape 完全兼容**：
- FSDP2 保存的 checkpoint 可以直接转换为 HuggingFace 格式
- HuggingFace 的 checkpoint 可以直接加载到 FSDP2 模型

**原因**：

1. **Key 不变**：
   - FSDP2：`layer.0.self_attn.q_proj.weight`
   - HF：`layer.0.self_attn.q_proj.weight`

2. **Shape 不变**：
   - FSDP2：`[3584, 3584]`（全局 shape）
   - HF：`[3584, 3584]`

3. **Dtype 不变**：
   - FSDP2：`torch.bfloat16`（可配置）
   - HF：`torch.bfloat16`（或其他）

---

### 4.2 转换为 HuggingFace 格式

**方式 1：使用 slime 的工具**

slime 提供了转换脚本（类似 Megatron）：

```bash
# 转换 FSDP checkpoint 为 HuggingFace 格式
# （具体脚本可能在 tools/ 目录下）

# 伪代码示例：
python tools/convert_fsdp_to_hf.py \
    --fsdp-checkpoint /path/to/fsdp/checkpoint \
    --output-dir /path/to/hf/checkpoint
```

**方式 2：直接使用 PyTorch API**

```python
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict
from transformers import AutoModelForCausalLM

# 步骤 1: 加载 FSDP checkpoint
model = ...  # FSDP2-wrapped model
checkpoint_dir = "/path/to/fsdp/checkpoint/iter_0000001/model"

model_state = ModelState(model)
state_dict = {"model_state": model_state}
dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)

# 步骤 2: 获取完整的 state_dict（触发 all-gather）
full_state_dict, _ = get_state_dict(
    model,
    optimizers=[],
    options=StateDictOptions(full_state_dict=True)
)

# 步骤 3: 保存为 HuggingFace 格式
# 方式 A: 使用 torch.save
torch.save(full_state_dict, "/path/to/hf/pytorch_model.bin")

# 方式 B: 使用 HuggingFace API（如果模型支持）
# 需要先 unwrap FSDP
unwrapped_model = model  # 或 model._fsdp_wrapped_module
unwrapped_model.save_pretrained("/path/to/hf")
```

**方式 3：在保存时直接使用 full_state_dict**

```python
# 在 checkpoint.py 的 save 函数中：

# 如果需要同时保存 HuggingFace 格式
if dist.get_rank() == 0:
    # 获取完整的 state_dict
    full_state_dict, _ = get_state_dict(
        actor.model,
        optimizers=[],
        options=StateDictOptions(full_state_dict=True)
    )

    # 保存为 HuggingFace 格式
    hf_dir = checkpoint_dir / "hf"
    hf_dir.mkdir(exist_ok=True)

    # 保存 weights
    torch.save(full_state_dict, hf_dir / "pytorch_model.bin")

    # 保存 config（从原始 HF checkpoint 复制）
    config_path = Path(actor.args.hf_checkpoint) / "config.json"
    if config_path.exists():
        shutil.copy(config_path, hf_dir / "config.json")

    # 保存 tokenizer（从原始 HF checkpoint 复制）
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for file in tokenizer_files:
        src = Path(actor.args.hf_checkpoint) / file
        if src.exists():
            shutil.copy(src, hf_dir / file)
```

---

### 4.3 从 HuggingFace 格式加载

**直接加载（无需转换）**：

```python
# 在 actor.py 的 init 方法中，已经支持：

model = AutoModelForCausalLM.from_pretrained(
    self.args.hf_checkpoint,  # 直接指定 HuggingFace checkpoint
    trust_remote_code=True,
    attn_implementation=self.args.attn_implementation,
)

# 后续步骤自动处理：
# 1. apply_fsdp2() 将参数转换为 DTensor
# 2. _fsdp2_load_full_state_dict() 从 rank 0 广播权重
# 3. 每个 rank 自动切分到本地分片
```

**Megatron 需要转换，FSDP2 不需要**：

| 特性 | Megatron | FSDP2 |
|------|----------|-------|
| 加载 HF checkpoint | ❌ 需要转换为 `torch_dist` 格式 | ✅ 直接加载 |
| 保存 HF checkpoint | ❌ 需要转换工具 | ✅ 直接保存（或简单转换）|
| checkpoint 格式 | `torch_dist`（专有格式） | PyTorch DCP（标准格式）|
| 跨 DP size 加载 | ❌ 困难 | ✅ 容易（full_state_dict）|

---

## 5. 实际示例：state_dict 对比

### 5.1 HuggingFace 原始 state_dict

```python
# 从 HuggingFace 加载的模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
state_dict = model.state_dict()

# 输出示例：
{
    "model.embed_tokens.weight": Tensor([151936, 3584]),
    "model.layers.0.self_attn.q_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.self_attn.k_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.self_attn.v_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.self_attn.o_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.mlp.gate_proj.weight": Tensor([18944, 3584]),
    "model.layers.0.mlp.up_proj.weight": Tensor([18944, 3584]),
    "model.layers.0.mlp.down_proj.weight": Tensor([3584, 18944]),
    "model.layers.0.input_layernorm.weight": Tensor([3584]),
    "model.layers.0.post_attention_layernorm.weight": Tensor([3584]),
    ...
    "model.layers.39.self_attn.q_proj.weight": Tensor([3584, 3584]),
    ...
    "model.norm.weight": Tensor([3584]),
    "lm_head.weight": Tensor([151936, 3584]),
}
```

---

### 5.2 FSDP2 包装后的 state_dict

```python
# 应用 FSDP2 包装
model = apply_fsdp2(model, mesh=dp_mesh, cpu_offload=False)

# 获取 state_dict（full_state_dict=True）
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
state_dict, _ = get_state_dict(
    model,
    optimizers=[],
    options=StateDictOptions(full_state_dict=True)
)

# 输出示例（完全相同！）：
{
    "model.embed_tokens.weight": Tensor([151936, 3584]),  # ← Key 不变，Shape 不变
    "model.layers.0.self_attn.q_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.self_attn.k_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.self_attn.v_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.self_attn.o_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.mlp.gate_proj.weight": Tensor([18944, 3584]),
    "model.layers.0.mlp.up_proj.weight": Tensor([18944, 3584]),
    "model.layers.0.mlp.down_proj.weight": Tensor([3584, 18944]),
    "model.layers.0.input_layernorm.weight": Tensor([3584]),
    "model.layers.0.post_attention_layernorm.weight": Tensor([3584]),
    ...
}
```

**关键观察**：

✅ **Key 完全相同**
✅ **Shape 完全相同**（全局 shape）
✅ **无需任何转换**

---

### 5.3 FSDP2 内部的 DTensor（本地分片）

```python
# 查看实际的参数类型
for name, param in model.named_parameters():
    print(f"{name}:")
    print(f"  Type: {type(param)}")                    # DTensor
    print(f"  Global shape: {param.shape}")            # 全局 shape
    print(f"  Local shape: {param.to_local().shape}")  # 本地 shape
    print(f"  Placements: {param.placements}")         # [Shard(0)]
    print()

# 输出示例（Rank 0, DP=4）：
model.layers.0.self_attn.q_proj.weight:
  Type: <class 'torch.distributed.tensor.DTensor'>
  Global shape: torch.Size([3584, 3584])     # ← 对外接口：全局 shape
  Local shape: torch.Size([896, 3584])       # ← 内部存储：本地 shape（1/4）
  Placements: [Shard(0)]

model.layers.0.mlp.gate_proj.weight:
  Type: <class 'torch.distributed.tensor.DTensor'>
  Global shape: torch.Size([18944, 3584])    # ← 对外接口：全局 shape
  Local shape: torch.Size([4736, 3584])      # ← 内部存储：本地 shape（1/4）
  Placements: [Shard(0)]
```

**关键点**：

- **对外接口**（`param.shape`）：返回**全局 shape**（与 HF 一致）
- **内部存储**（`param.to_local()`）：只存储**本地分片**
- **保存时**：`get_state_dict` 触发 all-gather，保存**完整 tensor**

---

## 6. FSDP1 vs FSDP2：Checkpoint 对比

### 6.1 FSDP1 的 Checkpoint

```python
# FSDP1 保存的 state_dict（简化示例）
{
    "_fsdp_wrapped_module.flat_param_0": Tensor([1234567890]),  # 巨大的 1D 向量
    "_fsdp_wrapped_module.flat_param_1": Tensor([9876543210]),
    "_fsdp_wrapped_module._metadata": {
        # 复杂的 metadata，描述如何还原参数
        "param_shapes": [...],
        "param_names": [...],
        "padding_size": 1024,
        ...
    }
}
```

**问题**：

1. ❌ Key 改变：`_fsdp_wrapped_module.flat_param_0`（无法识别）
2. ❌ Shape 改变：`[1234567890]`（1D 向量，无法识别原始 shape）
3. ❌ 需要 metadata：丢失 metadata 则无法还原
4. ❌ 不兼容 HF：需要复杂的转换逻辑

---

### 6.2 FSDP2 的 Checkpoint

```python
# FSDP2 保存的 state_dict（与 HF 完全相同）
{
    "model.embed_tokens.weight": Tensor([151936, 3584]),
    "model.layers.0.self_attn.q_proj.weight": Tensor([3584, 3584]),
    "model.layers.0.self_attn.k_proj.weight": Tensor([3584, 3584]),
    ...
}
```

**优势**：

1. ✅ Key 不变：完全兼容 HF
2. ✅ Shape 不变：全局 shape
3. ✅ 无需 metadata：shape 和 stride 保留在 DTensor 中
4. ✅ 直接兼容 HF：无需转换

---

## 7. 在其他框架中复现的关键点

### 7.1 必须实现的功能

1. **DTensor 的 shape 属性**：
   - `tensor.shape` 返回**全局 shape**
   - `tensor.to_local()` 返回**本地分片**

2. **get_state_dict 的 full_state_dict 模式**：
   - 自动 all-gather 所有参数
   - 返回完整的 state_dict（key 和 shape 不变）

3. **set_state_dict 的 broadcast_from_rank0 模式**：
   - Rank 0 加载完整 state_dict
   - 自动切分并广播到其他 rank

4. **PyTorch Distributed Checkpoint (dcp)**：
   - 支持分布式保存/加载
   - 生成 `.distcp` 和 `.metadata` 文件

### 7.2 关键挑战

1. **DTensor 的实现**：
   - 保持全局 shape 的同时，只存储本地分片
   - 正确处理 stride 和 memory layout

2. **All-Gather 的优化**：
   - 避免重复 all-gather
   - 支持异步操作

3. **与现有生态的兼容**：
   - 确保 state_dict 格式与 HuggingFace 一致
   - 支持直接加载 HF checkpoint

### 7.3 实现建议

**步骤 1：实现 DTensor**

```python
class DTensor:
    def __init__(self, local_data, global_shape, global_stride, placements, device_mesh):
        self._local_data = local_data
        self._global_shape = global_shape  # 关键：保存全局 shape
        self._global_stride = global_stride
        self._placements = placements
        self._device_mesh = device_mesh

    @property
    def shape(self):
        """对外接口：返回全局 shape"""
        return self._global_shape

    @property
    def stride(self):
        """对外接口：返回全局 stride"""
        return self._global_stride

    def to_local(self):
        """返回本地分片"""
        return self._local_data

    def full_tensor(self):
        """All-gather，返回完整 tensor"""
        return self._all_gather()
```

**步骤 2：实现 get_state_dict**

```python
def get_state_dict(model, optimizers, options):
    state_dict = {}

    for name, param in model.named_parameters():
        if isinstance(param, DTensor) and options.full_state_dict:
            # All-gather 获取完整 tensor
            state_dict[name] = param.full_tensor()
        else:
            state_dict[name] = param

    return state_dict, optimizer_state_dict
```

**步骤 3：实现 set_state_dict**

```python
def set_state_dict(model, optimizers, model_state_dict, optim_state_dict, options):
    for name, param in model.named_parameters():
        if name in model_state_dict:
            loaded_tensor = model_state_dict[name]

            if isinstance(param, DTensor):
                if options.broadcast_from_rank0:
                    # Rank 0 广播到其他 rank
                    if dist.get_rank() == 0:
                        full_tensor = loaded_tensor
                    else:
                        full_tensor = torch.empty(param.shape, dtype=param.dtype, device=param.device)
                    dist.broadcast(full_tensor, src=0)

                # 切分到本地分片
                local_shard = shard_tensor(full_tensor, param.placements, param.device_mesh)
                param.data.copy_(local_shard)
            else:
                param.data.copy_(loaded_tensor)
```

---

## 8. 总结

### 8.1 回答问题-2 的核心要点

**是的！DTensor 保持原始结构意味着 state_dict 的 key 和 shape 完全不需要改变。**

**具体来说**：

1. ✅ **Key 完全不变**：
   - FSDP2：`model.layers.0.self_attn.q_proj.weight`
   - HF：`model.layers.0.self_attn.q_proj.weight`

2. ✅ **Shape 完全不变**：
   - FSDP2：`[3584, 3584]`（全局 shape）
   - HF：`[3584, 3584]`

3. ✅ **Dtype 完全不变**：
   - 可配置，通常是 `torch.bfloat16`

4. ✅ **直接兼容 HuggingFace**：
   - 无需转换工具
   - 无需 metadata
   - 直接保存/加载

---

### 8.2 FSDP2 的核心优势

**相对于 FSDP1**：

1. **State_dict 兼容性**：
   - FSDP1：完全不兼容（FlatParameter）
   - FSDP2：完全兼容（DTensor）

2. **Metadata 依赖**：
   - FSDP1：必须依赖 metadata
   - FSDP2：shape 和 stride 保留在 DTensor 中

3. **Padding 开销**：
   - FSDP1：需要 padding 对齐
   - FSDP2：每个参数独立切分

4. **HuggingFace 支持**：
   - FSDP1：需要复杂转换
   - FSDP2：开箱即用

**相对于 Megatron**：

1. **权重格式**：
   - Megatron：需要转换为 `torch_dist`
   - FSDP2：直接使用 HF 格式

2. **Checkpoint 工具**：
   - Megatron：需要专门的转换脚本
   - FSDP2：使用标准的 PyTorch API

3. **跨 DP size 加载**：
   - Megatron：困难（需要重新切分）
   - FSDP2：容易（full_state_dict）

---

### 8.3 Checkpoint 流程总结

**保存流程**：

```
DTensor (Shard) → get_state_dict → All-Gather → Full Tensor → dcp.save → 磁盘
```

**加载流程**：

```
磁盘 → dcp.load → Full Tensor → set_state_dict → 切分 → DTensor (Shard)
```

**关键点**：

- **保存时**：自动 all-gather 成完整 tensor
- **加载时**：自动切分到各个 rank
- **全程**：key 和 shape 完全不变

---

## 附录

### A. 相关代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| Checkpoint 保存 | `checkpoint.py` | 163-214 |
| Checkpoint 加载 | `checkpoint.py` | 65-132 |
| ModelState | `checkpoint.py` | 18-29 |
| OptimizerState | `checkpoint.py` | 32-46 |
| _fsdp2_load_full_state_dict | `actor.py` | 236-273 |

### B. 关键 PyTorch API

```python
# State dict 操作
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions

# Distributed checkpoint
import torch.distributed.checkpoint as dcp

# DTensor
from torch.distributed.tensor import DTensor, Shard, Replicate
```

### C. 调试技巧

```python
# 1. 检查参数类型
for name, param in model.named_parameters():
    if isinstance(param, DTensor):
        print(f"{name}: DTensor, global_shape={param.shape}, local_shape={param.to_local().shape}")
    else:
        print(f"{name}: Tensor, shape={param.shape}")

# 2. 对比 state_dict
hf_state_dict = hf_model.state_dict()
fsdp_state_dict, _ = get_state_dict(fsdp_model, optimizers=[], options=StateDictOptions(full_state_dict=True))

# 检查 key 是否一致
assert set(hf_state_dict.keys()) == set(fsdp_state_dict.keys())

# 检查 shape 是否一致
for key in hf_state_dict.keys():
    assert hf_state_dict[key].shape == fsdp_state_dict[key].shape, f"{key}: {hf_state_dict[key].shape} != {fsdp_state_dict[key].shape}"

# 检查数值是否一致（如果从同一个 checkpoint 加载）
for key in hf_state_dict.keys():
    assert torch.allclose(hf_state_dict[key], fsdp_state_dict[key], atol=1e-5), f"{key}: values differ"
```

---

**文档生成日期**: 2025-12-03
**Slime 版本**: main branch (commit 9d7f34d)
**作者**: 基于源码分析生成
