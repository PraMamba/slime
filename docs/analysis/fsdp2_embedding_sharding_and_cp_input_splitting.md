# FSDP2 Embedding 层分片与 CP Input 切分机制分析

## Problem: Embedding 层的计算与存储方式

### 问题描述

在 forward 之前，input_ids 被切分成了 cp_size 份。这是否意味着 Embedding 层也是被切分计算的，还是每张卡都存了一份完整的 Embedding Table？

### 核心发现总结

1. **Embedding Table 是分片存储的**: 每张卡只存部分 vocab 的 embedding（在 dp 维度分片）
2. **CP 维度的卡存相同分片**: 同一 dp_rank 的不同 cp_rank 存储相同的 Embedding 分片
3. **input_ids 在 CP 维度切分**: 但不影响 Embedding lookup（通过 all-gather 解决）
4. **FSDP2 自动 All-Gather**: 前向传播时临时聚合完整 Embedding Table
5. **通信开销**: 每次前向传播需要 all-gather，通信量 = embedding_size * (dp_size - 1) / dp_size

---

## 1. input_ids 的切分机制

### 1.1 _get_model_inputs_args() 方法

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 811-831)

```python
def _get_model_inputs_args(self, packed_sequence: dict) -> dict:
    # 初始化：添加 batch 维度
    input_ids = packed_sequence["tokens"].unsqueeze(0)       # [1, seq_len]
    position_ids = packed_sequence["position_ids"].unsqueeze(0)  # [1, seq_len]

    # CP 模式下的处理
    if self.cp_size > 1:
        # 1. Padding: 确保 seq_len 能被 cp_size 整除
        packed_sequence = pad_packed_sequence_with_cp(packed_sequence, self.cp_size)

        # 2. 更新 Ring Flash Attention 参数
        if not packed_sequence["cu_seqlens"].is_cuda:
            packed_sequence["cu_seqlens"] = packed_sequence["cu_seqlens"].cuda()
        cu_seqlens = packed_sequence["cu_seqlens"]
        update_ring_flash_attn_params(cu_seqlens, self.cp_group)

        # 3. ⚠️ 关键：在序列维度（dim=1）切分 input_ids
        input_ids = torch.chunk(
            packed_sequence["tokens"].unsqueeze(0),
            self.cp_size,
            dim=1
        )[self.cp_rank]

        # 4. 同样切分 position_ids
        position_ids = torch.chunk(
            packed_sequence["position_ids"].unsqueeze(0),
            self.cp_size,
            dim=1
        )[self.cp_rank]

    model_args = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
    }
    return model_args
```

**切分逻辑**:

```python
# torch.chunk 的行为
tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # shape: [8]
chunks = torch.chunk(tokens.unsqueeze(0), chunks=2, dim=1)
# chunks[0]: [1, 2, 3, 4]  # cp_rank=0 获得前半部分
# chunks[1]: [5, 6, 7, 8]  # cp_rank=1 获得后半部分
```

### 1.2 切分示例

**配置**: cp_size=2, seq_len=1024

| GPU | dp_rank | cp_rank | input_ids 范围 | 说明 |
|-----|---------|---------|---------------|------|
| 0   | 0       | 0       | tokens[0:512] | 序列前半部分 |
| 1   | 0       | 1       | tokens[512:1024] | 序列后半部分 |
| 2   | 1       | 0       | tokens[0:512] | 序列前半部分（复制） |
| 3   | 1       | 1       | tokens[512:1024] | 序列后半部分（复制） |

**关键点**:

1. 切分在 **序列维度** (dim=1)，不是 batch 维度
2. 每个 cp_rank 看到序列的不同部分
3. 相同 cp_rank 的不同 dp_rank 看到相同的序列部分（数据并行的复制）

---

## 2. 2D DeviceMesh 的结构

### 2.1 DeviceMesh 创建

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 165-209)

```python
def _setup_device_mesh(self, args):
    """Setup device mesh for parallelism (always called, handles both CP and non-CP cases).

    Creates 2D mesh (dp_size, cp_size) for all cases:
    - When context_parallel_size > 1: hybrid CP + DP
    - When context_parallel_size = 1: pure DP (equivalent to 1D mesh)

    This ensures consistent group management across all parallelism modes.
    """
    from torch.distributed.device_mesh import init_device_mesh

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Use context_parallel_size directly (defaults to 1 for pure DP)
    self.cp_size = self.args.context_parallel_size
    self.dp_size = world_size // self.cp_size

    # Create 2D device mesh: (dp_size, cp_size)
    # Ranks laid out in row-major: mesh[dp_idx, cp_idx] = dp_idx * cp_size + cp_idx
    # - CP groups: consecutive ranks along dim 1, e.g., [0,1], [2,3], [4,5], [6,7]
    # - DP groups: striped ranks along dim 0, e.g., [0,2,4,6], [1,3,5,7]
    # When cp_size=1, this degenerates to pure DP
    self.mesh = init_device_mesh(
        "cuda",
        mesh_shape=(self.dp_size, self.cp_size),
        mesh_dim_names=("dp", "cp")
    )

    # Extract process groups from mesh
    self.dp_group = self.mesh.get_group("dp")  # For FSDP gradient sync, metric reduction
    self.cp_group = self.mesh.get_group("cp")  # For Ring Flash Attention, logit gathering
    self.dp_mesh = self.mesh["dp"]  # For FSDP

    # Compute local ranks within each dimension
    self.dp_rank = rank // self.cp_size
    self.cp_rank = rank % self.cp_size
```

### 2.2 Mesh 布局示例

**配置**: 8 GPUs, context_parallel_size=2

```
Total GPUs: 8
cp_size = 2
dp_size = 8 // 2 = 4

2D Mesh 布局 (row-major):
┌─────────────────────────────────────────┐
│ mesh[dp=0, cp=0] = GPU 0                │
│ mesh[dp=0, cp=1] = GPU 1                │  ← dp_rank=0 组
├─────────────────────────────────────────┤
│ mesh[dp=1, cp=0] = GPU 2                │
│ mesh[dp=1, cp=1] = GPU 3                │  ← dp_rank=1 组
├─────────────────────────────────────────┤
│ mesh[dp=2, cp=0] = GPU 4                │
│ mesh[dp=2, cp=1] = GPU 5                │  ← dp_rank=2 组
├─────────────────────────────────────────┤
│ mesh[dp=3, cp=0] = GPU 6                │
│ mesh[dp=3, cp=1] = GPU 7                │  ← dp_rank=3 组
└─────────────────────────────────────────┘

Process Groups:
  dp_group (FSDP sharding):
    - cp_rank=0 的 dp_group: [GPU 0, 2, 4, 6]
    - cp_rank=1 的 dp_group: [GPU 1, 3, 5, 7]

  cp_group (Ring Flash Attention):
    - dp_rank=0 的 cp_group: [GPU 0, 1]
    - dp_rank=1 的 cp_group: [GPU 2, 3]
    - dp_rank=2 的 cp_group: [GPU 4, 5]
    - dp_rank=3 的 cp_group: [GPU 6, 7]
```

### 2.3 关键：dp_mesh 的提取

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (line 192)

```python
self.dp_mesh = self.mesh["dp"]  # For FSDP
```

**dp_mesh 的含义**:

- 从 2D mesh 中提取 dp 维度，形成 1D mesh
- 不同 cp_rank 有独立的 dp_mesh

**示例**:

```python
# 对于 GPU 0 (dp_rank=0, cp_rank=0):
self.dp_mesh = [GPU 0, 2, 4, 6]  # 只包含 cp_rank=0 的 GPUs

# 对于 GPU 1 (dp_rank=0, cp_rank=1):
self.dp_mesh = [GPU 1, 3, 5, 7]  # 只包含 cp_rank=1 的 GPUs
```

---

## 3. Embedding 层的 FSDP2 包装

### 3.1 apply_fsdp2() 函数

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 1016-1057)

```python
def apply_fsdp2(model, mesh=None, cpu_offload=False):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    # 收集需要包装的模块
    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        # ⚠️ 关键：Embedding 层也被单独包装（除非 tie_word_embeddings）
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,  # ← 传入的是 dp_mesh (1D)
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model
```

**关键点**:

1. **Embedding 层被单独包装**: `isinstance(module, torch.nn.Embedding)`
2. **使用 dp_mesh**: FSDP2 只在 dp 维度分片
3. **tie_word_embeddings 的特殊处理**: 如果启用，Embedding 层不单独包装

### 3.2 FSDP2 的 Sharding 行为

**在 1D mesh 上的 sharding**:

```python
# 调用方式
model = apply_fsdp2(model, mesh=self.dp_mesh, cpu_offload=False)

# self.dp_mesh 是 1D mesh，例如 [GPU 0, 2, 4, 6]
# FSDP2 会在这个 mesh 上的所有 ranks 之间分片参数
```

**Embedding 层的分片**:

```python
# 假设：
# - vocab_size = 50000
# - embedding_dim = 4096
# - dp_size = 4
# - Embedding table shape: [50000, 4096]

# FSDP2 沿第 0 维（vocab 维度）分片：
# dp_rank=0: embedding_table[0:12500, :]          # 12500 个 token
# dp_rank=1: embedding_table[12500:25000, :]      # 12500 个 token
# dp_rank=2: embedding_table[25000:37500, :]      # 12500 个 token
# dp_rank=3: embedding_table[37500:50000, :]      # 12500 个 token
```

**CP 维度的复制**:

由于每个 cp_rank 有独立的 dp_mesh：

```python
# cp_rank=0 的 dp_mesh: [GPU 0, 2, 4, 6]
# GPU 0: embedding[0:12500, :]
# GPU 2: embedding[12500:25000, :]
# GPU 4: embedding[25000:37500, :]
# GPU 6: embedding[37500:50000, :]

# cp_rank=1 的 dp_mesh: [GPU 1, 3, 5, 7]
# GPU 1: embedding[0:12500, :]        ← 与 GPU 0 相同
# GPU 3: embedding[12500:25000, :]    ← 与 GPU 2 相同
# GPU 5: embedding[25000:37500, :]    ← 与 GPU 4 相同
# GPU 7: embedding[37500:50000, :]    ← 与 GPU 6 相同
```

**结论**: **Embedding Table 在 dp 维度分片，在 cp 维度复制**

---

## 4. FSDP2 的 All-Gather 机制

### 4.1 为什么需要 All-Gather？

**问题**:

```python
# GPU 0 (dp_rank=0, cp_rank=0):
# - 存储: embedding[0:12500, :]
# - 接收: input_ids = [128, 256, 30000, 45000, ...]
#         ↑ 包含 token 30000, 45000（不在本地分片）

# 如何查找 token 30000 的 embedding？
# GPU 0 只有 token [0, 12500) 的 embedding！
```

**解决**: FSDP2 的自动 All-Gather

### 4.2 All-Gather 的时机

**FSDP2 的前向传播生命周期**:

```python
# 伪代码：FSDP2 内部逻辑

class FSDPModule:
    def forward(self, input):
        # 1. Pre-forward hook: All-Gather 参数
        self._all_gather_parameters()
        # 现在 self.embedding_table 是完整的 [50000, 4096]

        # 2. 执行原始 forward
        output = self.original_forward(input)
        # output = self.embedding_table[input_ids]

        # 3. Post-forward hook: 释放其他分片
        self._free_other_shards()
        # 现在 self.embedding_table 又变回 [12500, 4096]

        return output
```

**All-Gather 的通信组**: `dp_group`

```python
# GPU 0 的 dp_group: [GPU 0, 2, 4, 6]
# All-Gather 过程：
# 1. GPU 0 持有: embedding[0:12500, :]
# 2. 从 GPU 2 接收: embedding[12500:25000, :]
# 3. 从 GPU 4 接收: embedding[25000:37500, :]
# 4. 从 GPU 6 接收: embedding[37500:50000, :]
# 5. 拼接成完整的 embedding[0:50000, :]
```

### 4.3 显存占用峰值

**时间线**:

```
T0: 初始状态（只存分片）
    GPU 0: embedding[0:12500, :] = 100 MB

T1: Pre-forward hook (All-Gather)
    GPU 0: embedding[0:50000, :] = 400 MB  ← 峰值

T2: 执行 Embedding lookup
    GPU 0: embedding[0:50000, :] = 400 MB
    output = embedding[input_ids]

T3: Post-forward hook (释放其他分片)
    GPU 0: embedding[0:12500, :] = 100 MB  ← 回到初始
```

**显存占用**:

| 阶段 | 存储的参数 | 显存占用 |
|-----|-----------|---------|
| 初始 | 自己的分片 | embedding_size / dp_size |
| All-Gather 后 | 完整参数 | embedding_size |
| 释放后 | 自己的分片 | embedding_size / dp_size |

**关键**: 每个 layer 都会经历 all-gather → compute → free 的循环，不会所有层同时 all-gather。

---

## 5. CP 模式下的完整数据流

### 5.1 端到端示例

**配置**:
- 8 GPUs, dp_size=4, cp_size=2
- vocab_size=50000, seq_len=1024
- input_ids: [batch=1, seq_len=1024]

**Step 1: input_ids 切分**

```python
# _get_model_inputs_args() 执行切分
GPU 0 (dp=0, cp=0): input_ids[:, 0:512]    # 序列前半
GPU 1 (dp=0, cp=1): input_ids[:, 512:1024] # 序列后半
GPU 2 (dp=1, cp=0): input_ids[:, 0:512]    # 序列前半（复制）
GPU 3 (dp=1, cp=1): input_ids[:, 512:1024] # 序列后半（复制）
GPU 4 (dp=2, cp=0): input_ids[:, 0:512]    # 序列前半（复制）
GPU 5 (dp=2, cp=1): input_ids[:, 512:1024] # 序列后半（复制）
GPU 6 (dp=3, cp=0): input_ids[:, 0:512]    # 序列前半（复制）
GPU 7 (dp=3, cp=1): input_ids[:, 512:1024] # 序列后半（复制）
```

**Step 2: Embedding 层 All-Gather**

```python
# GPU 0 的 dp_group: [0, 2, 4, 6]
GPU 0 存储: embedding[0:12500, :]
GPU 0 All-Gather:
  - 从 GPU 2 接收: embedding[12500:25000, :]
  - 从 GPU 4 接收: embedding[25000:37500, :]
  - 从 GPU 6 接收: embedding[37500:50000, :]
  - 结果: 完整 embedding[0:50000, :]

# GPU 1 的 dp_group: [1, 3, 5, 7]
GPU 1 存储: embedding[0:12500, :]  ← 与 GPU 0 相同
GPU 1 All-Gather:
  - 从 GPU 3 接收: embedding[12500:25000, :]
  - 从 GPU 5 接收: embedding[25000:37500, :]
  - 从 GPU 7 接收: embedding[37500:50000, :]
  - 结果: 完整 embedding[0:50000, :]

# 其他 GPUs 同理
```

**Step 3: Embedding Lookup**

```python
# 每个 GPU 独立执行
GPU 0: embeddings = embedding_table[input_ids[:, 0:512]]
       # shape: [1, 512, 4096]

GPU 1: embeddings = embedding_table[input_ids[:, 512:1024]]
       # shape: [1, 512, 4096]

# 其他 GPUs 同理
```

**Step 4: 释放其他分片**

```python
# 每个 GPU 保留自己的分片，释放其他分片
GPU 0: 保留 embedding[0:12500, :]
GPU 1: 保留 embedding[0:12500, :]
GPU 2: 保留 embedding[12500:25000, :]
# ...
```

### 5.2 通信流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 输入: input_ids [batch, 1024]                                    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ CP 切分 (在序列维度)                    │
        │ chunk(input_ids, cp_size, dim=1)      │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌───────────────────┐               ┌───────────────────┐
│ cp_rank=0 的 GPUs │               │ cp_rank=1 的 GPUs │
│ input_ids[:,:512] │               │ input_ids[:,512:] │
└───────────────────┘               └───────────────────┘
        │                                       │
        ↓                                       ↓
┌───────────────────┐               ┌───────────────────┐
│ Embedding Layer   │               │ Embedding Layer   │
│ (FSDP2 wrapped)   │               │ (FSDP2 wrapped)   │
└───────────────────┘               └───────────────────┘
        │                                       │
        ↓                                       ↓
  ┌─────────────┐                       ┌─────────────┐
  │ All-Gather  │                       │ All-Gather  │
  │ dp_group    │                       │ dp_group    │
  │ [0,2,4,6]   │                       │ [1,3,5,7]   │
  └─────────────┘                       └─────────────┘
        │                                       │
        ↓                                       ↓
┌───────────────────┐               ┌───────────────────┐
│ 完整 Embedding    │               │ 完整 Embedding    │
│ [50000, 4096]     │               │ [50000, 4096]     │
└───────────────────┘               └───────────────────┘
        │                                       │
        ↓                                       ↓
┌───────────────────┐               ┌───────────────────┐
│ Lookup            │               │ Lookup            │
│ embedding[input]  │               │ embedding[input]  │
└───────────────────┘               └───────────────────┘
        │                                       │
        ↓                                       ↓
┌───────────────────┐               ┌───────────────────┐
│ Output[:,:512,:]  │               │ Output[:,512:,:]  │
└───────────────────┘               └───────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            ↓
                ┌───────────────────────┐
                │ Ring Flash Attention  │
                │ (在 cp_group 内通信)   │
                └───────────────────────┘
```

---

## 6. 通信开销分析

### 6.1 Embedding All-Gather 的通信量

**配置**: vocab_size=50000, hidden_dim=4096, dp_size=4

**Embedding Table 大小**:
```
total_size = vocab_size × hidden_dim × dtype_size
          = 50000 × 4096 × 2 bytes (bf16)
          = 409,600,000 bytes
          ≈ 400 MB
```

**每个 GPU 存储**:
```
shard_size = total_size / dp_size
          = 400 MB / 4
          = 100 MB
```

**All-Gather 通信量** (每个 GPU):

```python
# 接收其他 (dp_size - 1) 个分片
receive_size = shard_size × (dp_size - 1)
            = 100 MB × 3
            = 300 MB

# 发送自己的分片给其他 (dp_size - 1) 个 GPU
send_size = shard_size × (dp_size - 1)
         = 100 MB × 3
         = 300 MB

# All-Gather 是 all-to-all，总通信量（双向）
total_per_gpu = receive_size + send_size
             = 300 MB + 300 MB
             = 600 MB
```

**但实际上**: PyTorch 的 all-gather 实现通常优化为环形 reduce-scatter + all-gather，实际通信量更接近 300 MB/GPU。

### 6.2 CP 模式的通信开销

**CP 维度对 Embedding All-Gather 的影响**:

```python
# cp_size=2, dp_size=4
# 两个独立的 dp_group 并行执行 All-Gather

dp_group[0]: [GPU 0, 2, 4, 6] → All-Gather 300 MB/GPU
dp_group[1]: [GPU 1, 3, 5, 7] → All-Gather 300 MB/GPU

# CP 维度没有 Embedding 参数通信（参数已复制）
# 总通信量: 300 MB/GPU（与不使用 CP 时相同）
```

**结论**: CP 模式不增加 Embedding All-Gather 的通信量。

### 6.3 与不分片的对比

| 方案 | 每个 GPU 存储 | All-Gather 通信量 | 显存占用峰值 |
|-----|-------------|------------------|------------|
| 不分片 | 400 MB | 0 MB | 400 MB |
| FSDP2 分片 (dp=4) | 100 MB | 300 MB | 400 MB (临时) |
| FSDP2 分片 (dp=8) | 50 MB | 437.5 MB | 400 MB (临时) |

**权衡**:
- 不分片: 无通信开销，但显存占用高
- FSDP2 分片: 有通信开销，但显存占用低（静态）

对于大词表模型（vocab_size > 100K），FSDP2 分片能显著减少显存占用。

---

## 7. tie_word_embeddings 的特殊处理

### 7.1 什么是 tie_word_embeddings？

**定义**: 输入 Embedding 层和输出 LM Head 共享权重

```python
# 不 tie 的情况:
model.embed_tokens = Embedding(50000, 4096)  # 输入 Embedding
model.lm_head = Linear(4096, 50000)          # 输出 LM Head
# 总参数: 50000 * 4096 * 2 = 200M * 2 = 400M params

# tie 的情况:
model.embed_tokens = Embedding(50000, 4096)
model.lm_head.weight = model.embed_tokens.weight  # 共享权重
# 总参数: 50000 * 4096 = 200M params (省一半)
```

### 7.2 slime 中的处理

**文件**: `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` (lines 216-233, 1038)

```python
# 检查是否使用 tied embeddings
use_meta_tensor = not self.hf_config.tie_word_embeddings

if use_meta_tensor:
    # 使用 meta tensor 初始化（节省内存）
    init_context = torch.device("meta")
else:
    # tie_word_embeddings=True 时，所有 rank 加载完整模型到 CPU
    logger.info(
        f"[Rank {dist.get_rank()}] tie_word_embeddings=True, "
        "loading full model to CPU on all ranks"
    )
    init_context = torch.device("cpu")
```

**apply_fsdp2 中的特殊处理** (line 1038):

```python
modules = [
    module
    for name, module in model.named_modules()
    if module.__class__.__name__ in layer_cls_to_wrap
    # ⚠️ 只有在不 tie 时才单独包装 Embedding
    or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
]
```

**原因**:

1. **meta tensor 问题**: tie_word_embeddings 与 meta tensor 初始化不兼容（会导致 hang）
2. **FSDP2 包装**: 如果 tie 了权重，Embedding 和 LM Head 必须在同一个 FSDP unit，不能单独包装

### 7.3 tie_word_embeddings 下的 Sharding

```python
# tie_word_embeddings=False:
# - Embedding 单独包装为 FSDP unit
# - LM Head 单独包装为 FSDP unit
# - 分别分片

# tie_word_embeddings=True:
# - Embedding 不单独包装
# - 和 LM Head 一起在顶层包装
# - 共享权重仍然分片，但作为一个整体
```

**分片行为**:

```python
# tie=True, dp_size=4
# Embedding weight (共享): [50000, 4096]
# LM Head weight: 指向 Embedding weight

# FSDP2 分片:
# dp_rank=0: weight[0:12500, :]     (Embedding 和 LM Head 共享)
# dp_rank=1: weight[12500:25000, :]
# dp_rank=2: weight[25000:37500, :]
# dp_rank=3: weight[37500:50000, :]
```

---

## 8. 性能优化建议

### 8.1 Embedding All-Gather 的优化

**问题**: Embedding All-Gather 在每次前向传播时都会执行

**优化方向**:

1. **减少 dp_size**:
   ```bash
   # dp_size=8: 每次 all-gather 437.5 MB
   # dp_size=4: 每次 all-gather 300 MB
   # dp_size=2: 每次 all-gather 200 MB
   ```

2. **使用更大的 batch size**:
   - 摊薄 all-gather 的固定开销
   - All-gather 次数 = batch 数

3. **考虑不分片 Embedding**:
   - 如果显存充足，可以不对 Embedding 使用 FSDP
   - 需要自定义 FSDP 包装策略

### 8.2 CP 和 DP 的平衡

**场景**: 8 GPUs，选择 cp_size 和 dp_size

| 配置 | cp_size | dp_size | Embedding 通信 | CP 通信（Attention） |
|-----|---------|---------|---------------|-------------------|
| 1   | 1       | 8       | 437.5 MB/GPU  | 0 MB |
| 2   | 2       | 4       | 300 MB/GPU    | 中等 |
| 3   | 4       | 2       | 200 MB/GPU    | 高 |
| 4   | 8       | 1       | 0 MB          | 极高 |

**权衡**:

- **cp_size 越大**: Embedding 通信越少，但 Ring Flash Attention 通信越多
- **dp_size 越大**: Embedding 通信越多，但梯度同步更高效

**推荐**:

```bash
# 短序列 (<2K): 优先 dp_size
--context-parallel-size 1  # cp_size=1, 纯 DP

# 中等序列 (2K-8K): 平衡
--context-parallel-size 2  # cp_size=2

# 长序列 (>8K): 优先 cp_size
--context-parallel-size 4  # cp_size=4
```

### 8.3 自定义 Embedding 分片策略

**场景**: 超大词表（vocab_size > 100K）

**策略 1**: Vocabulary Parallel（词表并行）

```python
# 在 vocab 维度分片，但不使用 FSDP
# 每个 GPU 只计算部分 vocab 的 logits
# 需要自定义 Embedding 和 LM Head 的实现
```

**策略 2**: 混合并行

```python
# Embedding: 不分片（每个 GPU 存完整）
# Transformer layers: FSDP2 分片
# LM Head: Vocabulary Parallel

# 需要在 apply_fsdp2 中排除 Embedding
```

---

## 9. 总结

### 9.1 核心问题回答

**Q1: input_ids 被切分成 cp_size 份，Embedding 层是否也被切分计算？**

**答**: **Embedding 层不是"切分计算"，而是通过 FSDP2 的 All-Gather 机制实现**

- Embedding Table 在 **dp 维度分片存储**
- 每个 GPU 只存部分 vocab 的 embedding
- 前向传播时，FSDP2 自动执行 **All-Gather** 临时获得完整 Embedding Table
- 每个 GPU 可以查找任意 token ID 的 embedding
- 计算完成后，释放其他分片，保留自己的分片

**Q2: 每张卡是否存了完整的 Embedding Table？**

**答**: **静态存储：否；动态计算：是（临时）**

- **静态存储**（非前向传播时）: 每张卡只存 `vocab_size / dp_size` 的 embedding
- **动态计算**（前向传播中）: 每张卡临时拥有完整的 embedding（通过 all-gather）
- **CP 维度**: 同一 dp_rank 的不同 cp_rank 存储相同的分片（复制）

### 9.2 关键设计点

1. **2D DeviceMesh**:
   - 第一维 (dp): Data Parallel，用于 FSDP 分片
   - 第二维 (cp): Context Parallel，用于序列切分

2. **input_ids 切分**:
   - 在 CP 维度切分（序列维度）
   - 每个 cp_rank 处理序列的不同部分

3. **Embedding 分片**:
   - 在 DP 维度分片（vocab 维度）
   - 在 CP 维度复制

4. **All-Gather 机制**:
   - 在 dp_group 内进行
   - 临时聚合完整 Embedding Table
   - 通信量: `embedding_size * (dp_size - 1) / dp_size`

5. **tie_word_embeddings**:
   - 特殊处理，不单独包装 Embedding
   - 避免 meta tensor 初始化问题

### 9.3 实现要点

如果要在其他框架中复现 slime 的 Embedding 分片策略：

1. **创建 2D DeviceMesh**:
   ```python
   mesh = init_device_mesh("cuda", (dp_size, cp_size), ("dp", "cp"))
   dp_mesh = mesh["dp"]  # 用于 FSDP
   ```

2. **应用 FSDP2**:
   ```python
   # 单独包装 Embedding（如果不 tie）
   fully_shard(model.embed_tokens, mesh=dp_mesh)
   ```

3. **切分 input_ids**:
   ```python
   # 在序列维度切分
   input_ids = torch.chunk(input_ids, cp_size, dim=1)[cp_rank]
   ```

4. **信任 FSDP2**:
   - FSDP2 会自动处理 all-gather
   - 不需要手动实现参数通信
   - 前向传播时自动聚合，计算后自动释放

### 9.4 性能权衡

| 方面 | 不分片 | FSDP2 分片 |
|-----|--------|-----------|
| 显存占用 | 高（每卡存完整） | 低（每卡存 1/dp_size） |
| 通信开销 | 无 | 有（每次前向 all-gather） |
| 适用场景 | 小词表 (<50K) | 大词表 (>100K) |
| 扩展性 | 差（显存限制） | 好（可扩展到更大模型） |

**最终建议**: 对于大词表模型（如 vocab_size > 100K），使用 FSDP2 分片是必要的；对于小词表模型，可以考虑不分片以减少通信开销。

---

## 10. 相关源码索引

| 功能 | 文件路径 | 行号 |
|-----|---------|------|
| input_ids 切分 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 823-824 |
| _get_model_inputs_args() | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 811-831 |
| 2D DeviceMesh 创建 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 165-209 |
| dp_mesh 提取 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 192 |
| apply_fsdp2() | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 1016-1057 |
| Embedding 单独包装逻辑 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 1038 |
| tie_word_embeddings 处理 | `/home/scbjtfy/slime/slime/backends/fsdp_utils/actor.py` | 216-233 |
| pad_packed_sequence_with_cp | `/home/scbjtfy/slime/slime/backends/fsdp_utils/data_packing.py` | 165-185 |

---

**生成时间**: 2025-12-04
**分析框架版本**: slime (commit: 9d7f34d)
**分析者**: Claude Code (Sonnet 4.5)
