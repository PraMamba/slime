# FSDP2 DeviceMesh 与参数切分机制深度解析

> **面向对象**: Infra 工程师，目标是在其他框架中复现/支持 FSDP2 后端
> **作者**: 基于 slime 源码分析
> **日期**: 2025-12-03
> **版本**: v1.0

本文档深入分析 FSDP2 在 slime 中的 **DeviceMesh 构建**和**参数切分机制**，专注于数据流转、内存管理、并行通信的底层细节。

---

## 问题-1 回答：DeviceMesh 构建与参数切分机制

### 1.1 DeviceMesh 的维度：DP + CP 混合并行

#### 1.1.1 核心代码分析

在 slime 中，DeviceMesh 的构建位于 `actor.py:164-210`：

```python
def _setup_device_mesh(self) -> None:
    """Setup device mesh for parallelism (always called, handles both CP and non-CP cases).

    Creates 2D mesh (dp_size, cp_size) for all cases:
    - When context_parallel_size > 1: hybrid CP + DP
    - When context_parallel_size = 1: pure DP (equivalent to 1D mesh)
    """
    from torch.distributed.device_mesh import init_device_mesh

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Use context_parallel_size directly (defaults to 1 for pure DP)
    self.cp_size = self.args.context_parallel_size  # 从命令行参数获取
    self.dp_size = world_size // self.cp_size

    # 关键：创建 2D device mesh: (dp_size, cp_size)
    # Ranks laid out in row-major: mesh[dp_idx, cp_idx] = dp_idx * cp_size + cp_idx
    # - CP groups: consecutive ranks along dim 1, e.g., [0,1], [2,3], [4,5], [6,7]
    # - DP groups: striped ranks along dim 0, e.g., [0,2,4,6], [1,3,5,7]
    # When cp_size=1, this degenerates to pure DP
    self.mesh = init_device_mesh(
        "cuda",
        mesh_shape=(self.dp_size, self.cp_size),  # 关键：2D shape
        mesh_dim_names=("dp", "cp")                # 关键：命名维度
    )

    # Extract process groups from mesh
    self.dp_group = self.mesh.get_group("dp")  # For FSDP gradient sync, metric reduction
    self.cp_group = self.mesh.get_group("cp")  # For Ring Flash Attention, logit gathering
    self.dp_mesh = self.mesh["dp"]             # For FSDP wrapping（关键：只传入 DP 维度）

    # Compute local ranks within each dimension
    self.dp_rank = rank // self.cp_size
    self.cp_rank = rank % self.cp_size

    logger.info(
        f"[Rank {rank}] Device mesh (2D): world_size={world_size}, "
        f"cp_size={self.cp_size}, dp_size={self.dp_size}"
    )
    logger.info(
        f"[Rank {rank}] Mesh shape: {self.mesh.shape}, "
        f"dp_rank={self.dp_rank}, cp_rank={self.cp_rank}"
    )

    # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
    if self.cp_size > 1:
        substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
        logger.info(f"[Rank {rank}] CP initialized via device mesh")
    else:
        logger.info(f"[Rank {rank}] Pure DP mode (cp_size=1)")
```

#### 1.1.2 DeviceMesh 维度图解

**示例：8 GPUs, cp_size=2, dp_size=4**

```
全局 Mesh (2D): shape = (dp_size=4, cp_size=2)

     CP Dim (dim=1) →
DP    [  0    1  ]    CP Group 0: [0, 1]
Dim   [  2    3  ]    CP Group 1: [2, 3]
(dim  [  4    5  ]    CP Group 2: [4, 5]
=0)   [  6    7  ]    CP Group 3: [6, 7]
↓
      DP Group 0: [0, 2, 4, 6]
      DP Group 1: [1, 3, 5, 7]

Rank 计算公式: rank = dp_idx * cp_size + cp_idx

例如：
- Rank 0: dp_idx=0, cp_idx=0
- Rank 5: dp_idx=2, cp_idx=1
- Rank 7: dp_idx=3, cp_idx=1
```

**关键要点**：

1. **Row-major 布局**：rank = dp_idx * cp_size + cp_idx
2. **CP 组是连续的**：[0,1], [2,3], [4,5], [6,7]，便于序列切分
3. **DP 组是跨步的**：[0,2,4,6], [1,3,5,7]，便于梯度同步
4. **维度命名**：`("dp", "cp")` 使得代码更清晰

#### 1.1.3 为什么是 2D Mesh？

**设计考量**：

1. **统一接口**：即使 `cp_size=1`（纯 DP 模式），仍然使用 2D mesh，避免条件分支
2. **清晰的通信组**：
   - `mesh.get_group("dp")` → FSDP 梯度同步
   - `mesh.get_group("cp")` → Ring Flash Attention
3. **易于扩展**：未来如果加入 TP，可以扩展为 3D mesh: `(dp_size, tp_size, cp_size)`

**对比 1D Mesh（纯 DP）**：

```python
# 如果只有 DP，可以用 1D mesh
mesh = init_device_mesh("cuda", mesh_shape=(dp_size,), mesh_dim_names=("dp",))

# 但 slime 选择统一使用 2D mesh：
mesh = init_device_mesh("cuda", mesh_shape=(dp_size, 1), mesh_dim_names=("dp", "cp"))
```

**优势**：代码逻辑统一，无需判断 `if cp_size > 1`。

---

### 1.2 FSDP2 如何利用 Mesh 切分参数？

#### 1.2.1 核心代码：`apply_fsdp2` 函数

位于 `actor.py:1016-1058`：

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

    # ====== 关键 1：获取需要切分的层类型 ======
    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    # ====== 关键 2：自动遍历 model 找到所有匹配的 module ======
    modules = [
        module
        for name, module in model.named_modules()  # 自动遍历所有子模块
        if module.__class__.__name__ in layer_cls_to_wrap  # 匹配层类型
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    # ====== 关键 3：构造 FSDP 配置 ======
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # 参数存储为 bf16
            reduce_dtype=torch.float32,   # 梯度聚合为 fp32
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,  # 传入 dp_mesh（只包含 DP 维度）
    }

    # ====== 关键 4：对每个 module 应用 fully_shard ======
    for module in modules:
        fully_shard(module, **fsdp_kwargs)  # 这里会自动切分该 module 的参数

    # ====== 关键 5：对顶层 model 应用 fully_shard ======
    fully_shard(model, **fsdp_kwargs)  # 包装 embedding、lm_head 等

    return model
```

#### 1.2.2 `model._no_split_modules` 是什么？

**定义**：这是 HuggingFace Transformers 库中的一个约定，用于指定哪些模块不应该被拆分（即应该作为一个整体进行并行化）。

**实际例子**：

```python
# 对于 Qwen2/Qwen3 模型：
model._no_split_modules = ['Qwen2DecoderLayer']

# 对于 LLaMA 模型：
model._no_split_modules = ['LlamaDecoderLayer']

# 对于 GPT-2 模型：
model._no_split_modules = ['GPT2Block']
```

**含义**：
- `Qwen2DecoderLayer` 是 Transformer 的一个 Decoder 层（包含 attention + MLP）
- FSDP 会将每个 `Qwen2DecoderLayer` 作为一个单元进行切分，而不是拆开 attention 和 MLP

**为什么这样设计？**
1. **通信效率**：保持层级结构完整，减少细粒度通信
2. **内存管理**：FSDP2 以层为单位进行 all-gather 和释放，粒度适中
3. **数值稳定性**：层级 normalization (LayerNorm) 在同一设备上计算

#### 1.2.3 自动遍历 vs 手动指定

**slime 的实现：自动遍历**

```python
modules = [
    module
    for name, module in model.named_modules()  # 自动遍历
    if module.__class__.__name__ in layer_cls_to_wrap
    or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
]
```

**关键点**：
1. **自动发现**：`model.named_modules()` 递归遍历整个模型树
2. **类型匹配**：检查 `module.__class__.__name__` 是否在白名单中
3. **特殊处理**：非 tied 的 Embedding 也单独包装

**模型结构示例**（Qwen3-4B）：

```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 3584)  ← 单独包装（如果不是 tied）
    (layers): ModuleList(
      (0): Qwen2DecoderLayer(...)  ← 包装
      (1): Qwen2DecoderLayer(...)  ← 包装
      ...
      (39): Qwen2DecoderLayer(...) ← 包装
    )
    (norm): Qwen2RMSNorm()
  )
  (lm_head): Linear(3584, 151936)  ← 顶层 fully_shard 包装
)
```

**fully_shard 的调用顺序**：

```python
# 步骤 1: 包装每个 DecoderLayer (40 次)
for layer in model.model.layers:
    fully_shard(layer, **fsdp_kwargs)  # 参数在 DP 维度上切分

# 步骤 2: 包装 Embedding（如果需要）
if not model.config.tie_word_embeddings:
    fully_shard(model.model.embed_tokens, **fsdp_kwargs)

# 步骤 3: 包装顶层 model（包含 lm_head）
fully_shard(model, **fsdp_kwargs)
```

**你是否需要手动指定？**

**答案：不需要！FSDP2 + HuggingFace 的生态已经自动化了这个过程。**

**原因**：
1. **HuggingFace 约定**：所有模型都定义了 `_no_split_modules`
2. **自动遍历**：`model.named_modules()` 自动找到所有匹配的层
3. **统一包装**：`fully_shard` API 简洁，无需手动管理

**对比手动指定（如果你在其他框架复现）**：

```python
# 如果没有 _no_split_modules，你需要手动指定：
modules_to_wrap = [
    model.model.layers[0],
    model.model.layers[1],
    ...
    model.model.layers[39],
    model.model.embed_tokens,
]

for module in modules_to_wrap:
    fully_shard(module, mesh=dp_mesh, ...)
```

**但在 HuggingFace 生态中，这完全不需要。**

---

### 1.3 `fully_shard` 如何切分参数？

#### 1.3.1 `fully_shard` 的工作机制

**PyTorch FSDP2 的核心 API**：

```python
from torch.distributed.fsdp import fully_shard

fully_shard(
    module,                  # 要包装的模块
    mesh=dp_mesh,            # DeviceMesh（只包含 DP 维度）
    mp_policy=...,           # 混合精度策略
    offload_policy=...,      # CPU offload 策略
)
```

**工作流程**：

1. **参数注册**：`fully_shard` 会遍历 `module.parameters()`，找到所有参数
2. **DTensor 转换**：将每个 `torch.Tensor` 参数转换为 `DTensor`（分布式张量）
3. **Shard 放置**：根据 `mesh` 的维度，将参数在 DP 维度上切分（Shard）

**DTensor 的 Placement**（放置策略）：

PyTorch 的 DTensor 支持三种放置策略：

```python
from torch.distributed.tensor import Shard, Replicate, Partial

# Shard(dim): 在指定维度上切分
# Replicate(): 在所有 rank 上复制完整数据
# Partial(): 部分数据（用于梯度累积）
```

**FSDP2 的默认策略**：

```python
# 参数：Shard(0) - 在第 0 维（通常是 out_features）上切分
# 梯度：Partial() → Shard(0) - 先部分累积，再切分

# 例如：Linear(3584, 151936)
# 参数 shape: [151936, 3584]
# 在 DP=4 的情况下，每个 rank 只存储:
#   Rank 0: [0:37984, :]        (1/4 的 out_features)
#   Rank 1: [37984:75968, :]    (1/4 的 out_features)
#   Rank 2: [75968:113952, :]   (1/4 的 out_features)
#   Rank 3: [113952:151936, :]  (1/4 的 out_features)
```

#### 1.3.2 DTensor 的实际例子

**创建一个 DTensor**：

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate

# 初始化分布式环境
dist.init_process_group(backend="nccl")

# 创建 1D mesh (纯 DP)
mesh = init_device_mesh("cuda", mesh_shape=(4,), mesh_dim_names=("dp",))

# 创建一个全局 tensor (只在 rank 0 有数据)
if dist.get_rank() == 0:
    global_tensor = torch.randn(151936, 3584, device="cuda")
else:
    global_tensor = torch.empty(151936, 3584, device="cuda")

# 转换为 DTensor，在 dim=0 上切分
dtensor = DTensor.from_local(
    global_tensor[dist.get_rank() * 37984 : (dist.get_rank() + 1) * 37984],  # 本地分片
    device_mesh=mesh,
    placements=[Shard(0)],  # 在 dim=0 上切分
)

# 每个 rank 只存储 1/4 的数据
print(f"Rank {dist.get_rank()}: local shape = {dtensor.to_local().shape}")
# Rank 0: local shape = torch.Size([37984, 3584])
# Rank 1: local shape = torch.Size([37984, 3584])
# Rank 2: local shape = torch.Size([37984, 3584])
# Rank 3: local shape = torch.Size([37984, 3584])
```

**DTensor 的关键操作**：

```python
# 1. 获取本地分片
local_tensor = dtensor.to_local()  # 返回当前 rank 存储的部分

# 2. 获取完整 tensor（需要 all-gather）
full_tensor = dtensor.full_tensor()  # 通信：all-gather 所有分片

# 3. 改变放置策略（redistribute）
replicated_dtensor = dtensor.redistribute(
    placements=[Replicate()],  # 从 Shard(0) 变为 Replicate()
    async_op=True,             # 异步操作
)

# 4. 等待异步操作完成
replicated_dtensor.wait()
```

#### 1.3.3 FSDP2 的前向/反向传播流程

**前向传播**：

```python
# 假设 model 是 FSDP2 包装的模型
output = model(input_ids)

# 内部流程（以一个 Linear 层为例）：
# 1. Pre-forward hook: all-gather 参数（从 Shard(0) → Replicate()）
#    - 每个 rank 从其他 rank 收集参数分片
#    - 重建完整的 weight: [151936, 3584]
#
# 2. Forward: 计算 output = input @ weight.T
#    - 使用完整的 weight 进行计算
#
# 3. Post-forward hook: 释放完整参数（从 Replicate() → Shard(0)）
#    - 只保留本地分片，释放其他部分的显存
```

**反向传播**：

```python
loss.backward()

# 内部流程：
# 1. Pre-backward hook: all-gather 参数（再次收集完整参数）
#    - 因为计算梯度需要完整的 weight
#
# 2. Backward: 计算梯度 grad_weight = grad_output.T @ input
#    - 每个 rank 都计算完整的 grad_weight
#
# 3. Post-backward hook: reduce-scatter 梯度（从 Replicate() → Shard(0)）
#    - 将完整的 grad_weight 按照参数切分方式进行 reduce-scatter
#    - Rank 0 得到 grad_weight[0:37984, :]
#    - Rank 1 得到 grad_weight[37984:75968, :]
#    - ...
#
# 4. 释放完整参数（从 Replicate() → Shard(0)）
```

**通信模式图解**：

```
前向传播：
  All-Gather (collect full param) → Compute → Free full param

反向传播：
  All-Gather (collect full param) → Compute grad → Reduce-Scatter (shard grad) → Free full param

关键优化：
  - 参数只在计算时收集，用完立即释放
  - 梯度在计算后立即进行 reduce-scatter 并切分
  - 峰值显存 = 本地参数 + 一个层的完整参数
```

---

### 1.4 在 slime 中的实际调用

#### 1.4.1 初始化阶段

**代码位置**：`actor.py:82-101`

```python
# 在 init 方法中：

# 步骤 1: 设置 Device Mesh（2D: DP + CP）
self._setup_device_mesh()  # 创建 mesh, dp_mesh, cp_mesh

# 步骤 2: 加载模型（rank 0 从磁盘加载，其他 rank 使用 meta device）
init_context = self._get_init_weight_context_manager()
with init_context():
    model = AutoModelForCausalLM.from_pretrained(
        self.args.hf_checkpoint,
        trust_remote_code=True,
        attn_implementation=self.args.attn_implementation,
    )

model.train()
full_state = model.state_dict()  # Rank 0 有完整权重，其他 rank 为空

# 步骤 3: 应用 FSDP2 包装（传入 dp_mesh，不传 cp_mesh）
model = apply_fsdp2(
    model,
    mesh=self.dp_mesh,  # 关键：只传入 DP 维度的 mesh
    cpu_offload=self.fsdp_cpu_offload
)

# 步骤 4: 从 rank 0 广播权重到所有 rank
model = self._fsdp2_load_full_state_dict(
    model,
    full_state,
    self.dp_mesh,
    cpu_offload=True if self.fsdp_cpu_offload else None
)

self.model = model  # 现在 model 已经是 FSDP2 包装的，参数已切分
```

#### 1.4.2 为什么只传 `dp_mesh`，不传完整的 2D mesh？

**关键设计决策**：

```python
# slime 的实现：
self.mesh = init_device_mesh(..., mesh_shape=(dp_size, cp_size), ...)  # 2D mesh
self.dp_mesh = self.mesh["dp"]  # 提取 DP 维度（1D mesh）

# 传给 FSDP2：
model = apply_fsdp2(model, mesh=self.dp_mesh, ...)  # 只传 1D dp_mesh
```

**原因**：

1. **FSDP2 只负责 DP 并行**：
   - 参数切分只在 DP 维度进行
   - CP 并行由 Ring Flash Attention 处理，不涉及参数切分

2. **避免混淆**：
   - 如果传入 2D mesh，FSDP2 不知道应该在哪个维度切分
   - 1D mesh 明确了切分维度

3. **与 Megatron 的对比**：
   - Megatron：TP 和 DP 都由框架管理，需要 2D mesh
   - FSDP2：只管 DP，TP/CP 由其他机制管理

**提取 DP 维度的 mesh**：

```python
# PyTorch 的 DeviceMesh 支持索引操作
mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("dp", "cp"))

# 提取 DP 维度（返回 1D mesh）
dp_mesh = mesh["dp"]  # 包含 DP 组的通信信息

# 提取 CP 维度（返回 1D mesh）
cp_mesh = mesh["cp"]  # 包含 CP 组的通信信息
```

---

### 1.5 DTensor 在权重更新中的应用

#### 1.5.1 从 Training Backend 到 Inference Engine

**代码位置**：`update_weight_utils.py:45-74`

```python
def update_weights(self) -> None:
    bucket = []
    bucket_size = 0

    for name, param in self.model.state_dict().items():
        param_size = param.numel() * param.element_size()

        if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
            self.wait_and_update_bucket_weights(bucket)
            del bucket
            bucket = []
            bucket_size = 0

        param = param.cuda()

        # ====== 关键：处理 DTensor ======
        if isinstance(param, DTensor):
            # 异步 redistribute: Shard(0) → Replicate()
            param = param.redistribute(
                placements=[Replicate()] * param.device_mesh.ndim,  # 所有维度 Replicate
                async_op=True,  # 异步操作，提前启动通信
            ).to_local()  # 转换为本地 tensor（等待通信完成）

        bucket.append((name, param))
        bucket_size += param_size

    if bucket:
        self.wait_and_update_bucket_weights(bucket)
```

**关键点**：

1. **DTensor 检测**：`isinstance(param, DTensor)` 判断参数是否被 FSDP2 包装
2. **Redistribute**：从 `Shard(0)` 改为 `Replicate()`，触发 all-gather
3. **异步操作**：`async_op=True` 提前启动通信，重叠计算和通信
4. **to_local()**：转换为普通 tensor，隐式等待通信完成

**通信流程**：

```
Training Backend (FSDP2):
  param: DTensor, placements=[Shard(0)], local_shape=[37984, 3584]

  ↓ redistribute(placements=[Replicate()], async_op=True)

  all-gather 通信开始（在后台异步执行）

  ↓ to_local()

  等待 all-gather 完成
  param: Tensor, shape=[151936, 3584]

  ↓ 传输到 Inference Engine

Inference Engine (SGLang):
  收到完整的 param: Tensor, shape=[151936, 3584]
  更新推理引擎的权重
```

#### 1.5.2 分桶优化的必要性

**为什么需要分桶？**

```python
# 假设模型有 40 层，每层的 Linear 权重：
# - q_proj: [151936, 3584]
# - k_proj: [151936, 3584]
# - v_proj: [151936, 3584]
# - o_proj: [151936, 3584]
# - gate_proj: [151936, 18944]
# - up_proj: [151936, 18944]
# - down_proj: [18944, 3584]

# 如果一次性 all-gather 所有参数：
# - 峰值显存占用 = 完整模型参数 × 2（原始 + gathered）
# - 对于 Qwen3-4B：约 8GB × 2 = 16GB

# 使用分桶（每次 512MB）：
# - 峰值显存占用 = 本地参数 + 512MB
# - 更可控，避免 OOM
```

---

## 2. 数据流转全流程

### 2.1 从 Rollout 到 Training

```
1. Rollout 阶段（SGLang Inference Engine）:
   ┌─────────────────────────────┐
   │ Prompt → Model → Response  │
   │ 记录 tokens, log_probs     │
   └─────────────────────────────┘
                 ↓
   rollout_data = {
     "tokens": [...],           # shape: [num_samples, seq_len]
     "log_probs": [...],        # shape: [num_samples, seq_len]
     "rewards": [...],          # shape: [num_samples]
     ...
   }

2. 数据传输（Ray Object Store）:
   rollout_data → Ray.put() → ObjectRef

3. Training 端接收:
   process_rollout_data(rollout_data_ref, dp_rank, dp_size)
   ↓
   根据 dp_rank 拆分数据到各个 DP rank

4. Data Packing:
   pack_sequences() → packed_batches
   ↓
   将变长序列打包成定长 pack，消除 padding

5. Context Parallel 切分（如果 cp_size > 1）:
   pad_packed_sequence_with_cp() → 对齐到 cp_size 的倍数
   ↓
   torch.chunk(tokens, cp_size, dim=1)[cp_rank] → 每个 CP rank 只处理 1/cp_size 的序列

6. Forward Pass（FSDP2）:
   对于每个 packed_batch:
     For each layer:
       - All-Gather 参数（Shard → Replicate）
       - 计算 forward
       - Free 完整参数（Replicate → Shard）
   ↓
   logits: shape = [local_seq_len, vocab_size]（每个 CP rank）

7. Log Prob 计算（CP 模式）:
   get_logprob_and_entropy_with_cp()
   ↓
   每个 CP rank 计算本地的 log_probs
   ↓
   All-Gather 所有 CP rank 的结果
   ↓
   拼接成完整的 log_probs: shape = [total_seq_len]

8. Loss 计算:
   unpack_sequences() → 还原每个样本的 log_probs
   ↓
   计算 PPO/GRPO loss
   ↓
   loss.backward()

9. Backward Pass（FSDP2）:
   对于每个 layer（逆序）:
     - All-Gather 参数（Shard → Replicate）
     - 计算梯度 grad_weight
     - Reduce-Scatter 梯度（Replicate → Shard）
     - Free 完整参数

10. Optimizer Step:
    optimizer.step()
    ↓
    更新本地参数分片（每个 DP rank 更新自己的 Shard）

11. 权重同步（Colocated 模式）:
    update_weights()
    ↓
    For each param:
      - DTensor.redistribute(Replicate(), async_op=True)  # All-Gather
      - 传输到 SGLang Inference Engine
    ↓
    SGLang 使用新权重进行下一轮 Rollout
```

---

## 3. 内存管理策略

### 3.1 FSDP2 的内存占用

**公式**：

```
峰值显存 = 本地参数分片 + 一个层的完整参数 + 激活值 + 梯度分片 + 优化器状态分片

具体：
  本地参数分片 = 总参数 / dp_size
  一个层的完整参数 = 单层参数（临时，计算后释放）
  激活值 = batch_size × seq_len × hidden_size（可用 gradient checkpointing 减少）
  梯度分片 = 本地参数分片（与本地参数对应）
  优化器状态分片 = 本地参数分片 × 2（Adam 的 m 和 v）
```

**示例计算（Qwen3-4B, DP=4, Gradient Checkpointing=On）**：

```
总参数：4B
本地参数分片：4B / 4 = 1B = 2GB (bf16)

单层参数：4B / 40 = 100M = 200MB (bf16, 临时)

激活值（开启 Gradient Checkpointing）：
  只保留每个 checkpoint 的激活，大约减少 40 倍
  原本：batch_size × seq_len × hidden_size × 40 layers
  现在：batch_size × seq_len × hidden_size × 1 layer ≈ 512MB

梯度分片：2GB (bf16)

优化器状态分片（Adam）：
  m: 2GB (fp32) × 2 = 4GB
  v: 2GB (fp32) × 2 = 4GB
  总计：8GB

峰值显存 = 2GB + 0.2GB + 0.5GB + 2GB + 8GB ≈ 13GB
```

### 3.2 CPU Offload 的内存管理

**FSDP2 原生 CPUOffloadPolicy**：

```python
offload_policy = CPUOffloadPolicy()

fully_shard(model, offload_policy=offload_policy, ...)
```

**工作机制**：

```
前向传播：
  参数从 CPU 加载到 GPU → All-Gather → 计算 → Free → 参数回到 CPU

反向传播：
  参数从 CPU 加载到 GPU → All-Gather → 计算梯度 → Reduce-Scatter
  → 梯度写回 CPU → Free GPU 参数

Optimizer Step：
  在 CPU 上执行（使用 CPU 上的参数和梯度）
```

**优势**：
- GPU 显存需求大幅降低（只需存储激活值）
- 适合显存受限的场景

**劣势**：
- CPU-GPU 数据传输开销
- Optimizer step 在 CPU 上执行，速度较慢

### 3.3 Mixed Precision 的内存优化

**MixedPrecisionPolicy**：

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,  # 参数存储为 bf16
    reduce_dtype=torch.float32,   # 梯度聚合为 fp32
)
```

**效果**：
- 参数内存减半（fp32 → bf16）
- 梯度聚合精度不受影响（reduce 时转为 fp32）
- 训练速度提升 20-30%

---

## 4. 并行通信模式

### 4.1 DP 通信（Data Parallel）

**主要操作**：

1. **Forward**: 无通信（每个 DP rank 独立计算）
2. **Backward**: Reduce-Scatter 梯度
3. **All-Gather**: 在需要完整参数时（forward/backward）

**通信量**：

```
每个训练步（假设模型 4B 参数，DP=4）：
  All-Gather（forward）：1B × 2 bytes × (4-1) / 4 = 1.5GB
  All-Gather（backward）：1B × 2 bytes × (4-1) / 4 = 1.5GB
  Reduce-Scatter（backward）：1B × 2 bytes × (4-1) / 4 = 1.5GB

  总计：4.5GB（每个 rank）
```

### 4.2 CP 通信（Context Parallel）

**主要操作**：

1. **Ring Flash Attention**: 在 CP 组内循环传递 KV cache
2. **All-Gather Log Probs**: 收集所有 CP rank 的结果

**通信量**：

```
每个 forward（假设 seq_len=8192, cp_size=2, hidden_size=3584）：
  Ring Flash Attention：
    传递 KV cache：seq_len/2 × hidden_size × 2 (K and V) × 2 bytes × 2 (ring rounds)
                  = 4096 × 3584 × 2 × 2 × 2 ≈ 117MB

  All-Gather Log Probs：
    seq_len/2 × 4 bytes × (cp_size-1) / cp_size = 4096 × 4 × 0.5 ≈ 8KB（很小）

  总计：≈ 117MB（每个 rank）
```

### 4.3 通信与计算重叠

**FSDP2 的优化**：

```python
# 异步 All-Gather
param = param.redistribute(
    placements=[Replicate()],
    async_op=True,  # 提前启动通信
)

# 在通信进行时，可以处理其他操作

# 等待通信完成
param = param.to_local()
```

**效果**：
- 减少通信等待时间
- 提高 GPU 利用率

---

## 5. 与原有生态的兼容性

### 5.1 HuggingFace 生态兼容

**关键点**：

1. **AutoModel API**：直接使用 `AutoModelForCausalLM.from_pretrained()`
2. **_no_split_modules**：自动推断切分粒度
3. **无需权重转换**：直接加载 HuggingFace checkpoint

**对比 Megatron**：

| 特性 | FSDP2 | Megatron |
|------|-------|----------|
| 权重格式 | HuggingFace 原生 | 需要转换为 `torch_dist` |
| 架构配置 | 自动推断（AutoConfig） | 手动指定大量参数 |
| 新模型支持 | 开箱即用 | 需要适配 Megatron Core |

### 5.2 SGLang 推理引擎兼容

**关键点**：

1. **权重同步**：通过 `update_weights()` 将训练后的权重传给 SGLang
2. **格式一致**：FSDP2 使用标准 torch.Tensor，SGLang 直接加载
3. **无缝切换**：Colocated 模式下，训练和推理共享同一组 GPU

**工作流程**：

```
Training (FSDP2):
  DTensor (Shard) → redistribute(Replicate) → Tensor (完整) → 传输

Inference (SGLang):
  接收 Tensor (完整) → 加载到推理引擎 → 开始推理
```

---

## 6. 在其他框架中复现 FSDP2 的关键点

### 6.1 必须实现的核心功能

1. **DeviceMesh 构建**：
   - 支持多维 mesh（至少 2D: DP × CP）
   - 提供维度命名和通信组提取

2. **DTensor 实现**：
   - 支持 Shard, Replicate, Partial 三种放置策略
   - 实现 `redistribute()` 方法（触发通信）
   - 实现 `to_local()` 和 `full_tensor()` 方法

3. **fully_shard API**：
   - 自动遍历模块并包装参数
   - 注册 pre/post hooks 管理参数生命周期
   - 支持 MixedPrecisionPolicy 和 CPUOffloadPolicy

4. **通信优化**：
   - 异步通信（async_op=True）
   - 通信与计算重叠
   - 分桶传输（避免峰值显存）

### 6.2 关键挑战

1. **参数生命周期管理**：
   - 何时 all-gather，何时释放
   - 避免重复通信
   - 处理 nested modules

2. **与 Autograd 集成**：
   - 在反向传播中自动触发 reduce-scatter
   - 处理梯度累积
   - 支持 gradient checkpointing

3. **内存优化**：
   - 精确控制临时 tensor 的生命周期
   - 避免内存泄漏
   - 处理 CPU-GPU 数据传输

### 6.3 参考实现

**PyTorch FSDP2**：
- 代码：`torch/distributed/fsdp/`
- 文档：https://pytorch.org/docs/stable/fsdp.html

**verl（字节跳动）**：
- 代码：https://github.com/volcengine/verl
- 参考：`verl/utils/fsdp_utils.py`（slime 借鉴了这个实现）

**DeepSpeed ZeRO**：
- 代码：https://github.com/microsoft/DeepSpeed
- 思想相同，实现不同（ZeRO-3 = FSDP 的前身）

---

## 7. 总结

### 7.1 回答问题-1 的核心要点

1. **DeviceMesh 维度**：
   - DP + CP 模式下是 **2D mesh**: `(dp_size, cp_size)`
   - Row-major 布局：rank = dp_idx * cp_size + cp_idx
   - 只有 **DP 维度** 传给 FSDP2，CP 维度用于 Ring Flash Attention

2. **fully_shard 自动切分**：
   - **无需手动指定**：通过 `model._no_split_modules` 自动发现需要包装的层
   - **自动遍历**：`model.named_modules()` 递归查找所有匹配的模块
   - **分层包装**：先包装 DecoderLayer，再包装顶层 model

3. **DTensor 切分机制**：
   - 参数在 **dim=0** 上切分（`Shard(0)`）
   - 前向/反向时临时 all-gather 成 `Replicate()`
   - 用完立即释放，只保留本地分片

### 7.2 关键 Takeaways

1. **Mesh 设计**：2D mesh 统一管理 DP 和 CP，但 FSDP2 只使用 DP 维度
2. **自动化**：HuggingFace 生态 + FSDP2 实现了完全自动化的参数切分
3. **DTensor 是核心**：理解 DTensor 的 placement 和 redistribute 是关键
4. **内存效率**：FSDP2 通过临时 all-gather 和即时释放实现显存优化
5. **通信优化**：异步操作和分桶传输降低通信开销

### 7.3 学习路径建议

1. **理解 DeviceMesh**：学习 PyTorch 的 `torch.distributed.device_mesh`
2. **学习 DTensor**：理解 Shard/Replicate/Partial 的含义和使用
3. **阅读 FSDP2 源码**：从 `fully_shard` 的实现入手
4. **实践 slime 代码**：运行 `scripts/run-qwen3-4B-fsdp.sh`，观察日志
5. **对比 verl**：参考字节跳动的 verl 实现，理解工程实践

---

## 附录

### A. 相关代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| DeviceMesh 构建 | `actor.py` | 164-210 |
| apply_fsdp2 | `actor.py` | 1016-1058 |
| DTensor redistribute | `update_weight_utils.py` | 57-62 |
| Log Prob 计算 (CP) | `actor.py` | 888-977 |
| Data Packing | `data_packing.py` | 11-101 |

### B. 关键 PyTorch API

```python
# DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("dp", "cp"))

# DTensor
from torch.distributed.tensor import DTensor, Shard, Replicate
dtensor = DTensor.from_local(local_tensor, device_mesh=mesh, placements=[Shard(0)])

# FSDP2
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
fully_shard(module, mesh=mesh, mp_policy=MixedPrecisionPolicy(...))

# Redistribute
replicated_dtensor = dtensor.redistribute(placements=[Replicate()], async_op=True)
```

### C. 调试技巧

```python
# 1. 打印 DeviceMesh 信息
print(f"Mesh shape: {mesh.shape}")
print(f"Mesh ndim: {mesh.ndim}")
print(f"DP group ranks: {mesh.get_group('dp').size()}")

# 2. 检查参数是否是 DTensor
for name, param in model.named_parameters():
    if isinstance(param, DTensor):
        print(f"{name}: DTensor, placements={param.placements}, local_shape={param.to_local().shape}")
    else:
        print(f"{name}: Tensor, shape={param.shape}")

# 3. 监控通信
import torch.distributed as dist
print(f"Rank {dist.get_rank()}: DP rank {dp_rank}, CP rank {cp_rank}")
```

---

**文档生成日期**: 2025-12-03
**Slime 版本**: main branch (commit 9d7f34d)
**作者**: 基于源码分析生成
