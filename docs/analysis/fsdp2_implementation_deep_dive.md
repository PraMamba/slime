# FSDP2 训练后端代码实现深度解读

> **作者**: AI 代码分析
> **日期**: 2025-12-03
> **版本**: v1.0

本文档深入分析 slime 项目中 FSDP2 作为训练后端的完整代码实现，覆盖关键 PR 和核心功能模块。

---

## 目录

1. [整体架构设计](#1-整体架构设计)
2. [核心模块代码解析](#2-核心模块代码解析)
3. [关键 PR 实现细节](#3-关键-pr-实现细节)
4. [性能优化机制](#4-性能优化机制)
5. [与 Megatron 的对比](#5-与-megatron-的对比)

---

## 1. 整体架构设计

### 1.1 代码结构

FSDP 后端的核心代码位于 `slime/backends/fsdp_utils/` 目录下：

```
slime/backends/fsdp_utils/
├── actor.py                    # 核心训练 Actor (1058 行)
├── data_packing.py             # 数据打包工具 (186 行)
├── update_weight_utils.py      # 权重更新机制 (259 行)
├── checkpoint.py               # Checkpoint 管理
├── arguments.py                # FSDP 参数定义
├── models/                     # 模型特定优化
│   ├── qwen3_moe.py           # Qwen3-MoE True On-Policy 优化
│   └── qwen3_moe_hf.py        # Qwen3-MoE HF 版本适配
└── kernels/                    # 自定义算子
    └── fused_experts.py       # Fused MoE 算子
```

### 1.2 接口标准化设计

FSDP Actor (`FSDPTrainRayActor`) 继承自 `TrainRayActor`，对外暴露统一接口：

```python
class FSDPTrainRayActor(TrainRayActor):
    """核心方法："""

    # 公开接口 (对外暴露)
    def init(self, args, role, with_ref=False) -> int
    def train(self, rollout_id, rollout_data_ref) -> None
    def save_model(self, iteration) -> None
    def update_weights(self) -> None
    def sleep(self) -> None
    def wake_up(self) -> None

    # 私有方法 (内部使用，下划线约定)
    def _train_core(self, rollout_id, rollout_data) -> None
    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum)
    def _compute_log_prob(self, model_tag, packed_batches, store_prefix="")
    def _packed_data(self, rollout_data) -> tuple
    def _setup_device_mesh(self) -> None
    def _get_model_inputs_args(self, packed_sequence) -> dict
    ...
```

**设计优势**：
- **物理隔离**：利用 Ray Actor 机制将 FSDP 和 Megatron 封装在独立进程空间
- **统一原语**：上层调度器无需关注分布式细节，只需调用标准接口
- **易于维护**：减少全局变量冲突和条件分支复杂度

---

## 2. 核心模块代码解析

### 2.1 初始化流程 (`init` 方法)

#### 2.1.1 Device Mesh 设置

**代码位置**: `actor.py:164-210`

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
    self.cp_size = self.args.context_parallel_size
    self.dp_size = world_size // self.cp_size

    # Create 2D device mesh: (dp_size, cp_size)
    # Ranks laid out in row-major: mesh[dp_idx, cp_idx] = dp_idx * cp_size + cp_idx
    # - CP groups: consecutive ranks along dim 1, e.g., [0,1], [2,3], [4,5], [6,7]
    # - DP groups: striped ranks along dim 0, e.g., [0,2,4,6], [1,3,5,7]
    self.mesh = init_device_mesh(
        "cuda",
        mesh_shape=(self.dp_size, self.cp_size),
        mesh_dim_names=("dp", "cp")
    )

    # Extract process groups from mesh
    self.dp_group = self.mesh.get_group("dp")  # For FSDP gradient sync, metric reduction
    self.cp_group = self.mesh.get_group("cp")  # For Ring Flash Attention, logit gathering
    self.dp_mesh = self.mesh["dp"]             # For FSDP

    # Compute local ranks within each dimension
    self.dp_rank = rank // self.cp_size
    self.cp_rank = rank % self.cp_size

    # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
    if self.cp_size > 1:
        substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
        logger.info(f"[Rank {rank}] CP initialized via device mesh")
    else:
        logger.info(f"[Rank {rank}] Pure DP mode (cp_size=1)")
```

**关键设计**：
1. **统一的 2D Mesh**: 即使在纯 DP 模式 (`cp_size=1`) 下也使用 2D mesh，保证代码逻辑统一
2. **Row-major 布局**: `rank = dp_idx * cp_size + cp_idx`，使得：
   - CP 组：连续的 rank，如 `[0,1]`, `[2,3]`，便于序列切片
   - DP 组：跨步的 rank，如 `[0,2,4,6]`，便于梯度同步
3. **Ring Flash Attention 集成**: 通过 `substitute_hf_flash_attn` 替换 HuggingFace 的标准 attention

#### 2.1.2 模型加载与 FSDP 包装

**代码位置**: `actor.py:82-101`

```python
init_context = self._get_init_weight_context_manager()

with init_context():
    model = AutoModelForCausalLM.from_pretrained(
        self.args.hf_checkpoint,
        trust_remote_code=True,
        attn_implementation=self.args.attn_implementation,
    )

model.train()

full_state = model.state_dict()

# Apply FSDP2 wrapping
model = apply_fsdp2(model, mesh=self.dp_mesh, cpu_offload=self.fsdp_cpu_offload)

# Load weights with efficient broadcast from rank 0
model = self._fsdp2_load_full_state_dict(
    model, full_state, self.dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
)

self.model = model
```

**关键机制**：

1. **Rank-0 Broadcast 优化** (`_fsdp2_load_full_state_dict` 方法):

```python
def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
    """Load full state dict into FSDP2 model with efficient broadcast from rank 0.

    This function loads weights from rank 0 and broadcasts to all other ranks,
    avoiding the need for each rank to load the full model from disk.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # Rank 0: move with weights, others: allocate empty tensors on device
    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        # to_empty creates tensors on device without initializing memory
        model = model.to_empty(device=torch.cuda.current_device())

    is_cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=is_cpu_offload,
        broadcast_from_rank0=True  # 关键参数：从 rank 0 广播
    )

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

**优化效果**：
- **PR #915**: 引入此优化，避免所有 rank 从磁盘加载完整模型
- **性能提升**: 大模型加载时间从 O(world_size) 降低到 O(1)

2. **Meta Device 初始化** (`_get_init_weight_context_manager` 方法):

```python
def _get_init_weight_context_manager(self):
    """Get context manager for model initialization.

    Uses meta device (no memory allocation) for non-rank-0 processes,
    UNLESS tie_word_embeddings=True (which causes hangs with meta tensors).
    """
    from accelerate import init_empty_weights

    # Check if model uses tied word embeddings (which doesn't work with meta tensors)
    use_meta_tensor = not self.hf_config.tie_word_embeddings

    def cpu_init_weights():
        return torch.device("cpu")

    if use_meta_tensor:
        # Rank 0: CPU, others: meta device (memory efficient for large models)
        return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
    else:
        logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
        return cpu_init_weights
```

**边界情况处理**：
- `tie_word_embeddings=True` 时不使用 meta tensor（会导致hang）
- 大多数模型可以使用 meta device 节省初始化内存

3. **FSDP2 包装** (`apply_fsdp2` 函数):

```python
def apply_fsdp2(model, mesh=None, cpu_offload=False):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    # Get layer classes to wrap (e.g., DecoderLayer)
    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # 参数存储为 bf16
            reduce_dtype=torch.float32,   # 梯度聚合为 fp32
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model
```

**关键配置**：
- **Mixed Precision**: 参数 bf16 存储，梯度 fp32 聚合，平衡性能和精度
- **分层包装**: 先包装 Decoder Layer，再包装顶层模型
- **Embedding 特殊处理**: 非 tied 的 Embedding 也单独包装

#### 2.1.3 Reference Model 创建

**代码位置**: `actor.py:768-810`

```python
def _create_ref_model(self, ref_load_path: str | None):
    """Create and initialize a separate reference model with FSDP2 CPUOffloadPolicy.

    Parameters:
        ref_load_path: Path to a directory containing a HF checkpoint. If
            None, a ValueError is raised.

    Returns:
        FSDP2-wrapped ref model with CPU offload enabled

    Note:
        Creates a separate FSDP2 model instance for the reference model.
        ALWAYS uses CPUOffloadPolicy for the reference model to save memory,
        regardless of the actor model's CPU offload setting.
    """
    if ref_load_path is None:
        raise ValueError("ref_load_path must be provided when loading reference model")

    import os

    if os.path.isdir(ref_load_path):
        logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

        init_context = self._get_init_weight_context_manager()

        with init_context():
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_load_path,
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
            )

        full_state = ref_model.state_dict()

        # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload.
        # It is faster than model.cpu().
        ref_model = apply_fsdp2(ref_model, mesh=self.dp_mesh, cpu_offload=True)
        ref_model = self._fsdp2_load_full_state_dict(ref_model, full_state, self.dp_mesh, cpu_offload=True)

        logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
        return ref_model
    else:
        raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")
```

**关键设计**：
- **独立的 FSDP2 实例**: Ref model 是独立的 FSDP2 包装，非手动权重交换
- **强制 CPU Offload**: Ref model 始终开启 CPU offload（无论 actor 是否开启）
- **按需加载**: 仅在计算 KL penalty 时加载到 GPU，用完立即 offload

**相关 PR**:
- **#780**: 修复 PPO KL 精度误差，从手动权重交换改为独立 FSDP2 实例
- **旧实现问题**: 手动交换 DTensor 会引入微小数值偏差，导致 on-policy KL ≠ 0
- **新实现优势**: 利用 FSDP2 原生 CPU/GPU 转移机制，完全避免数值漂移

---

### 2.2 训练流程 (`train` 方法)

#### 2.2.1 数据准备与打包

**训练流程概览** (`_train_core` 方法):

```python
def _train_core(self, rollout_id: int, rollout_data) -> None:
    # 1. 计算 advantages
    if self.args.advantage_estimator in ["grpo", "gspo"]:
        rollout_data["advantages"] = rollout_data["returns"] = [
            torch.tensor([rollout_data["rewards"][i]] * rollout_data["response_lengths"][i])
            for i in range(len(rollout_data["rewards"]))
        ]
    else:
        raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

    # 2. Pack data into micro-batches
    packed_batches, grad_accum = self._packed_data(rollout_data)

    # 3. Compute ref log_probs (if needed)
    if self.ref_model is not None:
        self._compute_log_prob("ref", packed_batches, store_prefix="ref_")

    # 4. Compute actor log_probs
    self._compute_log_prob("actor", packed_batches)

    # 5. Log rollout metrics
    self._log_rollout_data(rollout_id, rollout_data, packed_batches)

    # 6. Train actor
    with timer("actor_train"):
        reported_accum: dict[str, list[torch.Tensor]] = {}
        self.optimizer.zero_grad(set_to_none=True)
        for mbs_id, packed_batch in enumerate(tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0)):
            self._train_step(
                packed_batch=packed_batch,
                reported_accum=reported_accum,
                mbs_id=mbs_id,
                grad_accum=grad_accum,
            )
```

**数据打包流程** (`_packed_data` 方法):

```python
def _packed_data(self, rollout_data: dict[str, list[torch.Tensor]]) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
    """Pack variable-length sequences for efficient processing."""

    tokens = rollout_data["tokens"]

    packed_batches = []
    mbs_size_list = []
    local_batch_size = self.args.global_batch_size // self.dp_size

    # Determine num_microbatches for each local_batch_size chunk
    if self.args.use_dynamic_batch_size:
        # In CP mode, CP group shares sequences, so total capacity is max_tokens_per_gpu * cp_size
        max_tokens = self.args.max_tokens_per_gpu
        if self.cp_size > 1:
            max_tokens = max_tokens * self.cp_size

        for i in range(0, len(tokens), local_batch_size):
            mbs_size_list.append(
                get_minimum_num_micro_batch_size(
                    [len(t) for t in rollout_data["tokens"][i : i + local_batch_size]],
                    max_tokens,
                )
            )
        # Synchronize across DP ranks (take max to ensure same num steps)
        num_microbatches = torch.tensor(mbs_size_list, dtype=torch.int, device=torch.cuda.current_device())
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=self.dp_group)
        num_microbatches = num_microbatches.tolist()
    else:
        num_microbatches = [self.args.global_batch_size // (self.args.micro_batch_size * self.dp_size)] * (
            len(tokens) // local_batch_size
        )

    # Pack each chunk
    start = 0
    for mbs_size in num_microbatches:
        end = start + local_batch_size
        packed_batches.extend(
            pack_sequences(
                rollout_data["tokens"][start:end],
                rollout_data["loss_masks"][start:end],
                rollout_data["rewards"][start:end],
                rollout_data["raw_reward"][start:end],
                rollout_data["response_lengths"][start:end],
                rollout_data["advantages"][start:end],
                rollout_data["returns"][start:end],
                rollout_log_probs=(
                    rollout_data["rollout_log_probs"][start:end] if "rollout_log_probs" in rollout_data else None
                ),
                num_packs=mbs_size,
            )
        )
        start = end

    grad_accum = list(accumulate(num_microbatches))

    return packed_batches, grad_accum
```

**关键机制**：
1. **动态 Batch Size**: 根据序列长度动态调整 micro-batch 数量
2. **CP 模式调整**: CP 模式下 `max_tokens *= cp_size`（因为序列会被切分）
3. **DP 同步**: 所有 DP rank 使用相同的 micro-batch 数量（取 max）
4. **梯度累积点**: `grad_accum` 记录何时执行 optimizer step

**Data Packing 实现** (`pack_sequences` 函数):

```python
def pack_sequences(
    tokens: list[list[int]],
    loss_masks: list[list[int]],
    rewards: list[float],
    raw_rewards: list,
    response_lengths: list[int],
    advantages: list[float],
    returns: list[float],
    rollout_log_probs: list[list[float]] | None = None,
    max_tokens_per_gpu: int | None = None,
    num_packs: int | None = None,
) -> list[dict]:
    """Pack sequences into dense batches with cumulative sequence lengths."""

    if not tokens:
        return []

    seq_lengths = [len(t) for t in tokens]

    # Determine number of packs
    if num_packs:
        k_partitions = num_packs
    elif max_tokens_per_gpu:
        total_tokens = sum(seq_lengths)
        k_partitions = max(1, math.ceil(total_tokens / max_tokens_per_gpu))
    else:
        k_partitions = 1

    # Use balanced partitioning for optimal load distribution
    partitions = get_seqlen_balanced_partitions(
        seq_lengths,
        k_partitions=k_partitions,
        equal_size=False  # Allow variable sizes for better balance
    )

    # Pack each partition
    result = []
    for indices in partitions:
        # Build cumulative sequence lengths
        cu_seqlens = [0]
        flat_tokens = []
        flat_masks = []
        flat_positionids = []
        flat_advantages = []
        flat_returns = []
        flat_rollout_log_probs = []

        for i in indices:
            seq_tokens = tokens[i]
            seq_mask = loss_masks[i]
            seq_positionids = list(range(len(seq_tokens)))

            flat_tokens.extend(seq_tokens)
            flat_positionids.extend(seq_positionids)
            flat_masks.extend(seq_mask)
            flat_advantages.extend(advantages[i])
            flat_returns.extend(returns[i])
            if rollout_log_probs:
                flat_rollout_log_probs.extend(rollout_log_probs[i])
            cu_seqlens.append(cu_seqlens[-1] + len(seq_tokens))

        result.append({
            "tokens": torch.tensor(flat_tokens, dtype=torch.long),
            "loss_masks": torch.tensor(flat_masks, dtype=torch.int),
            "position_ids": torch.tensor(flat_positionids, dtype=torch.int),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),  # 关键：记录边界
            "rewards": torch.tensor([rewards[i] for i in indices], dtype=torch.float32),
            "raw_reward": [raw_rewards[i] for i in indices],
            "response_lengths": [response_lengths[i] for i in indices],
            "advantages": torch.tensor(flat_advantages, dtype=torch.float32),
            "returns": torch.tensor(flat_returns, dtype=torch.float32),
            "rollout_log_probs": torch.tensor(flat_rollout_log_probs, dtype=torch.float32, device=torch.cuda.current_device()),
        })

    return result
```

**相关 PR**:
- **#321**: 引入 Data Packing 机制
- **算法**: Karmarkar-Karp (最大差分法) 实现负载均衡分配
- **效果**: 消除传统 Padding 带来的算力浪费，各 pack 的 token 数高度均衡

**Unpack 实现** (`unpack_sequences` 函数):

```python
def unpack_sequences(packed_batch: dict) -> list[dict]:
    """Unpack sequences from a packed batch."""

    cu_seqlens = packed_batch["cu_seqlens"]
    num_sequences = len(cu_seqlens) - 1
    response_lengths = packed_batch["response_lengths"]

    instances = []

    # Calculate pad_length by counting trailing zeros
    tokens = packed_batch["tokens"]
    nonzero_indices = (tokens != 0).nonzero(as_tuple=True)[0]
    if len(nonzero_indices) > 0:
        pad_length = len(tokens) - nonzero_indices[-1].item() - 1
    else:
        pad_length = 0

    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        instance = {}

        for key, value in packed_batch.items():
            if isinstance(value, torch.Tensor):
                if key in ["log_probs", "ref_log_probs", "cur_log_probs", "entropy"]:
                    # These are computed from logits[:-1] so they have length seq_len-1
                    instance[key] = value[end_idx - 1 - response_lengths[i] - pad_length : end_idx - 1 - pad_length]
                elif key == "rollout_log_probs":
                    # rollout_log_probs is packed based on response_lengths
                    instance[key] = value[sum(response_lengths[:i]) : sum(response_lengths[: i + 1])]
                elif key in ["tokens", "position_ids"]:
                    instance[key] = value[start_idx:end_idx]
                elif key in ["loss_masks", "advantages", "returns"]:
                    instance[key] = value[sum(response_lengths[:i]) : sum(response_lengths[: i + 1])]
            elif isinstance(value, list):
                instance[key] = value[i]

        instances.append(instance)

    return instances
```

**关键点**：
- 根据 `cu_seqlens` 精确还原每条序列
- 处理 CP padding (通过 trailing zeros 检测)
- 不同 tensor 有不同的切片逻辑（log_probs 是 seq_len-1）

#### 2.2.2 Log Probability 计算

**代码位置**: `actor.py:307-377`

```python
def _compute_log_prob(
    self,
    model_tag: str,
    packed_batches: list[dict[str, torch.Tensor]],
    store_prefix: str = "",
) -> dict[str, list[torch.Tensor]]:
    """Compute token log-probabilities for a list of packed batches.

    Parameters:
        model_tag: Which parameters to use, e.g. "actor" or "ref".
        packed_batches: A list of packed batch dictionaries produced by
            `pack_sequences`, each containing at least `tokens` and
            `position_ids`; may also include multimodal keys like `pixel_values`.
        store_prefix: Prefix to use for keys in outputs (e.g., "ref_").

    Note:
        Uses separate ref model when model_tag == "ref". The ref model is
        loaded from CPU to GPU on-demand and offloaded back after use.
    """
    # Select which model to use
    if model_tag == "ref" and self.ref_model is not None:
        if not self.fsdp_cpu_offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            dist.barrier(group=get_gloo_group())

        active_model = self.ref_model
        active_model.eval()
    else:
        active_model = self.model

    try:
        rollout_data = {f"{store_prefix}log_probs": []}
        with timer(f"{store_prefix}log_probs"), torch.no_grad():
            for batch in tqdm(packed_batches, desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0):
                model_args = self._get_model_inputs_args(batch)
                if "pixel_values" in batch:
                    model_args["pixel_values"] = batch["pixel_values"]

                # Forward pass
                logits = active_model(**model_args).logits.squeeze(0).float()

                # Compute log_probs and entropy (unified for CP and non-CP)
                log_probs_result, entropy_result = get_logprob_and_entropy_with_cp(
                    logits=logits,
                    target_tokens=batch["tokens"],
                    cp_rank=self.cp_rank,
                    cp_size=self.cp_size,
                    cp_group=self.cp_group,
                    model_input_ids=model_args["input_ids"],
                    allow_compile=not self.args.true_on_policy_mode,  # 关键：True on-policy 禁用编译
                    temperature=self.args.rollout_temperature,        # 关键：复用 rollout temperature
                )

                batch[f"{store_prefix}log_probs"] = log_probs_result
                if store_prefix == "":
                    batch["entropy"] = entropy_result
        return rollout_data

    finally:
        # Restore actor model if it was offloaded
        if model_tag == "ref" and self.ref_model is not None:
            torch.cuda.empty_cache()
            dist.barrier(group=get_gloo_group())

            if not self.fsdp_cpu_offload:
                self.model.cuda()
                dist.barrier(group=get_gloo_group())
```

**关键机制**：

1. **Ref Model On-Demand Loading**:
   - Ref model 计算时才加载到 GPU
   - 用完立即 offload 回 CPU
   - 避免长时间占用 GPU 内存

2. **True On-Policy 优化**:
   - `allow_compile=not self.args.true_on_policy_mode`
   - 禁用 `torch.compile` 避免编译引入的数值偏差
   - 复用 `rollout_temperature` 确保完全一致

**Log Prob 计算核心** (`get_logprob_and_entropy_with_cp` 函数):

```python
def get_logprob_and_entropy_with_cp(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    cp_group,
    model_input_ids: torch.Tensor,
    allow_compile: bool,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy in Context Parallel mode.

    Returns:
        log_probs: Aggregated log probabilities with shape [total_seq_len - 1]
        entropy: Aggregated entropy with shape [total_seq_len - 1]
    """
    # Fast path for non-CP mode (cp_size=1): avoid unnecessary communication
    if cp_size == 1:
        shifted_logits = logits[:-1, :]
        local_log_probs = gather_log_probs_packed(
            shifted_logits, target_tokens, allow_compile=allow_compile, temperature=temperature
        )
        log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
        probs = torch.softmax(shifted_logits, dim=-1)
        entropy = -(probs * log_probs_full).sum(dim=-1)
        return local_log_probs, entropy

    # CP mode: process local chunk
    chunk_size = logits.shape[0]
    tokens_start_index = chunk_size * cp_rank
    tokens_end_index = (
        tokens_start_index + chunk_size + 1 if cp_rank < cp_size - 1
        else tokens_start_index + chunk_size
    )

    # For the last rank, remove the last logit
    logits = logits if cp_rank < cp_size - 1 else logits[:-1, :]

    # Get local tokens for current rank
    local_tokens = (
        target_tokens[tokens_start_index:tokens_end_index] if cp_rank < cp_size - 1
        else model_input_ids.squeeze(0)
    )

    # Compute local log probs
    local_log_probs = gather_log_probs_packed(
        logits, local_tokens, allow_compile=allow_compile, temperature=temperature
    )

    # Pad for the last rank
    if cp_rank == cp_size - 1:
        local_log_probs = F.pad(local_log_probs, (0, chunk_size - local_log_probs.shape[0]), value=0)

    # Compute entropy
    shifted_logits = logits[:-1, :] if cp_rank == cp_size - 1 else logits
    log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
    probs = torch.softmax(shifted_logits, dim=-1)
    entropy = -(probs * log_probs_full).sum(dim=-1)

    # Pad entropy for the last rank
    if cp_rank == cp_size - 1:
        entropy = F.pad(entropy, (0, chunk_size - entropy.shape[0]), value=0)

    # Merge with a single all_gather: stack as [2, chunk_size]
    stacked_local = torch.stack([local_log_probs, entropy], dim=0)
    gathered_stacked = torch.distributed.nn.functional.all_gather(stacked_local, group=cp_group)

    # Concatenate by effective length
    lp_parts, ent_parts = [], []
    for r in range(cp_size):
        eff_len = chunk_size if r < cp_size - 1 else max(0, chunk_size - 1)
        if eff_len > 0:
            lp_parts.append(gathered_stacked[r][0][:eff_len])
            ent_parts.append(gathered_stacked[r][1][:eff_len])

    log_probs = torch.cat(lp_parts, dim=0) if lp_parts else local_log_probs.new_zeros((0,))
    entropy_result = torch.cat(ent_parts, dim=0) if ent_parts else entropy.new_zeros((0,))

    # Truncate to global effective length T-1
    log_probs = log_probs[: len(target_tokens) - 1]
    entropy_result = entropy_result[: len(target_tokens) - 1]

    return log_probs, entropy_result
```

**CP 模式关键优化**：
1. **Fast Path**: `cp_size=1` 时直接返回，避免不必要的通信
2. **单次 All-Gather**: 将 log_probs 和 entropy stack 后一起 gather，减少通信次数
3. **边界处理**: 最后一个 rank 的 chunk 长度为 `chunk_size - 1`，需要特殊处理

**相关 PR**:
- **#467**: 引入 Context Parallel 支持
- **#906, #917, #934, #1001**: 多个 PR 逐步完善 True On-Policy 支持

#### 2.2.3 训练步骤与 Loss 计算

**代码位置**: `actor.py:561-747`

```python
def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
    # 1. Prepare model inputs
    model_args = self._get_model_inputs_args(packed_batch)
    logits = self.model(**model_args).logits.squeeze(0).float()

    # 2. Compute log probs and entropy
    log_probs, entropy_result = get_logprob_and_entropy_with_cp(
        logits=logits,
        target_tokens=packed_batch["tokens"],
        cp_rank=self.cp_rank,
        cp_size=self.cp_size,
        cp_group=self.cp_group,
        model_input_ids=model_args["input_ids"],
        allow_compile=not self.args.true_on_policy_mode,
        temperature=self.args.rollout_temperature,
    )
    packed_batch["cur_log_probs"] = log_probs
    packed_batch["entropy"] = entropy_result

    # 3. Unpack sequences
    unpacked_batches = unpack_sequences(packed_batch)

    # 4. Prepare tensors for loss computation
    old_log_prob_key = "rollout_log_probs" if self.args.use_rollout_logprobs else "log_probs"
    old_log_probs = torch.cat([batch[old_log_prob_key] for batch in unpacked_batches], dim=0)
    log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
    advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)
    loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
    response_lengths = [batch["response_lengths"] for batch in unpacked_batches]

    advantages = advantages.to(device=log_probs.device)
    old_log_probs = old_log_probs.to(device=log_probs.device)
    ppo_kl = old_log_probs - log_probs

    # 5. Compute OPSM mask (if enabled)
    if self.args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=self.args,
            full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
            full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
            advantages=[batch["advantages"] for batch in unpacked_batches],
            loss_masks=loss_masks,
        )

    # 6. Compute GSPO KL (if GSPO)
    if self.args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
            full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
            local_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
            loss_masks=loss_masks,
        )

    # 7. Compute policy gradient loss
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high)

    if self.args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # 8. Apply TIS (Truncated Importance Sampling) if enabled
    has_rollout_log_probs = all(
        isinstance(batch.get("rollout_log_probs"), torch.Tensor) and batch["rollout_log_probs"].numel() > 0
        for batch in unpacked_batches
    )
    rollout_log_probs = (
        torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)
        if has_rollout_log_probs
        else None
    )

    if self.args.use_tis:
        assert has_rollout_log_probs and rollout_log_probs is not None, \
            "rollout_log_probs must be provided as non-empty torch.Tensor for TIS"

        # TIS importance weight: π_old / π_rollout
        tis = torch.exp(old_log_probs - rollout_log_probs)
        ois = (-ppo_kl).exp()  # π_current / π_old
        tis_clip = torch.clamp(
            tis,
            min=getattr(self.args, "tis_clip_low", 0.1),
            max=getattr(self.args, "tis_clip", 2.0)
        )
        tis_clipfrac = tis_clip != tis

        # Apply TIS to policy gradient loss
        pg_loss = pg_loss * tis_clip

    # 9. Aggregate loss per sample
    pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac, response_lengths, loss_masks)
    ppo_kl = sum_of_sample_mean(ppo_kl.abs(), response_lengths, loss_masks)

    # 10. Compute train-rollout mismatch metric
    train_rollout_logprob_abs_diff = None
    if not self.args.use_rollout_logprobs and rollout_log_probs is not None:
        train_rollout_logprob_abs_diff = (old_log_probs - rollout_log_probs).abs()
        train_rollout_logprob_abs_diff = sum_of_sample_mean(
            train_rollout_logprob_abs_diff, response_lengths, loss_masks
        ).detach()

    # 11. Compute entropy loss
    entropy = torch.cat([batch["entropy"] for batch in unpacked_batches], dim=0)
    entropy_loss = sum_of_sample_mean(entropy, response_lengths, loss_masks)

    # 12. Total loss
    loss = pg_loss - self.args.entropy_coef * entropy_loss

    # 13. Add KL penalty (if using ref model)
    if self.args.use_kl_loss:
        ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=self.args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl, response_lengths, loss_masks)
        loss = loss + self.args.kl_loss_coef * kl_loss

    # 14. Collect metrics
    reported = {
        "loss": loss.detach(),
        "pg_loss": pg_loss.detach(),
        "pg_clipfrac": pg_clipfrac.detach(),
        "ppo_kl": ppo_kl.detach(),
        "entropy_loss": entropy_loss.detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff

    if self.args.use_kl_loss:
        reported["kl_loss"] = kl_loss.detach()

    if self.args.use_opsm:
        reported["opsm_clipfrac"] = opsm_clipfrac

    if self.args.use_tis and tis is not None:
        reported["tis"] = sum_of_sample_mean(tis, response_lengths, loss_masks).detach()
        reported["ois"] = sum_of_sample_mean(ois, response_lengths, loss_masks).detach()
        reported["tis_clipfrac"] = sum_of_sample_mean(tis_clipfrac.float(), response_lengths, loss_masks).detach()

    # 15. Scale loss for gradient accumulation
    loss = loss * self.dp_size / self.args.global_batch_size
    loss.backward()

    # 16. Accumulate reported metrics
    for k, v in reported.items():
        reported_accum.setdefault(k, []).append(v)

    # 17. Optimizer step (at gradient accumulation boundaries)
    if (mbs_id + 1) in grad_accum:
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
        grad_norm = float(grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Aggregate logs across DP ranks
        aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
        reduced_aggregated = [None] * self.dp_size
        dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
        aggregated = {}
        for k in reported_accum.keys():
            aggregated[k] = sum([r[k] for r in reduced_aggregated]) / (self.args.global_batch_size)
        reported_accum.clear()

        if dist.get_rank() == 0:
            log_dict = {
                f"train/{k}": (val.item() if torch.is_tensor(val) else val)
                for k, val in aggregated.items()
            }
            log_dict["train/grad_norm"] = grad_norm

            for gid, group in enumerate(self.optimizer.param_groups):
                if "lr" in group:
                    log_dict[f"train/lr-pg_{gid}"] = group["lr"]

            logger.info(f"step {self.global_step}: {log_dict}")
            log_dict["train/step"] = self.global_step
            tracking_utils.log(self.args, log_dict, step_key="train/step")

        self.global_step += 1
```

**TIS (Truncated Importance Sampling) 公式**：

$$
\mathcal{L}(\theta) = \frac{1}{L} \sum_{t=1}^L \left[ \bar{w}_t \cdot \mathcal{L}^{\text{clip}}_t(\theta) - \beta \text{KL}_t + \lambda H_t \right]
$$

其中：
- $\mathcal{L}^{\text{clip}}_t = \min \left( r_t(\theta) A_t, \ \text{clip}(r_t(\theta), 1\pm\epsilon) A_t \right)$
- $r_t(\theta) = \frac{\pi_{\theta}}{\pi_{\text{old}}}$ (importance ratio)
- $\bar{w}_t = \text{min}\left( \frac{\pi_{\text{old}}}{\pi_{\text{rollout}}}, C \right)$ (TIS weight)

**关键设计**：
1. **训推差异监控**: `train_rollout_logprob_abs_diff` 实时量化差异
2. **TIS 重加权**: 通过 importance weight 缓解 off-policy 影响
3. **Per-Sample Mean**: `sum_of_sample_mean` 确保每个样本的 loss 权重相等

---

### 2.3 权重更新机制

#### 2.3.1 两种更新模式

**UpdateWeightFromTensor** (Colocated 模式):

```python
class UpdateWeightFromTensor(UpdateWeight):
    """Push model weights to rollout engines using tensors.

    Streams parameters in size-bounded buckets; optionally groups tensors by dtype
    and flattens per dtype, gathers per-rank blobs to the source, and issues one
    RPC per dtype per bucket (or one per bucket if not flattened).
    """

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock) -> None:
        """Attach rollout engines and create per-engine IPC (Gloo) groups."""
        self.rollout_engines = rollout_engines

        # Create IPC groups for each engine
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")

            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine
                self.tp_rank = dist.get_rank() - start_rank

    def update_bucket_weights(self, named_tensors) -> None:
        """Send flattened tensor buckets to rollout engines."""
        monkey_patch_torch_reductions()

        # Group tensors by dtype
        named_tensors_by_dtypes = {}
        for name, tensor in named_tensors:
            dtype = tensor.dtype
            if dtype not in named_tensors_by_dtypes:
                named_tensors_by_dtypes[dtype] = []
            named_tensors_by_dtypes[dtype].append((name, tensor))

        # Create flattened bucket for each dtype group
        serialized_tensors = []
        for _dtype, named_tensors in named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

        # Gather to source rank
        if self._ipc_gather_src == dist.get_rank():
            gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
        else:
            gathered_serialized_batches = None

        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered_serialized_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        # Source rank sends to rollout engine
        if dist.get_rank() == self._ipc_gather_src:
            num_dtypes = len(gathered_serialized_batches[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                    "load_format": "flattened_bucket",
                    "flush_cache": False,
                }
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)

            # Flush cache
            ref = self._ipc_engine.flush_cache.remote()
            ray.get(ref)
```

**UpdateWeightFromDistributed** (Disaggregated 模式):

```python
class UpdateWeightFromDistributed(UpdateWeight):
    """Broadcast weights via a temporary NCCL group to rollout engines."""

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock) -> None:
        """On rank 0, initialize a temporary NCCL group for parameter broadcast."""
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        self._is_src_rank = dist.get_rank() == 0
        if self._is_src_rank:
            self._group_name = "slime"
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            world_size = self.args.rollout_num_gpus + 1

            # Initialize NCCL group on rollout engines
            refs = [
                engine.init_weights_update_group.remote(
                    master_address,
                    master_port,
                    i * self.args.rollout_num_gpus_per_engine + 1,
                    world_size,
                    self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(self.rollout_engines)
            ]

            # Initialize NCCL group on rank 0
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
            )
            ray.get(refs)

    def update_bucket_weights(self, named_tensors) -> None:
        """Send names/dtypes/shapes metadata to engines, then broadcast tensors."""
        if not self._is_src_rank or not named_tensors:
            return

        # Send metadata
        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named_tensors],
                dtypes=[param.dtype for _, param in named_tensors],
                shapes=[param.shape for _, param in named_tensors],
                group_name=self._group_name,
            )
            for engine in self.rollout_engines
        ]

        # Broadcast parameters one by one
        handles = []
        for _name, param in named_tensors:
            torch.cuda.empty_cache()
            param_data = param.data.contiguous()

            # Handle DTensor
            if dist.get_world_size() == 1 and isinstance(param_data, DTensor):
                param_data = param_data.full_tensor()

            handles.append(dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=True))

        for handle in handles:
            handle.wait()
        ray.get(refs)
```

**分桶异步更新** (`update_weights` 方法):

```python
def update_weights(self) -> None:
    bucket = []
    bucket_size = 0
    for name, param in self.model.state_dict().items():
        param_size = param.numel() * param.element_size()

        # Check if bucket is full
        if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
            self.wait_and_update_bucket_weights(bucket)
            del bucket
            bucket = []
            bucket_size = 0

        param = param.cuda()

        # Handle DTensor (async replicate)
        if isinstance(param, DTensor):
            param = param.redistribute(
                placements=[Replicate()] * param.device_mesh.ndim,
                async_op=True,  # 关键：异步操作
            ).to_local()

        bucket.append((name, param))
        bucket_size += param_size

    # Update remaining bucket
    if bucket:
        self.wait_and_update_bucket_weights(bucket)
        del bucket
        bucket = []
        bucket_size = 0

def wait_and_update_bucket_weights(self, bucket):
    # Wait for async operations to complete
    bucket = [(name, param.wait()) if hasattr(param, "wait") else (name, param) for name, param in bucket]
    self.update_bucket_weights(bucket)
```

**关键优化**：
1. **分桶更新**: 避免峰值内存占用过高
2. **异步 DTensor Replicate**: 提前启动 replicate，减少等待时间
3. **Dtype 分组**: 同一 dtype 的参数打包在一起，提高传输效率

**相关 PR**:
- **#341**: 引入 Distributed 模式权重更新
- **#729**: 优化 Distributed 模式性能
- **#861**: 重构权重更新，统一 Colocated 和 Distributed 逻辑

---

### 2.4 Context Parallel 实现

#### 2.4.1 输入切分

**代码位置**: `actor.py:811-832`

```python
def _get_model_inputs_args(self, packed_sequence: dict) -> dict:
    input_ids = packed_sequence["tokens"].unsqueeze(0)
    position_ids = packed_sequence["position_ids"].unsqueeze(0)

    if self.cp_size > 1:
        # Pad packed sequence to make length divisible by cp_size
        packed_sequence = pad_packed_sequence_with_cp(packed_sequence, self.cp_size)

        # Move cu_seqlens to CUDA
        if not packed_sequence["cu_seqlens"].is_cuda:
            packed_sequence["cu_seqlens"] = packed_sequence["cu_seqlens"].cuda()
        cu_seqlens = packed_sequence["cu_seqlens"]

        # Update ring flash attention parameters
        update_ring_flash_attn_params(cu_seqlens, self.cp_group)

        # Chunk inputs by cp_size (along sequence dimension)
        input_ids = torch.chunk(packed_sequence["tokens"].unsqueeze(0), self.cp_size, dim=1)[self.cp_rank]
        position_ids = torch.chunk(packed_sequence["position_ids"].unsqueeze(0), self.cp_size, dim=1)[self.cp_rank]

    model_args = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
    }
    return model_args
```

**CP Padding** (`pad_packed_sequence_with_cp` 函数):

```python
def pad_packed_sequence_with_cp(packed_sequence: dict, cp_size: int) -> dict:
    """Pad packed sequence to make total length divisible by cp_size.

    Returns:
        Padded packed sequence
    """
    seq_length = len(packed_sequence["tokens"])
    # Calculate padding needed: (cp_size - seq_length % cp_size) % cp_size
    remainder = seq_length % cp_size
    pad_length = (cp_size - remainder) % cp_size

    if pad_length > 0:
        packed_sequence["tokens"] = F.pad(packed_sequence["tokens"], (0, pad_length), value=0)
        packed_sequence["position_ids"] = F.pad(packed_sequence["position_ids"], (0, pad_length), value=0)
        packed_sequence["loss_masks"] = F.pad(packed_sequence["loss_masks"], (0, pad_length), value=0)
        packed_sequence["cu_seqlens"][-1] += pad_length
    return packed_sequence
```

**关键设计**：
1. **最小 Padding**: 最多 `cp_size - 1` 个 token，开销可控
2. **直接 Chunk**: 使用 `torch.chunk` 简单切分，负载均衡交给 Ring Flash Attention
3. **cu_seqlens 共享**: 所有 CP rank 共享全局 `cu_seqlens`，由 Ring Flash Attention 处理

#### 2.4.2 Ring Flash Attention 集成

**Device Mesh 中的 CP 初始化** (`_setup_device_mesh` 方法):

```python
# Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
if self.cp_size > 1:
    substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
    logger.info(f"[Rank {rank}] CP initialized via device mesh")
else:
    logger.info(f"[Rank {rank}] Pure DP mode (cp_size=1)")
```

**工作原理**：
- `substitute_hf_flash_attn` 替换 HuggingFace 模型的 attention 实现
- 使用 [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) 库
- 自动处理 KV cache 的 ring 通信

**相关 PR**:
- **#467**: 引入 Context Parallel 支持
- **#866**: 修复 CP 模式下 `max_tokens_per_gpu` 的 bug

---

### 2.5 True On-Policy 实现

#### 2.5.1 Batch Invariant Mode

**代码位置**: `actor.py:145-163`

```python
def _enable_true_on_policy_optimizations(self, args):
    if args.true_on_policy_mode:
        from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
        from .models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

        logger.info("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
        enable_batch_invariant_mode(
            # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
            # and disabling it will make it aligned
            enable_bmm=False,  # 关键：禁用 bmm 以消除 batch size 影响
        )

        apply_true_on_policy_patch_for_qwen3_moe()
    else:
        from .models.qwen3_moe_hf import apply_fsdp_moe_patch
        apply_fsdp_moe_patch()
```

**Batch Invariant 原理**：
- SGLang 提供的特性，确保不同 batch size 下算子结果一致
- 关键是 RoPE 实现中的 `bmm` 操作，不同 batch size 可能导致数值差异
- 通过 `enable_bmm=False` 强制使用确定性实现

#### 2.5.2 禁用 Torch Compile

**代码位置**: `actor.py:360, 574`

```python
# In _compute_log_prob and _train_step:
log_probs_result, entropy_result = get_logprob_and_entropy_with_cp(
    logits=logits,
    target_tokens=batch["tokens"],
    cp_rank=self.cp_rank,
    cp_size=self.cp_size,
    cp_group=self.cp_group,
    model_input_ids=model_args["input_ids"],
    allow_compile=not self.args.true_on_policy_mode,  # 关键：禁用编译
    temperature=self.args.rollout_temperature,        # 关键：复用 rollout temperature
)
```

**原因**：
- `torch.compile` 可能引入优化导致数值偏差
- `selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)`
- True on-policy 模式下使用未编译版本 `selective_log_softmax_raw`

#### 2.5.3 Mixed Precision Policy

**代码位置**: `actor.py:1016-1057`

```python
def apply_fsdp2(model, mesh=None, cpu_offload=False):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # 参数存储为 bf16
            reduce_dtype=torch.float32,   # 梯度聚合为 fp32
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # ... (apply FSDP)
```

**相关 PR**:
- **#911**: 从 autocast 迁移到 Mixed Precision Policy
- **#833**: 修复 autocast 导致的精度问题
- **旧实现问题**: autocast 作用域不当导致部分计算被错误地转换为 bf16
- **新实现优势**: FSDP2 原生 Mixed Precision Policy 更清晰可控

#### 2.5.4 CI 测试验证

**代码位置**: `actor.py:503-509`

```python
if self.args.ci_test and self.args.true_on_policy_mode:
    assert log_dict["rollout/log_probs"] == log_dict["rollout/rollout_log_probs"], (
        f"CI check failed: true_on_policy_mode is enabled, but log_probs "
        f"({log_dict['rollout/log_probs']}) != rollout_log_probs "
        f"({log_dict['rollout/rollout_log_probs']})"
    )
```

**验证逻辑**：
- CI 模式下强制检查 `log_probs == rollout_log_probs`
- 确保训练端和推理端的 log probs 完全一致

**相关 PR**:
- **#906, #917, #934, #1001**: 多个 PR 逐步完善 True On-Policy 支持
- **核心实现**: PR #906, #917, #934 (3-part series)
- **最终优化**: PR #1001 (重构 fused experts kernel)

---

## 3. 关键 PR 实现细节

### 3.1 PR #467: Context Parallel 支持

**提交信息**: `[FSDP] Support Context parallelism for FSDP using ring-flash-attn`

**变更文件**：
- `slime/backends/fsdp_utils/actor.py` (302 insertions, 67 deletions)
- `slime/backends/fsdp_utils/arguments.py` (3 insertions)
- `slime/backends/fsdp_utils/data_packing.py` (33 insertions, 1 deletion)

**核心实现**：

1. **Device Mesh 构建** (新增 `_setup_device_mesh` 方法):
   ```python
   self.mesh = init_device_mesh(
       "cuda",
       mesh_shape=(self.dp_size, self.cp_size),
       mesh_dim_names=("dp", "cp")
   )
   ```

2. **Ring Flash Attention 集成**:
   ```python
   if self.cp_size > 1:
       substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
   ```

3. **输入切分与 Padding**:
   ```python
   def pad_packed_sequence_with_cp(packed_sequence: dict, cp_size: int) -> dict:
       seq_length = len(packed_sequence["tokens"])
       remainder = seq_length % cp_size
       pad_length = (cp_size - remainder) % cp_size

       if pad_length > 0:
           packed_sequence["tokens"] = F.pad(packed_sequence["tokens"], (0, pad_length), value=0)
           # ... (pad other fields)
       return packed_sequence
   ```

4. **Log Prob 聚合** (重构 `get_logprob_and_entropy_with_cp` 函数):
   ```python
   # Merge with a single all_gather: stack as [2, chunk_size]
   stacked_local = torch.stack([local_log_probs, entropy], dim=0)
   gathered_stacked = torch.distributed.nn.functional.all_gather(stacked_local, group=cp_group)
   ```

**优化效果**：
- 支持 8k -> 16k sequence length (CP=2)
- 单次 all_gather 减少通信开销

---

### 3.2 PR #780: Fix PPO KL

**提交信息**: `[FSDP]Fix ppo_kl (#780)`

**变更文件**：
- `slime/backends/fsdp_utils/actor.py` (53 insertions, 90 deletions)

**核心问题**：
- 使用 Ref Model 时，on-policy KL 应为 0，但实际观察到微小正值
- 原因：手动交换 Ref 和 Actor 权重时引入数值偏差

**旧实现** (手动权重交换):
```python
# 旧代码（已移除）
def _swap_ref_actor_weights(self):
    """Manually swap weights between ref and actor models."""
    for ref_param, actor_param in zip(self.ref_model.parameters(), self.model.parameters()):
        # Create DTensor manually for FSDP2
        ref_data = ref_param.data
        actor_data = actor_param.data

        # Swap
        ref_param.data = actor_data
        actor_param.data = ref_data
```

**问题分析**：
- 手动创建 DTensor 时可能引入数值偏差
- CPU/GPU 转移过程中的精度损失
- FSDP2 内部状态不一致

**新实现** (独立 FSDP2 实例):
```python
def _create_ref_model(self, ref_load_path: str | None):
    """Create and initialize a separate reference model with FSDP2 CPUOffloadPolicy."""

    init_context = self._get_init_weight_context_manager()

    with init_context():
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_load_path,
            trust_remote_code=True,
            attn_implementation=self.args.attn_implementation,
        )

    full_state = ref_model.state_dict()

    # Always use CPUOffloadPolicy for reference
    ref_model = apply_fsdp2(ref_model, mesh=self.dp_mesh, cpu_offload=True)
    ref_model = self._fsdp2_load_full_state_dict(ref_model, full_state, self.dp_mesh, cpu_offload=True)

    return ref_model
```

**优化效果**：
- On-policy KL 精确收敛到 0
- 避免手动权重交换的复杂性
- 利用 FSDP2 原生 CPU Offload 机制

---

### 3.3 PR #911: Mixed Precision Policy

**提交信息**: `[FSDP] convert from autocast to mixed_policy (#911)`

**变更文件**：
- `slime/backends/fsdp_utils/actor.py`

**核心问题**：
- 使用 `torch.cuda.amp.autocast` 时，部分计算被错误地转换为 bf16
- 导致 True On-Policy 模式下 log probs 不一致

**旧实现** (autocast):
```python
# 旧代码（已移除）
def _compute_log_prob(self, model_tag, packed_batches, store_prefix=""):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in packed_batches:
            model_args = self._get_model_inputs_args(batch)
            logits = active_model(**model_args).logits.squeeze(0).float()
            # ... (compute log probs)
```

**问题分析**：
- `autocast` 作用域不明确，可能影响 log_softmax 计算
- 不同 batch size 下 autocast 行为可能不同

**新实现** (FSDP2 Mixed Precision):
```python
def apply_fsdp2(model, mesh=None, cpu_offload=False):
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # 参数存储为 bf16
            reduce_dtype=torch.float32,   # 梯度聚合为 fp32
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
```

**优化效果**：
- 精度控制更清晰明确
- 避免 autocast 的隐式类型转换
- FSDP2 原生支持，性能更好

---

### 3.4 PR #915: Rank-0 Broadcast 优化

**提交信息**: `[FSDP] Optimize FSDP2 Model Loading with Rank-0 Broadcast (#915)`

**变更文件**：
- `slime/backends/fsdp_utils/actor.py`

**核心优化**：
- 只有 rank 0 从磁盘加载完整模型
- 其他 rank 使用 meta device 或 empty tensors
- 通过 `broadcast_from_rank0=True` 广播权重

**实现** (`_fsdp2_load_full_state_dict` 方法):
```python
def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # Rank 0: move with weights, others: allocate empty tensors on device
    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        # to_empty creates tensors on device without initializing memory
        model = model.to_empty(device=torch.cuda.current_device())

    is_cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=is_cpu_offload,
        broadcast_from_rank0=True  # 关键参数
    )

    set_model_state_dict(model, full_state, options=options)

    # Manually broadcast buffers (not handled by set_model_state_dict)
    for _name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if is_cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(torch.cuda.current_device())

    return model
```

**优化效果**：
- 加载时间从 O(world_size) 降低到 O(1)
- 对于大模型（如 Qwen3-30B）显著减少初始化时间

---

### 3.5 PR #1001: True On-Policy 最终优化

**提交信息**: `[FSDP][3/N] support true_on_policy training for FSDP2 (#1001)`

**变更文件**：
- `slime/backends/fsdp_utils/kernels/__init__.py` (新增)
- `slime/backends/fsdp_utils/kernels/fused_experts.py` (218 insertions)
- `slime/backends/fsdp_utils/models/qwen3_moe.py` (244 insertions, 219 deletions)

**核心优化**：
- 重构 Fused Experts kernel，提取到独立文件
- 确保 MoE 模型的 True On-Policy 支持

**Fused Experts Kernel** (`kernels/fused_experts.py`):
```python
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    override_config: dict | None = None,
    use_fp8_w8a8: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused MoE kernel for Qwen3-MoE.

    Optimized kernel that fuses expert computation for better performance
    while maintaining numerical consistency for true on-policy training.
    """
    # ... (implementation)
```

**相关 PR 系列**：
- **#917**: [FSDP][1/N] support true_on_policy training for FSDP2
- **#906**: [FSDP][2/N] support true_on_policy training for FSDP2
- **#934**: [FSDP][3/N] support true on policy training for FSDP2 (duplicate)
- **#1001**: [FSDP][3/N] support true_on_policy training for FSDP2 (final)

**优化效果**：
- MoE 模型支持 True On-Policy 模式
- 代码结构更清晰（kernel 独立）

---

## 4. 性能优化机制

### 4.1 显存优化

#### 4.1.1 CPU Offload

**FSDP2 原生支持**：
```python
offload_policy = CPUOffloadPolicy() if cpu_offload else None

fsdp_kwargs = {
    "mp_policy": MixedPrecisionPolicy(...),
    "offload_policy": offload_policy,  # 关键
    "mesh": mesh,
}
```

**三种 Offload 场景**：

1. **Train Offload** (Colocated 模式):
   ```python
   def sleep(self) -> None:
       """Pause CUDA memory for all tracked tensors."""
       if not self.args.offload_train:
           return

       self.model.cpu()
       move_torch_optimizer(self.optimizer, "cpu")
       clear_memory()

   def wake_up(self) -> None:
       """Resume CUDA memory for all tracked tensors."""
       if not self.args.offload_train:
           return

       self.model.cuda()
       move_torch_optimizer(self.optimizer, "cuda")
   ```

2. **Ref Model Offload** (始终开启):
   ```python
   # Ref model 创建时强制 cpu_offload=True
   ref_model = apply_fsdp2(ref_model, mesh=self.dp_mesh, cpu_offload=True)
   ```

3. **Optimizer State Offload** (`fsdp_cpu_offload=True`):
   - 参数、梯度、优化器状态全部 offload 到 CPU
   - Optimizer step 在 CPU 上执行
   - 显著节省 GPU 内存，但训练速度下降

**相关 PR**:
- **#847**: 从 DeepSpeed 迁移到 FSDP2 原生 CPU Offload

#### 4.1.2 Mixed Precision

```python
"mp_policy": MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,  # 参数存储为 bf16，节省 50% 内存
    reduce_dtype=torch.float32,   # 梯度聚合为 fp32，保证精度
)
```

**效果**：
- 参数内存减半
- 梯度聚合精度不受影响
- 训练速度提升 20-30%

#### 4.1.3 Gradient Checkpointing

```python
if args.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

**效果**：
- 以计算换内存
- 适用于超长序列或超大模型

---

### 4.2 通信优化

#### 4.2.1 分桶异步权重更新

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
        if isinstance(param, DTensor):
            param = param.redistribute(
                placements=[Replicate()] * param.device_mesh.ndim,
                async_op=True,  # 异步 replicate
            ).to_local()

        bucket.append((name, param))
        bucket_size += param_size
```

**优化效果**：
- 避免峰值内存占用
- 异步 replicate 重叠计算和通信

#### 4.2.2 单次 All-Gather (CP 模式)

```python
# Merge with a single all_gather: stack as [2, chunk_size]
stacked_local = torch.stack([local_log_probs, entropy], dim=0)
gathered_stacked = torch.distributed.nn.functional.all_gather(stacked_local, group=cp_group)
```

**优化效果**：
- 原本需要 2 次 all_gather，现在只需 1 次
- 减少通信延迟和带宽占用

---

### 4.3 计算优化

#### 4.3.1 Data Packing

```python
partitions = get_seqlen_balanced_partitions(
    seq_lengths,
    k_partitions=k_partitions,
    equal_size=False
)
```

**Karmarkar-Karp 算法**：
- 最大差分法，负载均衡分配
- 确保每个 pack 的 token 数高度均衡
- 避免传统 padding 的算力浪费

**效果**：
- 消除 padding 开销
- 提升 GPU 利用率

#### 4.3.2 Torch Compile (非 True On-Policy)

```python
selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)

def gather_log_probs_packed(shifted_logits, input_ids, allow_compile, ...):
    selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    return selective_log_softmax(shifted_logits, targets)
```

**效果**：
- 非 True On-Policy 模式下可以使用编译优化
- 提升 log_softmax + gather 性能

---

## 5. 与 Megatron 的对比

### 5.1 代码复杂度

| 维度 | FSDP | Megatron |
|------|------|----------|
| **模型加载** | `AutoModelForCausalLM.from_pretrained()` | 手动配置架构参数或权重转换 |
| **权重格式** | HuggingFace 原生支持 | 需要转换为 `torch_dist` 格式 |
| **并行配置** | DP + CP (TP/EP/PP coming soon) | DP + TP + PP + CP + EP (全支持) |
| **架构适配** | 自动推断 (AutoConfig) | 手动指定 (大量参数) |
| **新模型支持** | 开箱即用 | 需要适配 Megatron Core |

### 5.2 功能对比

| 功能 | FSDP | Megatron |
|------|------|----------|
| **Tensor Parallel** | Coming Soon | ✅ |
| **Pipeline Parallel** | Coming Soon | ✅ |
| **Expert Parallel** | Coming Soon | ✅ |
| **Context Parallel** | ✅ (via Ring Flash Attention) | ✅ (原生实现) |
| **CPU Offload** | ✅ (原生 CPUOffloadPolicy) | ✅ (通过分布式优化器) |
| **Gradient Checkpointing** | ✅ (简单开关) | ✅ (多种粒度) |
| **Mixed Precision** | ✅ (MixedPrecisionPolicy) | ✅ (原生支持) |
| **True On-Policy** | ✅ | ✅ |
| **VLM 支持** | ✅ (HF 生态) | 部分支持 |

### 5.3 性能对比

**PR #788 验证结果**：
- **实验配置**: 单机 H100，Qwen3-4B，sglang 0.5.5post1
- **结果**: FSDP 和 Megatron 训练精度对齐，收敛效果相近

**Context Parallel 验证**：
- **实验配置**: 4 张 B200，global_batch_size = 64

| 配置 | response_length = 8k | response_length = 16k |
|------|---------------------|---------------------|
| FSDP, cp = 1 | work | OOM |
| FSDP, cp = 2 | work | work |
| Megatron(TP = 1), cp = 1 | work | OOM |
| Megatron(TP = 1), cp = 2 | work | work |

**结论**:
- FSDP 和 Megatron 性能相当
- FSDP 在易用性和灵活性上更优
- Megatron 在并行能力上更强（支持 TP/EP/PP）

### 5.4 使用场景建议

**选择 FSDP**：
- 快速原型和实验
- 新模型架构 (Qwen3-Next, GPT-OSS)
- VLM RL 训练
- 不需要 TP/EP/PP

**选择 Megatron**：
- 超大规模训练 (需要 TP/EP/PP)
- 对性能有极致要求
- 模型已有 Megatron 适配

---

## 6. 未来计划

### 6.1 待实现功能

1. **Tensor Parallel (TP)**:
   - 维持代码整洁的同时实现 TP
   - 参考 FSDP2 + DTensor 的实现方案

2. **Expert Parallel (EP)**:
   - MoE 模型的专家并行
   - 与 TP 协同工作

3. **VLM 联合训练**:
   - Vision + Language 联合训练
   - 部分冻结策略

4. **混合模型支持**:
   - Qwen3-Next
   - GPT-OSS
   - 其他创新架构

### 6.2 优化方向

1. **学习率调度**:
   - 目前只支持 constant
   - 计划支持 linear/cosine decay 和 warmup

2. **Gradient Checkpointing 粒度**:
   - 目前只是简单开关
   - 计划支持 selective/uniform/block 等多种策略

3. **性能优化**:
   - 进一步优化 CP 通信
   - 优化 Data Packing 算法
   - 探索 Fully Sharded Data Parallel + Tensor Parallel 混合并行

---

## 7. 总结

### 7.1 核心设计原则

1. **接口标准化**:
   - 对外暴露统一接口 (`init`, `train`, `save`, `update_weights`, `sleep`, `wake_up`)
   - 内部使用下划线约定私有方法

2. **物理隔离**:
   - 利用 Ray Actor 机制将 FSDP 和 Megatron 封装在独立进程空间
   - 避免全局变量冲突

3. **PyTorch 原生优先**:
   - 尽量使用 FSDP2 原生功能 (MixedPrecisionPolicy, CPUOffloadPolicy, DTensor)
   - 避免手动实现复杂的分布式逻辑

4. **性能与可读性平衡**:
   - 保持代码整洁的同时追求性能优化
   - 通过注释和文档解释关键设计决策

### 7.2 关键技术亮点

1. **Rank-0 Broadcast 优化** (PR #915):
   - 只有 rank 0 加载完整模型，其他 rank 使用 empty tensors
   - 大模型初始化时间从 O(world_size) 降低到 O(1)

2. **独立 Ref Model** (PR #780):
   - 避免手动权重交换引入的数值偏差
   - 利用 FSDP2 原生 CPU Offload，on-policy KL 精确为 0

3. **Mixed Precision Policy** (PR #911):
   - 从 autocast 迁移到 FSDP2 原生 MixedPrecisionPolicy
   - 避免 autocast 的隐式类型转换，精度控制更清晰

4. **Context Parallel** (PR #467):
   - 通过 Ring Flash Attention 实现 CP
   - 单次 all_gather 优化通信效率

5. **True On-Policy** (PR #906, #917, #934, #1001):
   - Batch Invariant Mode + 禁用 Torch Compile + Mixed Precision Policy
   - 实现训练端和推理端 log probs 完全一致

6. **Data Packing** (PR #321):
   - Karmarkar-Karp 算法实现负载均衡
   - 消除传统 padding 的算力浪费

### 7.3 代码质量

- **总代码量**: `actor.py` 1058 行，结构清晰
- **注释覆盖**: 关键函数都有详细 docstring
- **模块化**: 数据打包、权重更新、checkpoint 管理独立成模块
- **可扩展性**: 易于添加新的优化器、算法、模型适配

---

## 附录

### A. 关键 PR 时间线

| 日期 | PR | 描述 |
|------|----|----|
| 2025-11-16 | #467 | Context Parallel 支持 |
| 2025-11-19 | #780 | 修复 PPO KL (独立 Ref Model) |
| 2025-11-20 | #788 | FSDP/Megatron 对齐验证 |
| 2025-11-21 | #833 | 修复 autocast 导致的 True On-Policy 失效 |
| 2025-11-23 | #911 | 迁移到 Mixed Precision Policy |
| 2025-11-25 | #915 | Rank-0 Broadcast 优化 |
| 2025-11-26 | #917 | True On-Policy [1/N] |
| 2025-11-27 | #906 | True On-Policy [2/N] |
| 2025-11-28 | #934 | True On-Policy [3/N] (duplicate) |
| 2025-12-02 | #1001 | True On-Policy [3/N] (final) |
| 2025-12-02 | #1010 | Qwen3-30B 支持 |

### B. 主要贡献者

- **Zilin Zhu** (@zhuzilin): True On-Policy, MoE 优化
- **Huapeng Zhou** (@PopSoda2002): Context Parallel, 权重更新优化
- **Hecate** (@Hecate0821): PPO KL 修复, CP bug 修复
- **Zhuohao Li** (@Zhuohao-Li): 各种 bug 修复和优化

### C. 参考资料

- [FSDP Training Backend Deep Dive](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-2-en.md)
- [Weight Update Mechanisms](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-1-EN.md)
- [Ring Flash Attention](https://github.com/zhuzilin/ring-flash-attention)
- [PyTorch FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html)
- [verl FSDP Utils](https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py)

---

**文档生成日期**: 2025-12-03
**Slime 版本**: main branch (commit 9d7f34d)
**作者**: AI 代码分析系统
