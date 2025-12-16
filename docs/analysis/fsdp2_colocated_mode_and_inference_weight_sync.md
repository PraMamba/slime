# FSDP2 Colocated 模式：训练推理切换与权重同步深度分析

## 概述

本文档深入分析 Slime 框架中 FSDP2 在 Colocated 模式下的训练推理切换机制，重点回答：

1. **FSDP 模型训练后的状态**：Sharded 还是 Unsharded？
2. **权重同步机制**：如何从 FSDP 训练模型同步到 SGLang 推理引擎？
3. **推理引擎的权重管理**：SGLang 如何处理接收到的权重（TP 切分）？
4. **KV Cache 管理**：在 TP/CP 结合时如何适配？

本分析基于 Slime 框架源码（commit: 9d7f34d）、PyTorch FSDP2 文档和 SGLang 官方文档。

---

## 问题 1：FSDP 模型训练后的状态

### 核心结论

**FSDP 模型在训练完成后仍然保持 Sharded 状态（DTensor）**，不会自动转换为完整模型。

### 详细分析

#### 1.1 FSDP2 的参数表示

FSDP2 使用 **DTensor（Distributed Tensor）** 来表示分片参数：

```python
# FSDP2 参数的实际结构
for name, param in model.named_parameters():
    print(f"{name}: {type(param)}")
    # 输出：xxx: <class 'torch.distributed.tensor.DTensor'>

    print(f"Placement: {param.placements}")
    # 输出：[Shard(0)]（在第 0 维度分片）

    print(f"Device Mesh: {param.device_mesh}")
    # 输出：DeviceMesh('cuda', mesh=[0, 1, 2, 3])（4 GPU DP）

    print(f"Local Shape: {param.to_local().shape}")
    # 输出：(hidden_size / 4, ...)（每个 GPU 存储 1/4）
```

**关键点**：
- **DTensor 是分片的**：每个 GPU 只存储 1/N 的参数数据
- **Sharding Spec**：`[Shard(0)]` 表示在第 0 维度（通常是行或列）分片
- **Device Mesh**：记录参数分布在哪些 GPU 上

#### 1.2 训练完成后的状态

**训练循环结束后**（`train()` 方法返回时）：

```python
# slime/backends/fsdp_utils/actor.py:447-467
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    """Run one training update over a rollout batch."""

    # 1. Wake up（如果启用 offload_train）
    if self.args.offload_train:
        self.wake_up()  # 从 CPU 加载回 GPU

    # 2. 训练核心逻辑
    with inverse_timer("train_wait"), timer("train"):
        rollout_data = process_rollout_data(...)
        self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

    # 3. 训练结束
    # ⚠️ 注意：这里不调用 sleep()
    # 模型参数保持在 GPU 上，仍然是 Sharded 状态（DTensor）
```

**状态验证**：

```python
# 训练结束后，检查参数状态
for param in model.parameters():
    assert isinstance(param, DTensor), "参数仍然是 DTensor"
    assert param.placements == [Shard(0)], "参数仍然是分片的"

    # 每个 GPU 只存储部分数据
    local_data = param.to_local()  # 获取本地分片
    print(f"Local data shape: {local_data.shape}")
    # 输出：(param_size / world_size, ...)
```

**结论**：
- ✅ **训练后模型仍是 Sharded 状态**
- ✅ **参数仍然是 DTensor**
- ✅ **每个 GPU 只存储 1/N 的参数**
- ❌ **不会自动转换为完整模型**

#### 1.3 Colocated 模式的特殊处理

在 Colocated 模式下（`--colocate` 参数），训练和推理共享同一组 GPU：

```python
# slime/backends/fsdp_utils/actor.py:276-287
def sleep(self) -> None:
    """Pause CUDA memory for all tracked tensors."""
    if not self.args.offload_train:
        return

    print_memory("before offload model")

    # 1. 将整个 FSDP 模型 offload 到 CPU
    self.model.cpu()  # DTensor 会自动处理，将本地分片移到 CPU

    # 2. 将 Optimizer States offload 到 CPU
    move_torch_optimizer(self.optimizer, "cpu")

    # 3. 清理 GPU 显存
    clear_memory()
    dist.barrier(group=get_gloo_group())

    print_memory("after offload model")
```

**关键点**：
- **Offload 到 CPU**：整个 FSDP 模型（包括 Sharded 参数）被移到 CPU
- **仍然是 Sharded**：Offload 不改变参数的分片状态，只改变设备位置
- **释放 GPU 显存**：为 SGLang 推理引擎腾出显存

**Offload 后的状态**：

```python
# Offload 后，参数在 CPU 上，但仍然是 DTensor（Sharded）
for param in model.parameters():
    assert isinstance(param, DTensor), "仍然是 DTensor"
    assert param.placements == [Shard(0)], "仍然是分片的"
    assert param.to_local().device.type == "cpu", "数据在 CPU 上"
```

---

## 问题 2：权重同步机制

### 核心流程

**权重同步（Weight Sync）的完整流程**：

```
[FSDP 训练模型]                [SGLang 推理引擎]
    Sharded (DTensor)               TP Sharded (不同维度)
         ↓                                 ↑
    1. Redistribute                        |
    (Shard → Replicate)                    |
         ↓                                 |
    Complete Tensors                       |
         ↓                                 |
    2. Bucket & Flatten                    |
         ↓                                 |
    3. Gather to Rank 0                    |
    (each TP group)                        |
         ↓                                 |
    4. RPC to SGLang ----------------------+
         ↓
    SGLang receives complete weights
         ↓
    5. TP Sharding (SGLang internal)
```

### 详细分析

#### 2.1 权重收集（Redistribute）

**核心代码**：`slime/backends/fsdp_utils/update_weight_utils.py:45-70`

```python
class UpdateWeight(abc.ABC):
    def update_weights(self) -> None:
        bucket = []
        bucket_size = 0

        # 遍历所有参数
        for name, param in self.model.state_dict().items():
            param_size = param.numel() * param.element_size()

            # 分桶策略：避免一次性传输所有参数，导致显存峰值过高
            if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                del bucket
                bucket = []
                bucket_size = 0

            # 将参数移到 GPU（如果在 CPU）
            param = param.cuda()

            # ⚠️ 关键：将 DTensor 转换为完整张量
            if isinstance(param, DTensor):
                # redistribute: Shard → Replicate
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,  # 所有维度都 Replicate
                    async_op=True,  # 异步操作，不阻塞
                ).to_local()  # 转换为普通 Tensor

            bucket.append((name, param))
            bucket_size += param_size

        # 处理最后一个 bucket
        if bucket:
            self.wait_and_update_bucket_weights(bucket)
```

**Redistribute 的工作原理**：

```python
# 示例：4 GPU DP，参数形状 (1024, 1024)

# 1. 训练时的 Sharded 状态
dtensor = DTensor(
    local_tensor=torch.randn(256, 1024),  # 每个 GPU 存储 1/4 行
    device_mesh=DeviceMesh('cuda', [0, 1, 2, 3]),
    placements=[Shard(0)]  # 在第 0 维度分片
)

# 2. Redistribute 为 Replicate（All-Gather）
replicated_dtensor = dtensor.redistribute(
    placements=[Replicate()],  # 每个 GPU 都有完整数据
    async_op=True
)

# 3. 转换为普通 Tensor
full_tensor = replicated_dtensor.to_local()
print(full_tensor.shape)  # (1024, 1024)（完整参数）

# 4. 每个 GPU 现在都有完整的参数副本
```

**通信分析**：
- **操作**：All-Gather（收集分片数据）
- **通信量**：每个参数的完整大小（如 1024 × 1024 × 4 bytes = 4 MB）
- **目标**：每个 GPU 获得完整参数

#### 2.2 分桶与扁平化（Bucketing & Flattening）

**核心代码**：`slime/backends/fsdp_utils/update_weight_utils.py:117-138`

```python
def update_bucket_weights(self, named_tensors) -> None:
    """使用扁平化 Bucket 传输权重（类似 Megatron 的策略）"""

    # 1. 按 dtype 分组（减少通信次数）
    named_tensors_by_dtypes = {}
    for name, tensor in named_tensors:
        dtype = tensor.dtype
        if dtype not in named_tensors_by_dtypes:
            named_tensors_by_dtypes[dtype] = []
        named_tensors_by_dtypes[dtype].append((name, tensor))

    # 2. 为每个 dtype 创建扁平化 Bucket
    serialized_tensors = []
    for _dtype, named_tensors in named_tensors_by_dtypes.items():
        # 使用 SGLang 的 FlattenedTensorBucket
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)

        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,  # 包含每个参数的名称、shape、offset
        }

        # 序列化（用于跨进程传输）
        serialized_tensors.append(
            MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
        )
```

**扁平化 Bucket 的结构**：

```python
# 示例：3 个参数，相同 dtype
params = [
    ("layer1.weight", torch.randn(1024, 512)),  # 512K elements
    ("layer1.bias", torch.randn(1024)),         # 1K elements
    ("layer2.weight", torch.randn(512, 256)),   # 128K elements
]

# 扁平化为单个连续 Tensor
flattened_tensor = torch.cat([p.flatten() for _, p in params])
# Shape: (641K,)（所有参数拼接）

# Metadata 记录每个参数的位置
metadata = {
    "layer1.weight": {"shape": (1024, 512), "offset": 0, "numel": 524288},
    "layer1.bias": {"shape": (1024,), "offset": 524288, "numel": 1024},
    "layer2.weight": {"shape": (512, 256), "offset": 525312, "numel": 131072},
}
```

**优势**：
- ✅ **减少通信次数**：一次传输多个参数，而非逐个传输
- ✅ **降低延迟**：连续内存访问更高效
- ✅ **节省显存**：可以逐 bucket 传输，避免同时加载所有参数

#### 2.3 TP 组内 Gather（Gather to Rank 0）

在 Colocated 模式下，训练和推理的 GPU 分配：

```
示例：8 GPU，Colocated 模式
  - 训练：FSDP DP=8（每个 GPU 独立训练）
  - 推理：2 个 SGLang Engine，每个 TP=4

训练 GPU 分配：
  GPU 0, 1, 2, 3, 4, 5, 6, 7  (FSDP DP ranks)

推理 GPU 分配：
  Engine 0: GPU 0, 1, 2, 3  (TP ranks 0, 1, 2, 3)
  Engine 1: GPU 4, 5, 6, 7  (TP ranks 0, 1, 2, 3)
```

**Gather 策略**：

```python
# slime/backends/fsdp_utils/update_weight_utils.py:114-116
# 每个训练 rank 计算自己对应的 TP rank
self.tp_rank = dist.get_rank() - start_rank

# 示例：
# GPU 0: tp_rank = 0 - 0 = 0（Engine 0 的 rank 0）
# GPU 1: tp_rank = 1 - 0 = 1（Engine 0 的 rank 1）
# GPU 4: tp_rank = 4 - 4 = 0（Engine 1 的 rank 0）
```

**Gather 到 TP Group 的 Rank 0**：

```python
# slime/backends/fsdp_utils/update_weight_utils.py:140-152
if self._ipc_gather_src == dist.get_rank():
    # 只有 TP group 的 rank 0 准备接收
    gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
else:
    gathered_serialized_batches = None

# 在 TP group 内 Gather（使用 Gloo backend）
dist.gather_object(
    obj=serialized_tensors,  # 当前 rank 的序列化参数
    object_gather_list=gathered_serialized_batches,  # 收集到 rank 0
    dst=self._ipc_gather_src,  # TP group 的 rank 0
    group=self._ipc_gather_group,  # TP group
)
```

**为什么需要 Gather？**

因为每个 TP rank 可能需要不同的参数切片，但在 FSDP 训练时，参数是按 DP 分片的。需要先收集完整参数，然后由 SGLang 按 TP 维度重新切分。

#### 2.4 RPC 传输到 SGLang

**核心代码**：`slime/backends/fsdp_utils/update_weight_utils.py:154-171`

```python
if dist.get_rank() == self._ipc_gather_src:
    # 只有 TP group 的 rank 0 执行 RPC

    # 遍历每个 dtype bucket
    num_dtypes = len(gathered_serialized_batches[0])
    for i in range(num_dtypes):
        kwargs = {
            "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
            "load_format": "flattened_bucket",  # 使用扁平化格式
            "flush_cache": False,  # 先不刷新缓存（等所有 bucket 传输完）
        }

        # ⚠️ 关键：调用 SGLang Engine 的 update_weights_from_tensor 方法
        ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
        ray.get(ref)  # 等待完成

    # 所有 bucket 传输完成后，刷新缓存
    ref = self._ipc_engine.flush_cache.remote()
    ray.get(ref)
```

**传输协议**：
- **框架**：Ray RPC（Remote Procedure Call）
- **数据格式**：序列化的扁平化 Bucket
- **传输方向**：训练 Rank 0 → SGLang Engine（每个 TP group）

---

## 问题 3：SGLang 推理引擎的权重管理

### 核心机制

**SGLang 接收权重后的处理流程**：

```
[接收完整权重]
    ↓
[反序列化 Bucket]
    ↓
[解析 Metadata]
    ↓
[按名称匹配参数]
    ↓
[TP Sharding（如果启用 TP）]
    ↓
[加载到模型]
```

### 详细分析

#### 3.1 SGLang 的 TP 支持

**TP 配置**：

在 Slime 中，SGLang 的 TP size 由 `--rollout-num-gpus-per-engine` 决定：

```python
# slime/backends/sglang_utils/arguments.py:121
args.sglang_tp_size = args.rollout_num_gpus_per_engine

# 示例：
# --rollout-num-gpus-per-engine=4
# → SGLang TP size = 4
```

**SGLang 的 TP 实现**：

SGLang 使用 vLLM 的 TP 框架（基于 Megatron-LM）：

```python
# SGLang 内部（伪代码）
class TPLinear(nn.Module):
    """TP 分片的线性层"""
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # 列并行：将输出维度切分
        self.out_features_per_partition = out_features // tp_size

        # 每个 TP rank 只存储部分权重
        self.weight = nn.Parameter(
            torch.empty(in_features, self.out_features_per_partition)
        )

    def load_weight(self, full_weight):
        """从完整权重中提取对应的 TP 分片"""
        # 计算当前 TP rank 应该负责的列范围
        start = self.tp_rank * self.out_features_per_partition
        end = (self.tp_rank + 1) * self.out_features_per_partition

        # 提取对应的分片
        shard = full_weight[:, start:end]

        # 加载到模型
        self.weight.data.copy_(shard)
```

**关键点**：
- **接收完整权重**：SGLang 从训练侧接收完整的权重张量
- **自动 TP 切分**：SGLang 内部根据 TP rank 自动提取对应的切片
- **不同的切分维度**：FSDP DP 按 batch 切分，SGLang TP 按模型维度切分

#### 3.2 权重加载流程

**SGLang 的 `update_weights_from_tensor` 方法**（简化逻辑）：

```python
class SGLangEngine:
    def update_weights_from_tensor(self, serialized_named_tensors, load_format):
        """从序列化的 Bucket 加载权重"""

        # 1. 反序列化每个 TP rank 的数据
        all_named_tensors = []
        for serialized in serialized_named_tensors:
            data = MultiprocessingSerializer.deserialize(serialized)

            # 解包扁平化 Bucket
            flattened_tensor = data["flattened_tensor"]
            metadata = data["metadata"]

            # 根据 metadata 恢复每个参数
            named_tensors = {}
            for name, info in metadata.items():
                offset = info["offset"]
                numel = info["numel"]
                shape = info["shape"]

                # 从扁平化 Tensor 中提取
                param_data = flattened_tensor[offset:offset+numel].view(shape)
                named_tensors[name] = param_data

            all_named_tensors.append(named_tensors)

        # 2. 合并所有 TP rank 的数据（如果需要）
        # 在 Slime 的实现中，每个 TP rank 都接收完整权重
        # 所以 all_named_tensors[0] 就是完整权重
        merged_tensors = all_named_tensors[0]

        # 3. 加载到模型（自动处理 TP 切分）
        for name, param in self.model.named_parameters():
            if name in merged_tensors:
                full_weight = merged_tensors[name]

                # ⚠️ 关键：模型的 load_weight 方法会自动提取 TP 分片
                param.load_weight(full_weight)
```

**TP 切分示例**：

```python
# 假设：Linear layer (4096, 4096)，TP size = 4

# 训练侧（FSDP）：
# 完整权重传输：(4096, 4096)

# SGLang 接收后，按 TP 切分：
# TP rank 0: weight[:, 0:1024]     (4096, 1024)
# TP rank 1: weight[:, 1024:2048]  (4096, 1024)
# TP rank 2: weight[:, 2048:3072]  (4096, 1024)
# TP rank 3: weight[:, 3072:4096]  (4096, 1024)

# 每个 TP rank 只存储 1/4 的权重
```

#### 3.3 Colocated 模式的显存管理

**时间线**：

```
T0: 训练阶段
    - FSDP 模型在 GPU（Sharded，每个 GPU 1/N 参数）
    - SGLang Engine 未启动

T1: 训练完成，调用 sleep()
    - FSDP 模型 offload 到 CPU
    - GPU 显存被释放

T2: Weight Sync
    - 从 CPU 加载 FSDP 模型（Sharded）
    - Redistribute 为完整参数（临时占用显存）
    - 传输到 SGLang Engine
    - 释放临时显存

T3: 推理阶段
    - SGLang Engine 在 GPU（TP Sharded）
    - FSDP 模型仍在 CPU

T4: 推理完成，准备下一轮训练
    - SGLang Engine 保持在 GPU（不清空）
    - 调用 wake_up()，FSDP 模型从 CPU 加载回 GPU
    - 可能存在短暂的显存峰值（FSDP + SGLang 同时在 GPU）
```

**显存优化策略**：

1. **分桶传输**：
   - 避免一次性加载所有参数
   - 每个 bucket 传输完立即释放

2. **异步操作**：
   - `async_op=True` 的 Redistribute
   - 边传输边释放

3. **SGLang Memory Fraction**：
   ```bash
   --sglang-mem-fraction-static=0.8
   # 限制 SGLang 最多使用 80% 的 GPU 显存
   ```

---

## 问题 4：KV Cache 在 TP/CP 下的管理

### SGLang 的 KV Cache 管理

#### 4.1 TP 模式下的 KV Cache

**核心机制**：KV Cache 在 TP 下是分片的（Sharded）

```python
# SGLang TP 模式下的 KV Cache 分片

# 假设：
# - 模型：32 个 Attention Heads
# - TP size = 4
# - Sequence length = 2048
# - Head dim = 128

# 每个 TP rank 负责的 Heads：
heads_per_rank = 32 // 4 = 8

# KV Cache 形状（每个 TP rank）：
# K: (batch_size, 8, 2048, 128)  # 8 heads per rank
# V: (batch_size, 8, 2048, 128)

# 完整的 KV Cache（逻辑上）：
# K: (batch_size, 32, 2048, 128)  # 32 heads total
# V: (batch_size, 32, 2048, 128)
```

**关键设计**：

引用自 SGLang 文档：
> "For tensor parallelism, each GPU maintains a sharded KV cache, with no need for additional synchronization because the tree operations are the same."

**解释**：
- **每个 TP rank 独立管理**：KV Cache 按 Head 维度切分
- **无需同步**：因为每个 TP rank 的 Attention 操作是独立的
- **All-Reduce 仅在输出**：Attention 输出需要 All-Reduce 汇总

**Attention 计算流程（TP 模式）**：

```python
# TP rank i 的 Attention 计算
class TPAttention(nn.Module):
    def forward(self, hidden_states, kv_cache):
        # 1. 计算 Q, K, V（每个 TP rank 负责部分 heads）
        Q = self.q_proj(hidden_states)  # (batch, seq, heads_per_rank * head_dim)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # 2. 更新 KV Cache（每个 TP rank 独立）
        kv_cache.update(K, V)  # 存储当前 TP rank 的 K, V

        # 3. 计算 Attention（每个 TP rank 独立）
        attn_output = scaled_dot_product_attention(Q, K, V)

        # 4. 输出投影（列并行）
        output = self.o_proj(attn_output)

        # 5. All-Reduce（汇总所有 TP ranks 的输出）
        output = all_reduce(output)  # ← 唯一的通信点

        return output
```

**显存占用**：

```python
# 示例：Llama-7B，TP=4，Batch size=32，Seq len=2048

# 单层 KV Cache（完整）：
# K: (32, 32, 2048, 128) × 2 bytes (BF16) = 512 MB
# V: (32, 32, 2048, 128) × 2 bytes (BF16) = 512 MB
# Total: 1024 MB

# 单层 KV Cache（每个 TP rank）：
# K: (32, 8, 2048, 128) × 2 bytes = 128 MB
# V: (32, 8, 2048, 128) × 2 bytes = 128 MB
# Total: 256 MB（节省 75%）

# 全模型（32 layers）：
# 每个 TP rank: 256 MB × 32 = 8 GB
```

#### 4.2 CP 模式下的 KV Cache

**Context Parallelism（CP）**：用于超长序列（64K+）

**核心机制**：KV Cache 在序列维度切分

```python
# CP 模式下的 KV Cache 分片

# 假设：
# - Sequence length = 32768
# - CP size = 4
# - Batch size = 8
# - 32 Heads，Head dim = 128

# 每个 CP rank 负责的序列长度：
seq_per_rank = 32768 // 4 = 8192

# KV Cache 形状（每个 CP rank）：
# K: (8, 32, 8192, 128)  # 序列切分
# V: (8, 32, 8192, 128)

# 完整的 KV Cache（逻辑上）：
# K: (8, 32, 32768, 128)
# V: (8, 32, 32768, 128)
```

**Ring Flash Attention**：

在 CP 模式下，SGLang 使用 Ring Flash Attention（与 Slime FSDP 训练相同）：

```python
# CP rank i 的 Ring Attention 计算
class CPAttention(nn.Module):
    def forward(self, Q_local, kv_cache_local, cp_group):
        # 1. 初始化输出
        attn_output = torch.zeros_like(Q_local)

        # 2. Ring 通信：逐步收集其他 CP ranks 的 KV
        for step in range(cp_size):
            # 当前 step 的 KV（可能来自其他 rank）
            K_current = kv_cache_local.K
            V_current = kv_cache_local.V

            # 计算局部 Attention
            local_output = scaled_dot_product_attention(Q_local, K_current, V_current)
            attn_output += local_output

            # Ring 通信：发送当前 KV 到下一个 rank，接收上一个 rank 的 KV
            if step < cp_size - 1:
                kv_cache_local = ring_exchange(kv_cache_local, cp_group)

        return attn_output
```

引用自 SGLang 文档：
> "With Context Parallel (CP), increasing the cp_size partitions the KV cache across devices, eliminating redundancy and enabling much longer sequences, with the effective memory budget scaling roughly linearly with the number of CP shards."

**显存节省**：

```python
# 示例：32K 序列，CP=4

# 无 CP：
# KV Cache: (batch, heads, 32768, head_dim) → 完整存储

# 有 CP：
# KV Cache: (batch, heads, 8192, head_dim) → 节省 75% 显存
```

#### 4.3 TP + CP 混合模式

**混合并行的 KV Cache 管理**：

```python
# 假设：TP=4, CP=2

# KV Cache 同时在两个维度分片：
# - Head 维度：按 TP 切分
# - Sequence 维度：按 CP 切分

# 示例：
# - Total Heads = 32
# - Total Seq Len = 32768
# - TP size = 4, CP size = 2

# 每个 (TP, CP) rank 的 KV Cache：
heads_per_tp = 32 // 4 = 8
seq_per_cp = 32768 // 2 = 16384

# KV Cache 形状：
# K: (batch, 8, 16384, 128)  # 8 heads, 16K seq
# V: (batch, 8, 16384, 128)

# 相比完整 KV Cache：
# Original: (batch, 32, 32768, 128)
# Sharded: (batch, 8, 16384, 128)
# 节省: 1 - (8 * 16384) / (32 * 32768) = 87.5%
```

**通信模式**：

```python
class HybridTPCPAttention(nn.Module):
    def forward(self, Q, kv_cache, tp_group, cp_group):
        # 1. TP 维度：每个 TP rank 负责部分 heads
        # （无需通信，各自计算）

        # 2. CP 维度：Ring Attention
        attn_output = ring_flash_attention(Q, kv_cache, cp_group)

        # 3. TP All-Reduce：汇总输出
        attn_output = all_reduce(attn_output, tp_group)

        return attn_output
```

**关键点**：
- **TP 和 CP 正交**：可以同时启用，互不干扰
- **显存线性扩展**：`Memory / (TP_size × CP_size)`
- **通信开销**：TP 需要 All-Reduce，CP 需要 Ring Exchange

#### 4.4 Slime 中的 CP 支持

**训练侧（FSDP）**：

```python
# slime/backends/fsdp_utils/actor.py:164-210
def _setup_device_mesh(self) -> None:
    """Setup device mesh for parallelism (always called, handles both CP and non-CP cases)."""

    # 使用 context_parallel_size
    self.cp_size = self.args.context_parallel_size
    self.dp_size = world_size // self.cp_size

    # 创建 2D Device Mesh: (dp_size, cp_size)
    self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size, self.cp_size))

    # 提取 CP group
    self.cp_group = self.mesh.get_group("cp")
```

**推理侧（SGLang）**：

```python
# SGLang 也支持 CP（decode context parallel）
# 但需要确保训练和推理的 CP 配置一致

# 示例配置：
# --context-parallel-size=2  # 训练和推理都使用 CP=2
```

**注意事项**：
- ✅ **训练和推理的 CP 配置应一致**：避免权重同步时的维度不匹配
- ✅ **CP 主要用于长序列**：如 32K+，短序列（2K-8K）不需要 CP
- ⚠️ **CP 增加通信**：Ring Attention 需要多轮通信

---

## 实战案例：完整的训练推理切换流程

### 案例配置

```bash
# Colocated 模式，8 GPU
python train.py \
    --train-backend fsdp \
    --colocate \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus-per-engine 4 \
    --offload-train \
    --context-parallel-size 2
```

**配置解析**：
- **训练**：FSDP，DP=4（8 GPU / CP=2），CP=2
- **推理**：SGLang，2 个 Engine，每个 TP=4
- **Colocated**：训练和推理共享 8 个 GPU
- **Offload Train**：训练完成后 offload 到 CPU

### 完整流程

#### Step 1：训练阶段

```
[GPU 状态]
GPU 0-7: FSDP 模型（Sharded，每个 GPU 1/4 参数，CP=2）

[显存占用]
每个 GPU:
  - Sharded Params: 1.75 GB (7B / 4)
  - Optimizer States: 3.5 GB (exp_avg + exp_avg_sq)
  - Activations: 动态
  Total: ~5-10 GB
```

**训练代码**：

```python
# slime/backends/fsdp_utils/actor.py:447-467
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    # 1. Wake up（从 CPU 加载回 GPU）
    if self.args.offload_train:
        self.wake_up()  # FSDP 模型从 CPU → GPU

    # 2. 训练循环
    self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

    # 3. 训练结束（不调用 sleep）
    # 模型保持在 GPU，仍然是 Sharded 状态
```

#### Step 2：准备切换到推理

```python
# 训练完成后，准备 Weight Sync
# 注意：此时 FSDP 模型仍在 GPU

# 如果启用 offload_train，先调用 sleep()
if args.offload_train:
    actor.sleep()  # FSDP 模型从 GPU → CPU
```

**GPU 状态变化**：

```
Before sleep():
GPU 0-7: FSDP 模型（Sharded）+ Optimizer States
显存占用: ~5-10 GB / GPU

After sleep():
GPU 0-7: 空闲
CPU RAM: FSDP 模型（Sharded）+ Optimizer States
```

#### Step 3：权重同步

```python
# slime/backends/fsdp_utils/actor.py:749-766
def update_weights(self) -> None:
    """同步权重到 SGLang"""

    # 1. 连接 SGLang Engines（如果有新 Engine）
    rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
        self.rollout_manager.get_rollout_engines_and_lock.remote()
    )
    if num_new_engines > 0:
        self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)

    # 2. 执行权重更新
    self.weight_updater.update_weights()

    # 3. 清理显存
    clear_memory()
```

**详细步骤**：

```
[Step 3.1] 从 CPU 加载 FSDP 模型到 GPU（临时）
  - 每个 GPU 加载自己的 Sharded 参数
  - 显存占用临时增加

[Step 3.2] Redistribute（Shard → Replicate）
  - All-Gather 收集完整参数
  - 每个 GPU 现在有完整参数副本
  - 显存峰值：~28 GB（7B 模型，BF16）

[Step 3.3] 分桶传输
  - Bucket 1: layer 0-10
    - 扁平化、序列化
    - TP Group Gather（GPU 0-3 → GPU 0，GPU 4-7 → GPU 4）
    - RPC 到 SGLang Engine
    - 释放 Bucket 1 显存

  - Bucket 2: layer 11-20
    - 重复上述流程

  - Bucket 3: layer 21-31
    - 重复上述流程

[Step 3.4] SGLang 加载权重
  - Engine 0 (GPU 0-3): 接收完整权重 → TP 切分 → 加载
  - Engine 1 (GPU 4-7): 接收完整权重 → TP 切分 → 加载

[Step 3.5] 清理
  - 释放 FSDP 模型的临时 GPU 显存
  - FSDP 模型回到 CPU
```

**GPU 状态变化**：

```
During Weight Sync:
GPU 0-3:
  - FSDP Sharded Params (临时)
  - Redistributed Full Params (临时)
  - Bucketed Params (逐个)
  峰值显存: ~30 GB（短暂）

GPU 4-7: 同上

After Weight Sync:
GPU 0-7: 空闲（FSDP 模型回到 CPU）

SGLang Engines 准备就绪:
Engine 0 (GPU 0-3): TP Sharded 模型 + KV Cache Pool
Engine 1 (GPU 4-7): TP Sharded 模型 + KV Cache Pool
```

#### Step 4：推理阶段

```
[GPU 状态]
Engine 0 (GPU 0-3):
  - TP Sharded 模型 (每个 GPU ~1.75 GB)
  - KV Cache Pool (动态分配)

Engine 1 (GPU 4-7): 同上

[CPU 状态]
FSDP 模型（Sharded）+ Optimizer States
```

**推理代码**：

```python
# slime/rollout/sglang_rollout.py
async def generate(args, sample, sampling_params):
    """使用 SGLang 生成 response"""

    # 1. 路由到某个 Engine（通过 Router）
    url = await router.get_engine_url()

    # 2. 调用 SGLang API
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            f"{url}/generate",
            json={
                "prompt": sample.prompt,
                "sampling_params": sampling_params,
            }
        )

    # 3. SGLang 内部使用 TP + CP 进行推理
    # - TP: 每个 GPU 计算部分 heads
    # - CP: Ring Attention 处理长序列
    # - KV Cache: 在 TP 和 CP 维度都分片

    return response
```

#### Step 5：下一轮训练

```python
# 推理完成，准备下一轮训练

# 1. 调用 wake_up()（如果启用 offload_train）
if args.offload_train:
    actor.wake_up()  # FSDP 模型从 CPU → GPU

# 2. 开始新一轮训练
actor.train(rollout_id=next_rollout_id, rollout_data_ref=next_data)
```

**潜在的显存峰值**：

```
[并发时刻]
GPU 0-7:
  - FSDP 模型（Sharded，刚 wake_up）
  - SGLang 模型（TP Sharded，仍在运行）

峰值显存可能超出限制！

[解决方案]
1. 等待 SGLang 完成所有请求
2. 清理 SGLang KV Cache
3. 再 wake_up FSDP 模型

或者：使用 Disaggregated 模式（训练和推理在不同 GPU）
```

---

## 关键设计决策

### 1. 为什么不在推理时使用 Sharded 状态？

**问题**：能否直接在 FSDP Sharded 状态下进行推理，避免 Redistribute？

**答案**：不行，原因如下：

1. **切分维度不同**：
   ```
   FSDP (DP): 按 batch 维度切分（每个 GPU 处理不同的样本）
   SGLang (TP): 按模型维度切分（每个 GPU 计算不同的 heads/columns）
   ```

2. **推理是序列化的**：
   - 推理时，一个请求需要完整的模型参数
   - DP 切分无法提供单个请求的完整参数

3. **KV Cache 管理**：
   - TP 下，KV Cache 按 head 维度切分
   - DP 下，KV Cache 需要在 GPU 间复制（浪费显存）

### 2. 为什么使用分桶传输？

**问题**：为什么不一次性传输所有参数？

**答案**：避免显存峰值

```python
# 一次性传输（不推荐）：
all_params = gather_all_parameters()  # 需要 28 GB 显存
send_to_sglang(all_params)
free(all_params)  # 峰值已经产生

# 分桶传输（推荐）：
for bucket in param_buckets:
    bucket_params = gather_bucket(bucket)  # 需要 2-3 GB 显存
    send_to_sglang(bucket_params)
    free(bucket_params)  # 立即释放
# 峰值显存降低 10x
```

### 3. 为什么 SGLang 接收完整权重而不是分片？

**问题**：能否让 SGLang 直接接收 TP 分片的权重？

**答案**：理论上可以，但实现复杂：

1. **切分逻辑不同**：
   - FSDP: 按行切分（row-wise）或按参数切分
   - TP: 按列切分（column-wise）或按 head 切分

2. **映射复杂**：
   - 需要精确计算 FSDP rank i 的分片对应 TP rank j 的哪部分
   - 不同层的切分策略可能不同（如 QKV 合并层）

3. **灵活性**：
   - 接收完整权重允许 SGLang 自由选择 TP 策略
   - 可以动态调整 TP size 而不改变训练侧

**当前设计的优势**：
- ✅ **解耦训练和推理**：两侧独立优化
- ✅ **简化实现**：训练侧只需 Replicate，推理侧自行切分
- ✅ **容错性更好**：完整权重便于验证和调试

---

## 性能优化建议

### 1. 减少 Weight Sync 时间

**优化策略**：

```python
# 1. 增大 Bucket Size（减少 RPC 次数）
--update-weight-buffer-size=200000000  # 200 MB per bucket

# 2. 使用异步传输
# 在 update_weight_utils.py 中已实现
param.redistribute(..., async_op=True)

# 3. Overlap Weight Sync 和推理
# 在推理开始后异步传输（如果权重未变）
```

### 2. 优化 Colocated 模式的显存

**策略**：

```python
# 1. 限制 SGLang 显存
--sglang-mem-fraction-static=0.75  # 留出 25% 给 Weight Sync

# 2. 启用 FSDP CPU Offload
--offload-train  # 训练完成后 offload 到 CPU

# 3. 使用更小的 Batch Size（推理）
--sglang-max-num-reqs=128  # 限制并发请求数
```

### 3. 选择合适的并行策略

**场景对比**：

| 场景 | 推荐配置 | 原因 |
|------|---------|------|
| **短序列（<8K）** | TP=4, CP=1 | CP 开销大于收益 |
| **长序列（32K+）** | TP=2, CP=4 | CP 显存节省显著 |
| **多节点训练** | Disaggregated 模式 | 避免跨节点 Weight Sync |
| **显存受限** | Colocated + Offload | 共享 GPU，offload 到 CPU |

---

## 常见问题（FAQ）

### Q1: Colocated 模式下，训练和推理是否会同时在 GPU 上？

**A**: 不会。通过以下机制确保互斥：

1. **训练阶段**：FSDP 模型在 GPU，SGLang 未启动或空闲
2. **Weight Sync**：临时加载 FSDP 模型，传输后立即释放
3. **推理阶段**：SGLang 在 GPU，FSDP 模型在 CPU（如果启用 offload_train）

**时间线**：
```
Train → Sleep → Weight Sync → Inference → Wake Up → Train
```

### Q2: 如果不启用 `--offload-train`，会怎样？

**A**: FSDP 模型和 SGLang 会同时在 GPU，可能导致 OOM（显存不足）。

**建议**：
- **Colocated 模式**：必须启用 `--offload-train`
- **Disaggregated 模式**：不需要 offload

### Q3: 能否让 FSDP 使用 TP 而不是 DP？

**A**: FSDP2 目前只支持 DP + CP，不支持 TP。

**如果需要 TP**：
- 使用 Megatron-LM 后端（Slime 也支持）
- Megatron 支持 TP + PP + DP + CP 的全维度并行

### Q4: Weight Sync 的通信开销有多大？

**A**: 取决于模型大小和网络带宽。

**示例**（7B 模型）：
```
模型大小: 28 GB (FP32) 或 14 GB (BF16)
网络带宽: 100 Gbps (InfiniBand)

Weight Sync 时间: 14 GB / (100 Gbps / 8) = 1.12 秒

实际时间（包括序列化、RPC 开销）: ~2-3 秒
```

**优化**：
- 使用更快的网络（如 NVLink、InfiniBand HDR）
- 分桶传输，减少峰值显存
- 异步传输，与其他操作重叠

### Q5: KV Cache 在 TP + CP 下的显存占用如何计算？

**A**: 公式如下：

```python
# 单层 KV Cache 显存（BF16）
kv_cache_per_layer = (
    batch_size *
    (num_heads / tp_size) *  # TP 切分
    (seq_len / cp_size) *    # CP 切分
    head_dim *
    2 *  # K and V
    2    # BF16 (2 bytes)
)

# 示例：
# batch=32, num_heads=32, seq_len=32768, head_dim=128, TP=4, CP=2
kv_cache_per_layer = 32 * (32/4) * (32768/2) * 128 * 2 * 2
                   = 32 * 8 * 16384 * 128 * 4
                   = 2,147,483,648 bytes
                   = 2 GB

# 32 层: 2 GB * 32 = 64 GB（分布在 TP=4 个 GPU 上，每个 16 GB）
```

---

## 源码索引

| 功能 | 文件 | 行号 |
|------|------|------|
| FSDP Sleep/Wake Up | `slime/backends/fsdp_utils/actor.py` | 276-298 |
| Weight Sync 入口 | `slime/backends/fsdp_utils/actor.py` | 749-766 |
| Redistribute（Shard → Replicate） | `slime/backends/fsdp_utils/update_weight_utils.py` | 56-62 |
| 分桶传输 | `slime/backends/fsdp_utils/update_weight_utils.py` | 45-70 |
| 扁平化 Bucket | `slime/backends/fsdp_utils/update_weight_utils.py` | 117-138 |
| TP Group Gather | `slime/backends/fsdp_utils/update_weight_utils.py` | 140-152 |
| RPC 到 SGLang | `slime/backends/fsdp_utils/update_weight_utils.py` | 154-171 |
| Device Mesh 设置（CP） | `slime/backends/fsdp_utils/actor.py` | 164-210 |
| Colocated 配置 | `slime/ray/placement_group.py` | 87-89 |

---

## 总结

### 核心要点

1. **FSDP 训练后仍是 Sharded 状态**：
   - 参数保持 DTensor 形式
   - 每个 GPU 存储 1/N 的参数（DP 分片）
   - 不会自动转换为完整模型

2. **权重同步机制**：
   - Redistribute（Shard → Replicate）收集完整参数
   - 分桶传输，避免显存峰值
   - TP Group Gather + RPC 到 SGLang

3. **SGLang 使用 TP 进行推理**：
   - 接收完整权重，内部按 TP 切分
   - KV Cache 在 TP 下按 head 维度分片
   - 支持 TP + CP 混合并行

4. **Colocated 模式的显存管理**：
   - 训练后 Sleep（offload 到 CPU）
   - Weight Sync 时临时加载并传输
   - 推理时 SGLang 在 GPU，FSDP 在 CPU

5. **KV Cache 在 TP/CP 下高效分片**：
   - TP: 按 head 维度切分
   - CP: 按 sequence 维度切分
   - 显存占用线性扩展：`Memory / (TP_size × CP_size)`

### 学习要点

对于想要在其他框架中复现类似机制的 Infra 学习者：

1. **理解分片的本质**：
   - 训练（DP）：按 batch 切分
   - 推理（TP）：按模型维度切分
   - 两者不兼容，需要转换

2. **掌握 Redistribute 操作**：
   - DTensor 的 `redistribute` 方法
   - `async_op=True` 实现异步通信
   - All-Gather 的通信开销

3. **实现高效的权重传输**：
   - 分桶策略（避免显存峰值）
   - 扁平化（减少通信次数）
   - RPC 或 NCCL Broadcast

4. **设计 KV Cache 分片策略**：
   - TP: 按 head 切分（正交性）
   - CP: 按 sequence 切分（Ring Attention）
   - 混合：两个维度独立切分

5. **优化 Colocated 模式的显存**：
   - Offload 机制（CPU/GPU 切换）
   - 显存预留（Memory Fraction）
   - 时间复用（训练推理互斥）

---

## 参考资源

### 官方文档

- [PyTorch FSDP2 API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch DTensor Guide](https://docs.pytorch.org/tutorials/intermediate/dtensor_tutorial.html)
- [SGLang Documentation](https://github.com/sgl-project/sglang)

### 相关博客

- [SGLang v0.4: Cache-Aware Load Balancer](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [Deploying DeepSeek with Large-Scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

### Slime 框架文档

- `docs/analysis/fsdp2_implementation_deep_dive.md`
- `docs/analysis/fsdp2_communication_overlap_and_memory_management.md`
- `docs/analysis/fsdp2_master_weights_gradient_clip_communication.md`

---

**文档版本**：v1.0
**基于**：Slime (commit: 9d7f34d), PyTorch FSDP2 (2.9), SGLang (latest)
**生成日期**：2025-12-12
**目标读者**：Infra 学习者，希望理解训练推理切换机制的工程师
