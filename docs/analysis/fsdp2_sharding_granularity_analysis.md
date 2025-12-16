# FSDP2 Sharding Granularity Analysis

## Problem Statement

**问题-3**: 在 Megatron 中我们需要手动编写模型并控制并行，而在 FSDP2 中，通过 AutoModel 加载的模型，是针对每一层 Transformer Block 做 Shard，还是针对整个 Model 做 Shard？这通过什么参数控制？

**Translation**: In Megatron we need to manually write models and control parallelism, but in FSDP2, for models loaded through AutoModel, is sharding done per Transformer Block layer or for the entire Model? What parameter controls this?

---

## Executive Summary

**答案核心**: FSDP2 采用**分层分片（Hierarchical Sharding）**策略，即**对每一层 Transformer Block 单独应用分片，然后再对顶层模型整体应用分片**。这通过 HuggingFace 模型的 `_no_split_modules` 属性控制，该属性定义了哪些模块类型应该作为原子分片单元。

**Key Answer**: FSDP2 uses a **hierarchical sharding** strategy, which means it **applies sharding to each individual Transformer Block layer, then applies sharding to the top-level model**. This is controlled by the HuggingFace model's `_no_split_modules` attribute, which defines which module types should be treated as atomic sharding units.

---

## 1. Core Implementation Analysis

### 1.1 The `apply_fsdp2()` Function

**Location**: `slime/backends/fsdp_utils/actor.py:1016-1058`

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

    # Step 1: Get the list of module class names that should be treated as atomic units
    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    # Step 2: Collect all module instances matching these class names
    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # Step 3: Apply FSDP to each Transformer Block layer individually
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Step 4: Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model
```

---

## 2. Sharding Granularity: Per-Layer vs Whole-Model

### 2.1 Answer: Per-Layer Hierarchical Sharding

**FSDP2 采用分层分片策略，具体为:**

1. **第一阶段**: 遍历所有 Transformer Block 层，对每一层单独调用 `fully_shard(module, **fsdp_kwargs)`
2. **第二阶段**: 对顶层模型调用 `fully_shard(model, **fsdp_kwargs)`

这种分层策略意味着：
- 每个 Transformer Block 的参数被独立分片到不同的 rank
- 在前向/反向传播时，每个 Transformer Block 可以独立地进行参数的 all-gather 和 reduce-scatter
- 顶层模型的分片处理了不属于 Transformer Block 的参数（如 embedding 层、final layer norm 等）

**Translation:**

**FSDP2 uses a hierarchical sharding strategy, specifically:**

1. **Phase 1**: Iterate through all Transformer Block layers, calling `fully_shard(module, **fsdp_kwargs)` on each layer individually
2. **Phase 2**: Call `fully_shard(model, **fsdp_kwargs)` on the top-level model

This hierarchical strategy means:
- Each Transformer Block's parameters are independently sharded across different ranks
- During forward/backward propagation, each Transformer Block can independently perform all-gather and reduce-scatter operations on parameters
- The top-level model sharding handles parameters not belonging to Transformer Blocks (e.g., embedding layers, final layer norm, etc.)

### 2.2 Visualization

```
Model Structure:
├── Embeddings (处理于 top-level fully_shard)
├── TransformerBlock_0 (独立 fully_shard) ← 第一阶段处理
│   ├── Attention
│   ├── MLP
│   └── LayerNorm
├── TransformerBlock_1 (独立 fully_shard) ← 第一阶段处理
│   ├── Attention
│   ├── MLP
│   └── LayerNorm
├── ...
├── TransformerBlock_N (独立 fully_shard) ← 第一阶段处理
│   ├── Attention
│   ├── MLP
│   └── LayerNorm
└── Final LayerNorm (处理于 top-level fully_shard)

整体模型 (top-level fully_shard) ← 第二阶段处理
```

---

## 3. Control Parameter: `_no_split_modules`

### 3.1 What is `_no_split_modules`?

`_no_split_modules` 是 HuggingFace Transformers 模型的一个类属性，定义了**不应该被进一步拆分的模块类名列表**。在 FSDP2 中，这些模块被视为**原子分片单元**。

**Translation**: `_no_split_modules` is a class attribute of HuggingFace Transformers models that defines **a list of module class names that should not be further split**. In FSDP2, these modules are treated as **atomic sharding units**.

### 3.2 Examples from Different Models

```python
# GPT-2
class GPT2LMHeadModel(GPT2PreTrainedModel):
    _no_split_modules = ["GPT2Block"]
    # 每个 GPT2Block 是一个原子分片单元

# LLaMA
class LlamaForCausalLM(LlamaPreTrainedModel):
    _no_split_modules = ["LlamaDecoderLayer"]
    # 每个 LlamaDecoderLayer 是一个原子分片单元

# Qwen
class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _no_split_modules = ["Qwen2DecoderLayer"]
    # 每个 Qwen2DecoderLayer 是一个原子分片单元

# GLM-4
class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    _no_split_modules = ["GLMBlock"]
    # 每个 GLMBlock 是一个原子分片单元
```

### 3.3 How FSDP2 Uses `_no_split_modules`

**Source Code Analysis** (`actor.py:1031-1039`):

```python
# 获取需要包装的层类型名称
layer_cls_to_wrap = model._no_split_modules
assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

# 遍历模型的所有命名模块，筛选出匹配的层
modules = [
    module
    for name, module in model.named_modules()
    if module.__class__.__name__ in layer_cls_to_wrap
    or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
]
```

**关键逻辑**:
1. 从模型的 `_no_split_modules` 属性读取层类型名称列表（如 `["Qwen2DecoderLayer"]`）
2. 遍历模型的所有模块 (`model.named_modules()`)
3. 筛选出类名匹配的模块（如所有 `Qwen2DecoderLayer` 实例）
4. 额外处理：如果模型不使用 tied word embeddings，也将 Embedding 层作为独立的分片单元

**Translation of Key Logic**:
1. Read the list of layer type names from the model's `_no_split_modules` attribute (e.g., `["Qwen2DecoderLayer"]`)
2. Iterate through all modules in the model (`model.named_modules()`)
3. Filter out modules whose class names match (e.g., all `Qwen2DecoderLayer` instances)
4. Additional handling: If the model doesn't use tied word embeddings, also treat the Embedding layer as an independent sharding unit

### 3.4 Special Handling for Embeddings

```python
or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
```

**特殊处理逻辑**:
- 如果模型的 word embeddings 和 output embeddings 不共享参数 (`tie_word_embeddings=False`)
- 则 Embedding 层也会被单独包装为一个 FSDP 单元
- 这避免了 Embedding 层的参数被顶层包装时才处理，提高了内存效率

**Translation of Special Handling Logic**:
- If the model's word embeddings and output embeddings don't share parameters (`tie_word_embeddings=False`)
- The Embedding layer is also wrapped as a separate FSDP unit
- This avoids the Embedding layer parameters being handled only by the top-level wrapping, improving memory efficiency

---

## 4. Megatron vs FSDP2: Control Philosophy

### 4.1 Megatron: Manual Explicit Control

**Megatron 的并行控制方式**:

```python
# 在 Megatron 中，你需要手动编写模型代码并显式控制并行
class TransformerLayer(torch.nn.Module):
    def __init__(self, ...):
        # 手动指定 Tensor Parallel
        self.attention = ColumnParallelLinear(
            hidden_size,
            num_attention_heads * hidden_size,
            tensor_parallel_group=tensor_parallel_group
        )

        # 手动指定 Pipeline Parallel
        if is_pipeline_first_stage():
            self.embedding = Embedding(...)

        # 手动指定数据并行
        # (通过 DDP 或 Distributed Optimizer)

# 并行配置通过命令行参数显式控制
--tensor-model-parallel-size 4 \
--pipeline-model-parallel-size 2 \
--data-parallel-size 8
```

**特点**:
- ✅ **精细控制**: 可以精确控制每一层的并行策略
- ✅ **高性能**: 手动优化可以达到最优性能
- ❌ **高复杂度**: 需要重写模型代码，维护成本高
- ❌ **低灵活性**: 切换模型需要重新实现并行逻辑

### 4.2 FSDP2: Automatic Convention-Based Control

**FSDP2 的并行控制方式**:

```python
# 在 FSDP2 中，直接使用 HuggingFace AutoModel 加载模型
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/model",
    trust_remote_code=True,
    attn_implementation="flash_attention_3"
)

# 应用 FSDP2，自动根据 _no_split_modules 进行分片
model = apply_fsdp2(model, mesh=device_mesh, cpu_offload=False)

# 并行配置由 DeviceMesh 控制
mesh = init_device_mesh("cuda", mesh_shape=(dp_size, cp_size))
```

**特点**:
- ✅ **零代码修改**: 直接使用 HuggingFace 模型，无需重写
- ✅ **高灵活性**: 切换模型只需更改 checkpoint 路径
- ✅ **自动发现**: 通过 `_no_split_modules` 自动识别分片边界
- ⚠️ **有限控制**: 无法像 Megatron 那样精细控制每层的并行策略
- ⚠️ **性能差距**: 对于超大规模训练（>100B 参数），可能不如手动优化的 Megatron

### 4.3 Comparison Table

| 维度 | Megatron | FSDP2 |
|------|----------|-------|
| **模型编写** | 手动编写并行模型代码 | 直接使用 HuggingFace AutoModel |
| **分片控制** | 显式指定每层的 TP/PP/DP | 自动根据 `_no_split_modules` 分片 |
| **代码侵入性** | 高（需要重写模型） | 低（零代码修改） |
| **灵活性** | 低（切换模型需重新实现） | 高（直接加载不同 checkpoint） |
| **并行策略** | 支持 TP+PP+DP+CP | 主要支持 DP+CP（TP 需要额外工作） |
| **性能** | 最优（手动优化） | 良好（自动化有少量开销） |
| **适用场景** | 超大规模训练 (>100B) | 中等规模训练 (<100B) + 快速迭代 |

---

## 5. Implementation Details in slime

### 5.1 Model Loading Flow

**完整流程** (`actor.py:85-92`):

```python
# Step 1: 使用 HuggingFace AutoModel 加载模型
model = AutoModelForCausalLM.from_pretrained(
    self.args.hf_checkpoint,
    trust_remote_code=True,
    attn_implementation=self.args.attn_implementation,
)

# Step 2: 设置 DeviceMesh (DP + CP)
self._setup_device_mesh()
# 创建 2D mesh: (dp_size, cp_size)
# 例如: 8 GPUs, cp_size=2 → mesh shape = (4, 2)

# Step 3: 应用 FSDP2 包装
model = apply_fsdp2(
    model=model,
    mesh=self.mesh,
    cpu_offload=self.args.fsdp_cpu_offload
)
# 内部调用:
#   - 对每个 Transformer Block 调用 fully_shard()
#   - 对顶层模型调用 fully_shard()
```

### 5.2 Parameter Sharding During Training

**前向传播时的参数处理**:

```python
# 对于每个 Transformer Block:
# 1. All-Gather: 从所有 DP rank 收集完整参数
#    (例如: 每个 rank 有 1/4 参数 → all-gather 后每个 rank 有完整参数)

# 2. Forward Pass: 使用完整参数进行前向计算

# 3. Discard: 前向计算完成后，释放其他 rank 的参数，仅保留本 rank 的分片

# 反向传播时的梯度处理:
# 1. All-Gather: 再次收集完整参数用于梯度计算

# 2. Backward Pass: 计算梯度

# 3. Reduce-Scatter: 将梯度按分片维度聚合到对应的 rank
#    (例如: rank 0 收集参数 0-25% 的梯度，rank 1 收集 25-50% 的梯度)

# 4. Discard: 释放不属于本 rank 的梯度
```

### 5.3 Memory Efficiency Analysis

**为什么分层分片更高效？**

假设模型有 32 层 Transformer Blocks，每层参数 1GB，使用 4 个 DP rank:

**方案 A: 整个模型一次性分片 (如果 FSDP2 不分层)**
- 在前向传播时，需要 all-gather 整个模型的 32GB 参数
- 峰值内存: 32GB (完整参数) + 32GB (激活值) = 64GB

**方案 B: 分层分片 (FSDP2 实际做法)**
- 在前向传播时，每次只 all-gather 当前层的 1GB 参数
- 该层计算完成后，立即释放参数，只保留 256MB 分片
- 峰值内存: 1GB (当前层完整参数) + 激活值 ≈ 低得多

**内存节省计算**:
- 方案 A 峰值: 32GB 参数
- 方案 B 峰值: 1GB 参数 (当前层) + 32 * 0.25GB = 9GB
- **内存节省: ~71%**

---

## 6. Control Parameters Summary

### 6.1 Primary Control: `_no_split_modules`

**定义位置**: HuggingFace Transformers 模型类定义中

**作用**:
- 定义哪些模块类型是原子分片单元
- FSDP2 会对每个匹配的模块单独调用 `fully_shard()`
- 这是**唯一**控制分片粒度的参数

**如何修改** (通常不需要):
```python
# 如果你想自定义分片粒度（极少见的情况）
model._no_split_modules = ["CustomTransformerLayer"]
```

### 6.2 Secondary Control: `mesh` Parameter

**定义**: 传递给 `apply_fsdp2()` 的 DeviceMesh

**作用**:
- 控制参数如何在 rank 之间分布
- 在 DP+CP 模式下，使用 2D mesh (dp_size, cp_size)
- FSDP2 在 `dp` 维度上分片参数

**示例**:
```python
# 8 GPUs, context_parallel_size = 2
mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("dp", "cp"))
# FSDP2 会在 4 个 DP rank 之间分片参数
# 每个 DP rank 持有 1/4 的参数
```

### 6.3 Tertiary Control: `cpu_offload` Parameter

**定义**: `apply_fsdp2()` 的 `cpu_offload` 参数

**作用**:
- 当设置为 `True` 时，参数、梯度和优化器状态会 offload 到 CPU
- 优化器更新在 CPU 上执行
- 极大降低 GPU 显存需求，但会显著降低训练速度

**命令行参数**: `--fsdp-cpu-offload`

**示例**:
```bash
# 启用 CPU offload
bash scripts/run-qwen3-4B-fsdp.sh --fsdp-cpu-offload
```

---

## 7. Practical Examples

### 7.1 Example 1: Qwen3-4B with FSDP2

**配置**: `scripts/run-qwen3-4B-fsdp.sh`

```bash
TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --colocate
)
```

**执行流程**:

1. **加载模型**: `AutoModelForCausalLM.from_pretrained("/root/Qwen3-4B")`
   - Qwen3 模型的 `_no_split_modules = ["Qwen2DecoderLayer"]`

2. **创建 DeviceMesh**: `mesh_shape=(8, 1)` (纯 DP，无 CP)
   - 8 个 GPU 用于数据并行

3. **应用 FSDP2**:
   ```python
   # 找到所有 Qwen2DecoderLayer 实例 (假设有 32 层)
   modules = [layer_0, layer_1, ..., layer_31]

   # 对每层应用 fully_shard
   for layer in modules:
       fully_shard(layer, mesh=mesh, mp_policy=...)

   # 对顶层模型应用 fully_shard
   fully_shard(model, mesh=mesh, mp_policy=...)
   ```

4. **结果**:
   - 每个 GPU 持有每层参数的 1/8
   - 训练时，每层独立进行 all-gather 和 reduce-scatter
   - 内存高效，支持大 batch size

### 7.2 Example 2: Adding Context Parallel

**修改配置** (假设要添加 CP=2):

```bash
TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --context-parallel-size 2  # 添加这一行
   --attn-implementation ring  # CP 需要 Ring Attention
)

MISC_ARGS=(
   --actor-num-gpus-per-node 8
)
```

**执行流程变化**:

1. **DeviceMesh 变为 2D**: `mesh_shape=(4, 2)`
   - 4 个 DP group，每个 group 2 个 CP rank

2. **FSDP2 分片在 DP 维度**:
   ```python
   # 参数在 DP 维度 (第 0 维) 分片
   # 每个 DP rank 持有 1/4 参数
   # CP 维度 (第 1 维) 用于序列并行，参数在 CP 内 replicate
   ```

3. **结果**:
   - DP: 4-way 参数分片
   - CP: 2-way 序列分片 (all-to-all 通信)
   - 支持更长的序列 (2x context length)

---

## 8. Advanced Topics

### 8.1 Why Not Shard at Attention/MLP Level?

**问题**: 为什么不对 Attention 和 MLP 子模块分别分片？

**答案**:

1. **通信开销**:
   - 更细粒度的分片意味着更频繁的 all-gather/reduce-scatter
   - 每次通信都有固定的延迟开销 (latency)
   - 过于细粒度会使延迟开销占主导地位

2. **内存效率递减**:
   - Transformer Block 级别已经足够细粒度
   - 每层参数通常在 100MB-1GB 范围，all-gather 开销可接受
   - 更细粒度的节省很少（几十 MB），但通信成本显著增加

3. **代码复杂度**:
   - Transformer Block 是自然的语义边界
   - 在这个级别分片最符合模型架构直觉
   - 更细粒度需要对模型内部结构做更多假设

### 8.2 Can Users Customize Sharding Granularity?

**问题**: 用户能否自定义分片粒度？

**答案**: 理论上可以，实际上不推荐

**方法 1: 修改 `_no_split_modules`** (不推荐):
```python
# 加载模型后修改
model = AutoModelForCausalLM.from_pretrained(...)
model._no_split_modules = ["CustomModule"]  # 自定义分片单元

# 然后应用 FSDP2
model = apply_fsdp2(model, mesh=mesh)
```

**问题**:
- `_no_split_modules` 是 HuggingFace 的内部约定，不应该随意修改
- 可能导致 checkpoint 加载失败
- 可能破坏模型的其他功能（如 `device_map="auto"`）

**方法 2: 手动调用 `fully_shard()`** (高级用户):
```python
from torch.distributed.fsdp import fully_shard

# 手动控制每个子模块的分片
for name, module in model.named_modules():
    if "attention" in name:
        fully_shard(module, mesh=mesh_attention)
    elif "mlp" in name:
        fully_shard(module, mesh=mesh_mlp)

# 顶层包装
fully_shard(model, mesh=mesh)
```

**适用场景**:
- 研究新的并行策略
- 极端优化性能
- 不需要 HuggingFace 生态兼容性

**推荐**: 对于绝大多数用户，使用默认的 `_no_split_modules` 即可。

### 8.3 FSDP2 vs Tensor Parallelism

**对比**:

| 特性 | FSDP2 (Data Parallel) | Tensor Parallelism (TP) |
|------|----------------------|-------------------------|
| **分片维度** | 按层分片（Layer-wise） | 按张量维度分片（Tensor-wise） |
| **通信时机** | 每层前后 all-gather/reduce-scatter | 每个矩阵乘法前后 all-reduce/all-gather |
| **通信量** | 较少（每层一次） | 较多（每个 op 一次） |
| **内存效率** | 高（峰值内存低） | 中等 |
| **计算效率** | 高（通信少） | 中等（通信多） |
| **实现难度** | 低（自动化） | 高（需要手动修改模型） |
| **适用场景** | 多节点、中等规模 | 单节点、超大模型 |

**slime 的选择**:
- slime 主要使用 FSDP2 (DP + CP) 而非 Tensor Parallelism
- 原因：更好的 HuggingFace 兼容性 + 更低的实现复杂度
- 对于需要 TP 的场景（超大模型），建议使用 Megatron 后端

---

## 9. Key Takeaways

### 9.1 核心结论

1. **分片粒度**: FSDP2 对**每一层 Transformer Block** 单独分片，而非整个模型一次性分片

2. **控制参数**: 通过 HuggingFace 模型的 `_no_split_modules` 属性控制，该属性定义了原子分片单元的类名列表

3. **自动发现**: FSDP2 自动遍历模型，筛选出匹配 `_no_split_modules` 的所有模块实例，并对每个实例调用 `fully_shard()`

4. **层次化**: 先对每层调用 `fully_shard()`，再对顶层模型调用 `fully_shard()`，形成层次化的 FSDP 结构

5. **零修改**: 用户无需修改模型代码，直接使用 HuggingFace `AutoModel` 加载，FSDP2 自动处理分片

### 9.2 与 Megatron 对比

| 维度 | Megatron | FSDP2 |
|------|----------|-------|
| **并行控制** | 手动编写并行模型 | 自动基于 `_no_split_modules` |
| **分片粒度** | 显式指定每层的 TP/PP/DP | 自动按 Transformer Block 分片 |
| **代码修改** | 需要重写模型 | 零代码修改 |
| **灵活性** | 低（模型绑定） | 高（任意 HF 模型） |
| **性能** | 最优 | 良好 |
| **适用场景** | 超大规模训练 | 快速迭代 + 中等规模 |

### 9.3 实践建议

1. **默认配置**: 对于绝大多数场景，使用 FSDP2 的默认配置（基于 `_no_split_modules`）即可，无需自定义

2. **内存优化**: 如果 GPU 内存不足，考虑：
   - 启用 `--fsdp-cpu-offload`
   - 增加 `--gradient-checkpointing`
   - 增加 DP size (更多 GPU)

3. **性能优化**: 如果训练速度是瓶颈：
   - 保持 DP size 适中（4-8 通常最优）
   - 避免使用 CPU offload
   - 考虑使用 Flash Attention 3

4. **调试**: 使用 `--debug-rollout-only` 和 `--debug-train-only` 分离 FSDP 初始化和数据生成的问题

---

## 10. Source Code References

### 10.1 Key Files

1. **`slime/backends/fsdp_utils/actor.py`**:
   - `apply_fsdp2()`: Lines 1016-1058
   - `_setup_device_mesh()`: Lines 164-210
   - Model loading: Lines 85-92

2. **`slime/backends/fsdp_utils/checkpoint.py`**:
   - Checkpoint save/load with FSDP2

3. **`slime/backends/fsdp_utils/update_weight_utils.py`**:
   - Weight synchronization from FSDP2 to SGLang

### 10.2 Key Code Snippets

**Collecting modules to wrap** (actor.py:1031-1039):
```python
layer_cls_to_wrap = model._no_split_modules
assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

modules = [
    module
    for name, module in model.named_modules()
    if module.__class__.__name__ in layer_cls_to_wrap
    or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
]
```

**Hierarchical wrapping** (actor.py:1050-1055):
```python
# Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
for module in modules:
    fully_shard(module, **fsdp_kwargs)

# Apply FSDP to the top-level model
fully_shard(model, **fsdp_kwargs)
```

---

## 11. Conclusion

FSDP2 在 slime 框架中采用**分层分片（Hierarchical Sharding）**策略，对每一层 Transformer Block 单独应用 `fully_shard()`，然后对顶层模型整体应用 `fully_shard()`。这种策略通过 HuggingFace 模型的 `_no_split_modules` 属性自动控制，无需用户手动编写并行模型代码。

与 Megatron 的手动控制相比，FSDP2 提供了更高的灵活性和更低的代码复杂度，适合快速迭代和中等规模的训练场景。对于超大规模训练（>100B 参数），Megatron 的精细手动控制可能仍然是更好的选择。

**Translation**: FSDP2 in the slime framework uses a **hierarchical sharding** strategy, applying `fully_shard()` to each individual Transformer Block layer, then applying `fully_shard()` to the top-level model. This strategy is automatically controlled by the HuggingFace model's `_no_split_modules` attribute, requiring no manual parallel model code writing from users.

Compared to Megatron's manual control, FSDP2 provides higher flexibility and lower code complexity, making it suitable for rapid iteration and medium-scale training scenarios. For ultra-large-scale training (>100B parameters), Megatron's fine-grained manual control may still be the better choice.

---

**Document created**: 2025-12-03
**Framework version**: slime @ commit 9d7f34d
**Author**: Analysis based on source code examination
**Purpose**: Technical documentation for understanding FSDP2 sharding granularity in slime
