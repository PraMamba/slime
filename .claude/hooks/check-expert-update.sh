#!/bin/bash
# Hook script to remind updating expert agents when related code changes
# Called by Claude Code PostToolUse hook

# Check if jq is available
if ! command -v jq &> /dev/null; then
    exit 0
fi

# Read JSON input from stdin
INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Define mappings: code path pattern -> expert agent file
check_expert_update() {
    local file="$1"
    local reminder_file=""
    local reminder_desc=""

    # Megatron Backend (loss, model, CP, data, checkpoint)
    if [[ "$file" == *"slime/backends/megatron_utils/loss"* ]] || \
       [[ "$file" == *"slime/backends/megatron_utils/actor"* ]] || \
       [[ "$file" == *"slime/backends/megatron_utils/model"* ]] || \
       [[ "$file" == *"slime/backends/megatron_utils/cp_utils"* ]] || \
       [[ "$file" == *"slime/backends/megatron_utils/data"* ]] || \
       [[ "$file" == *"slime/backends/megatron_utils/checkpoint"* ]] || \
       [[ "$file" == *"slime/backends/megatron_utils/initialize"* ]]; then
        reminder_file="megatron-expert.md"
        reminder_desc="Megatron Backend"
    fi

    # Weight Sync (converters, NCCL updates)
    if [[ "$file" == *"slime/backends/megatron_utils/update_weight/"* ]] || \
       [[ "$file" == *"slime/backends/megatron_utils/megatron_to_hf/"* ]] || \
       [[ "$file" == *"slime_plugins/mbridge/"* ]]; then
        reminder_file="weight-sync-expert.md"
        reminder_desc="Weight Synchronization"
    fi

    # Rollout & SGLang (generation, reward, engines, config)
    if [[ "$file" == *"slime/ray/rollout"* ]] || \
       [[ "$file" == *"slime/rollout/"* ]] || \
       [[ "$file" == *"slime/backends/sglang_utils/"* ]]; then
        reminder_file="rollout-expert.md"
        reminder_desc="Rollout/SGLang"
    fi

    # Algorithm (PPO utils, advantage estimation)
    if [[ "$file" == *"slime/utils/ppo_utils"* ]] || \
       [[ "$file" == *"slime/backends/megatron_utils/loss"* ]] || \
       [[ "$file" == *"slime/rollout/rm_hub/"* ]]; then
        reminder_file="algorithm-expert.md"
        reminder_desc="Algorithm/Reward"
    fi

    # Output reminder if matched
    if [ -n "$reminder_file" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📝 Expert Update Reminder"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Modified: $file"
        echo "Consider updating: .claude/agents/$reminder_file ($reminder_desc)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
}

check_expert_update "$FILE_PATH"
