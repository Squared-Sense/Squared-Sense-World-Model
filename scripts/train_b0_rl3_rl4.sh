#!/usr/bin/env bash
# Trains B0 (b0_blind) and RL3/RL4 Dead-Reckoning PPO policies (with optional post-train eval).
#
# Usage:
#   ./scripts/train_b0_rl3_rl4.sh                              # Trains b0_blind on rl1 (50K steps) + eval
#   MODEL_TYPE=m1_full TASK=rl4_flash ./scripts/train_b0_rl3_rl4.sh
#   RUN_SUITE=1 ./scripts/train_b0_rl3_rl4.sh                  # Full B0 + M1/M2 RL3/RL4 suite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

REGION_DIR="${REGION_DIR:-data/baked_regions}"
MODEL_TYPE="${MODEL_TYPE:-b0_blind}"
TASK="${TASK:-rl1}"
CKPT="${CKPT:-}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-}"
INIT_POLICY="${INIT_POLICY:-}"
FLASH_COMPASS_TICKS="${FLASH_COMPASS_TICKS:--1}"
BC_EPOCHS="${BC_EPOCHS:--1}"
BC_STEPS="${BC_STEPS:-600}"
LR="${LR:--1}"
TOTAL_STEPS="${TOTAL_STEPS:--1}"
SAVE_DIR="${SAVE_DIR:-}"
DEVICE="${DEVICE:-auto}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-1000}"
RUN_SUITE="${RUN_SUITE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

run_train_and_eval() {
    local mtype="$1"
    local task_name="$2"
    local mtag="$mtype"
    case "$mtype" in
        b0_blind) mtag="b0" ;;
        m1_full) mtag="m1" ;;
        m2_static) mtag="m2" ;;
    esac

    local resolved_ckpt="$CKPT"
    if [[ "$mtype" != "b0_blind" && -z "$resolved_ckpt" ]]; then
        if [[ "$mtype" == "m1_full" && -f "checkpoints/spatial_finetuned_best.pkl" ]]; then
            resolved_ckpt="checkpoints/spatial_finetuned_best.pkl"
        elif [[ "$mtype" == "m2_static" && -f "checkpoints/m2_spatial_jepa/best.pkl" ]]; then
            resolved_ckpt="checkpoints/m2_spatial_jepa/best.pkl"
        fi
    fi

    local resolved_temp="$TEMPORAL_CKPT"
    if [[ "$mtype" == "m1_full" && -z "$resolved_temp" && -f "checkpoints/temporal_best.pkl" ]]; then
        resolved_temp="checkpoints/temporal_best.pkl"
    fi

    local resolved_init="$INIT_POLICY"
    if [[ -z "$resolved_init" && "$task_name" != "rl1" ]]; then
        if [[ "$mtype" == "m1_full" ]]; then
            for c in "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl" "checkpoints/ppo_nav_m1/best_ppo.pkl"; do
                [[ -f "$c" ]] && { resolved_init="$c"; break; }
            done
        elif [[ "$mtype" == "m2_static" ]]; then
            for c in "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl" "checkpoints/ppo_nav_m2/best_ppo.pkl"; do
                [[ -f "$c" ]] && { resolved_init="$c"; break; }
            done
        elif [[ "$mtype" == "b0_blind" ]]; then
            for c in "checkpoints/ppo_nav_b0_rl1/best_ppo.pkl" "checkpoints/ppo_nav_b0/best_ppo.pkl"; do
                [[ -f "$c" ]] && { resolved_init="$c"; break; }
            done
        fi
    fi

    local resolved_save="${SAVE_DIR:-checkpoints/ppo_nav_${mtag}_${task_name}}"
    local resolved_steps="$TOTAL_STEPS"
    if [[ "$resolved_steps" -le 0 ]]; then
        if [[ "$task_name" == "rl1" || "$mtype" == "b0_blind" ]]; then
            resolved_steps=50000
        else
            resolved_steps=25000
        fi
    fi

    local resolved_bc="$BC_EPOCHS"
    if [[ "$resolved_bc" -lt 0 ]]; then
        [[ -n "$resolved_init" ]] && resolved_bc=0 || resolved_bc=20
    fi

    local resolved_lr="$LR"
    if [[ "$resolved_lr" == "-1" ]]; then
        [[ -n "$resolved_init" ]] && resolved_lr="1e-4" || resolved_lr="3e-4"
    fi

    echo "======================================================="
    echo " Squared Sense: Train $mtype on $task_name"
    echo "======================================================="

    local args=(
        run python -m eval.train_ppo_nav
        --region_dir "$REGION_DIR"
        --model_type "$mtype"
        --task "$task_name"
        --bc_epochs "$resolved_bc"
        --bc_steps "$BC_STEPS"
        --lr "$resolved_lr"
        --total_steps "$resolved_steps"
        --save_dir "$resolved_save"
        --device "$DEVICE"
    )
    [[ -n "$resolved_ckpt" ]] && args+=(--ckpt "$resolved_ckpt")
    [[ "$mtype" == "m1_full" && -n "$resolved_temp" ]] && args+=(--temporal_ckpt "$resolved_temp")
    [[ -n "$resolved_init" && -f "$resolved_init" ]] && args+=(--init_policy "$resolved_init")
    [[ "$FLASH_COMPASS_TICKS" -ge 0 ]] && args+=(--flash_compass_ticks "$FLASH_COMPASS_TICKS")

    uv "${args[@]}"

    if [[ "$SKIP_EVAL" != "1" ]]; then
        local best_pol="$resolved_save/best_ppo.pkl"
        [[ ! -f "$best_pol" ]] && best_pol="$resolved_save/latest_ppo.pkl"
        local out_json="results/${mtag}_${task_name}.json"

        local eval_args=(
            run python -m eval.eval_navigation
            --region_dir "$REGION_DIR"
            --model_type "$mtype"
            --policy_ckpt "$best_pol"
            --task "$task_name"
            --num_episodes "$EVAL_EPISODES"
            --seed "$SEED"
            --out_json "$out_json"
        )
        [[ -n "$resolved_ckpt" ]] && eval_args+=(--backbone_ckpt "$resolved_ckpt")
        [[ "$mtype" == "m1_full" && -n "$resolved_temp" ]] && eval_args+=(--temporal_ckpt "$resolved_temp")
        [[ "$FLASH_COMPASS_TICKS" -ge 0 ]] && eval_args+=(--flash_compass_ticks "$FLASH_COMPASS_TICKS")

        uv "${eval_args[@]}"
    fi
}

if [[ "$RUN_SUITE" == "1" ]]; then
    run_train_and_eval "b0_blind" "rl1"
    for m in m1_full m2_static; do
        for t in rl3_beacon rl3_flash rl4_beacon rl4_flash; do
            run_train_and_eval "$m" "$t"
        done
    done
else
    run_train_and_eval "$MODEL_TYPE" "$TASK"
fi
