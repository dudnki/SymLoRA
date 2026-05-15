#!/usr/bin/env bash
# run_all_experiments.sh
# ─────────────────────────────────────────────────────────────────
# 3일짜리 multi-seed + LoRA-FA baseline sweep.
#
# 실행:
#   nohup bash scripts/run_all_experiments.sh > logs/run_all.log 2>&1 &
#   (또는 screen/tmux 안에서 직접 실행)
#
# 특징:
#   - resumable: 이미 학습/평가된 run은 자동 skip
#   - fault-tolerant: 한 run 실패해도 다음 run 계속 진행
#   - 진행상황: logs/master.log에 기록, 각 run은 logs/<exp_id>.log
#   - 우선순위: critical(LoRA-FA) → main multi-seed
# ─────────────────────────────────────────────────────────────────

set -u  # 미정의 변수만 에러 (set -e는 의도적으로 끄지 않음)

PROJECT_DIR=/home/lami/project/SymLoRA
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

START_TS=$(date +%s)
echo "" | tee -a "$MASTER_LOG"
echo "=========================================" | tee -a "$MASTER_LOG"
echo "Sweep started at $(date)" | tee -a "$MASTER_LOG"
echo "=========================================" | tee -a "$MASTER_LOG"

# ─── 헬퍼: 1 experiment = train (필요 시) + task-specific eval (필요 시) ───
# 사용법: run_experiment <exp_id> <output_dir> <task> <train_args...>
#   task: "math" → GSM8K만 평가, "code" → HumanEval만 평가
#   (cross-task eval은 paper에 안 들어감 — 시간 낭비라서 생략)
run_experiment() {
    local exp_id="$1"
    local output_dir="$2"
    local task="$3"
    shift 3
    local train_args=("$@")

    # task → eval 종류 매핑
    local eval_arg eval_key
    case "$task" in
        math) eval_arg="gsm8k";    eval_key="gsm8k" ;;
        code) eval_arg="humaneval"; eval_key="humaneval" ;;
        *)    echo "[ERROR] unknown task: $task" | tee -a "$MASTER_LOG"; return 1 ;;
    esac

    local exp_log="$LOG_DIR/${exp_id}.log"
    local adapter_file="$output_dir/adapter_model.safetensors"
    local eval_file="$output_dir/eval_results.json"

    echo "" | tee -a "$MASTER_LOG"
    echo "[$(date +%H:%M:%S)] === ${exp_id} (eval: ${eval_arg}) ===" | tee -a "$MASTER_LOG"
    echo "[$(date +%H:%M:%S)] output_dir: $output_dir" | tee -a "$MASTER_LOG"

    # ── train ──
    if [[ -f "$adapter_file" ]]; then
        echo "[$(date +%H:%M:%S)] adapter exists → skip train" | tee -a "$MASTER_LOG"
    else
        echo "[$(date +%H:%M:%S)] training... (log: $exp_log)" | tee -a "$MASTER_LOG"
        python scripts/train_sft.py "${train_args[@]}" \
            >> "$exp_log" 2>&1
        local rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "[$(date +%H:%M:%S)] !!! TRAIN FAILED (rc=$rc): ${exp_id}" | tee -a "$MASTER_LOG"
            return 1
        fi
    fi

    # ── eval (task에 맞는 결과가 이미 있으면 skip) ──
    if [[ -f "$eval_file" ]] && python -c "
import json, sys
with open('$eval_file') as f: d = json.load(f)
sys.exit(0 if isinstance(d, dict) and '$eval_key' in d and d['$eval_key'] else 1)
" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] eval exists ($eval_key) → skip eval" | tee -a "$MASTER_LOG"
    else
        echo "[$(date +%H:%M:%S)] evaluating ($eval_arg)..." | tee -a "$MASTER_LOG"
        python scripts/eval.py --model "$output_dir" --is_adapter --task "$eval_arg" \
            >> "$exp_log" 2>&1
        local rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "[$(date +%H:%M:%S)] !!! EVAL FAILED (rc=$rc): ${exp_id}" | tee -a "$MASTER_LOG"
            return 1
        fi
    fi

    # 결과 한 줄 출력 (task에 맞는 것만)
    if [[ -f "$eval_file" ]]; then
        python -c "
import json
with open('$eval_file') as f: d = json.load(f)
g = d.get('gsm8k', {}).get('accuracy')
h = d.get('humaneval', {}).get('pass@1')
parts = []
if g is not None: parts.append(f'GSM8K={g*100:.2f}%')
if h is not None: parts.append(f'HumanEval={h*100:.2f}%')
print('   ' + '  '.join(parts) if parts else '   (no scores)')
" 2>/dev/null | tee -a "$MASTER_LOG"
    fi

    echo "[$(date +%H:%M:%S)] OK: ${exp_id}" | tee -a "$MASTER_LOG"
    return 0
}

# ─────────────────────────────────────────────────────────────────
# Phase 1: LoRA-FA baseline (paper main claim에 직접 비교 baseline)
#   = standard mode + freeze_a (A=Kaiming random frozen, B=0 학습)
#   seed 42, 123, 777 (3 seeds) × {math, code} = 6 runs
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "### Phase 1: LoRA-FA baseline ###" | tee -a "$MASTER_LOG"

for seed in 42 123 777; do
    for task in math code; do
        out="outputs/multiseed/lora_fa/${task}_seed${seed}"
        run_experiment "lora_fa_${task}_s${seed}" "$out" "$task" \
            "init.mode=standard" \
            "init.freeze_a=true" \
            "data.task=${task}" \
            "model.random_state=${seed}" \
            "training.output_dir=${out}" \
            "training.run_name=lora_fa_${task}_s${seed}"
    done
done

# ─────────────────────────────────────────────────────────────────
# Phase 2: Main comparisons multi-seed (Standard / PiSSA / Frozen-A)
#   seed 123, 777 추가 (seed 42는 기존 결과 symlink로 재사용)
#   3 conditions × 2 tasks × 2 seeds = 12 runs
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "### Phase 2: Multi-seed for Standard / PiSSA / Frozen-A ###" | tee -a "$MASTER_LOG"

for seed in 123 777; do
    for task in math code; do
        # Standard
        out="outputs/multiseed/standard/${task}_seed${seed}"
        run_experiment "standard_${task}_s${seed}" "$out" "$task" \
            "init.mode=standard" \
            "data.task=${task}" \
            "model.random_state=${seed}" \
            "training.output_dir=${out}" \
            "training.run_name=standard_${task}_s${seed}"

        # PiSSA
        out="outputs/multiseed/pissa/${task}_seed${seed}"
        run_experiment "pissa_${task}_s${seed}" "$out" "$task" \
            "init.mode=pissa" \
            "data.task=${task}" \
            "model.random_state=${seed}" \
            "training.output_dir=${out}" \
            "training.run_name=pissa_${task}_s${seed}"

        # Frozen-A (Ours) = a_only_svd + freeze_a
        out="outputs/multiseed/frozen_a/${task}_seed${seed}"
        run_experiment "frozen_a_${task}_s${seed}" "$out" "$task" \
            "init.mode=a_only_svd" \
            "init.freeze_a=true" \
            "data.task=${task}" \
            "model.random_state=${seed}" \
            "training.output_dir=${out}" \
            "training.run_name=frozen_a_${task}_s${seed}"
    done
done

# ─────────────────────────────────────────────────────────────────
# Phase 3a: A-only SVD / B-only SVD multi-seed
#   사용자 통찰 (W_residual 효과로 PiSSA가 HumanEval에서 뒤짐) 검증.
#   B-only SVD HumanEval 32.93%이 single seed인데, 이게 robust한지 확인.
#   2 methods × 2 tasks × 2 seeds = 8 runs
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "### Phase 3a: A-only / B-only SVD multi-seed ###" | tee -a "$MASTER_LOG"

for seed in 123 777; do
    for task in math code; do
        # A-only SVD
        out="outputs/multiseed/a_only_svd/${task}_seed${seed}"
        run_experiment "a_only_svd_${task}_s${seed}" "$out" "$task" \
            "init.mode=a_only_svd" \
            "data.task=${task}" \
            "model.random_state=${seed}" \
            "training.output_dir=${out}" \
            "training.run_name=a_only_svd_${task}_s${seed}"

        # B-only SVD
        out="outputs/multiseed/b_only_svd/${task}_seed${seed}"
        run_experiment "b_only_svd_${task}_s${seed}" "$out" "$task" \
            "init.mode=b_only_svd" \
            "data.task=${task}" \
            "model.random_state=${seed}" \
            "training.output_dir=${out}" \
            "training.run_name=b_only_svd_${task}_s${seed}"
    done
done

# ─────────────────────────────────────────────────────────────────
# Phase 3b: Frozen-A 추가 seeds (variance claim 강화: n=3 → n=5)
#   현재 std=1.62 → seed{1, 999} 추가하여 정확도 향상
#   1 method × 2 tasks × 2 seeds = 4 runs
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "### Phase 3b: Frozen-A extra seeds (n=3 → n=5) ###" | tee -a "$MASTER_LOG"

for seed in 1 999; do
    for task in math code; do
        out="outputs/multiseed/frozen_a/${task}_seed${seed}"
        run_experiment "frozen_a_${task}_s${seed}" "$out" "$task" \
            "init.mode=a_only_svd" \
            "init.freeze_a=true" \
            "data.task=${task}" \
            "model.random_state=${seed}" \
            "training.output_dir=${out}" \
            "training.run_name=frozen_a_${task}_s${seed}"
    done
done

# ─── 종료 ───
END_TS=$(date +%s)
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))
ELAPSED_HOUR=$(( ELAPSED_MIN / 60 ))
ELAPSED_REM=$(( ELAPSED_MIN % 60 ))

echo "" | tee -a "$MASTER_LOG"
echo "=========================================" | tee -a "$MASTER_LOG"
echo "Sweep finished at $(date)" | tee -a "$MASTER_LOG"
echo "Elapsed: ${ELAPSED_HOUR}h ${ELAPSED_REM}m" | tee -a "$MASTER_LOG"
echo "=========================================" | tee -a "$MASTER_LOG"

# 결과 요약
echo "" | tee -a "$MASTER_LOG"
echo "### Aggregated results ###" | tee -a "$MASTER_LOG"
python scripts/aggregate_results.py 2>&1 | tee -a "$MASTER_LOG" | tee "$LOG_DIR/summary.txt"

echo "" | tee -a "$MASTER_LOG"
echo "logs:" | tee -a "$MASTER_LOG"
echo "  master:  $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "  summary: $LOG_DIR/summary.txt" | tee -a "$MASTER_LOG"
echo "  per-run: $LOG_DIR/<exp_id>.log" | tee -a "$MASTER_LOG"
