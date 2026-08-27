#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd)"

STUDY_ROOT="${1:-${E311_STUDY_ROOT:-/root/autodl-tmp/TFENN_e311_12_runs_v1}}"
COMET_PROJECT="${2:-${E311_COMET_PROJECT:-tfenn_e311_gnn_12_v1}}"
TMUX_SESSION="${E311_TMUX_SESSION:-tfenn_e311_12_v1}"
PYTHON_COMMAND="${E311_PYTHON:-python}"
RUNNER_MODULE="experiments.gnn.e311_gnn_12_experiment_runner_v1"

SEEDS=(20260824)
GROUP_A=(X01 X02 X03 X04 X08 X11 X12)
GROUP_B=(X05 X06 X07 X09 X10)

fail() {
    printf 'launch error: %s\n' "$1" >&2
    exit 1
}

command -v tmux >/dev/null 2>&1 || fail "tmux is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
PYTHON_BINARY="$(command -v "${PYTHON_COMMAND}" || true)"
[[ -n "${PYTHON_BINARY}" ]] || fail "Python command is unavailable: ${PYTHON_COMMAND}"
[[ -n "${COMET_API_KEY:-}" ]] || fail "COMET_API_KEY must already be exported"
[[ -n "${COMET_PROJECT}" ]] || fail "Comet project must not be empty"
[[ -f "${REPOSITORY_ROOT}/experiments/gnn/e311_gnn_12_experiment_runner_v1.py" ]] \
    || fail "the E311 experiment runner is unavailable"

"${PYTHON_BINARY}" -c \
    'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
    || fail "the selected Python environment cannot use CUDA"

GPU_IDENTIFIERS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "all" ]]; then
    IFS=',' read -r -a REQUESTED_GPU_IDENTIFIERS <<< "${CUDA_VISIBLE_DEVICES}"
    for identifier in "${REQUESTED_GPU_IDENTIFIERS[@]}"; do
        identifier="${identifier//[[:space:]]/}"
        if [[ -n "${identifier}" && "${identifier}" != "-1" && "${identifier}" != "NoDevFiles" ]]; then
            GPU_IDENTIFIERS+=("${identifier}")
        fi
    done
fi
if (( ${#GPU_IDENTIFIERS[@]} == 0 )); then
    mapfile -t GPU_IDENTIFIERS < <(
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
            | sed '/^[[:space:]]*$/d' \
            | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
    )
fi
(( ${#GPU_IDENTIFIERS[@]} >= 1 )) || fail "no visible GPU was detected"

GROUP_A_GPU="${GPU_IDENTIFIERS[0]}"
if (( ${#GPU_IDENTIFIERS[@]} >= 2 )); then
    GROUP_B_GPU="${GPU_IDENTIFIERS[1]}"
else
    GROUP_B_GPU="${GPU_IDENTIFIERS[0]}"
fi

if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    fail "tmux session already exists: ${TMUX_SESSION}"
fi

mkdir -p -- "${STUDY_ROOT}/launcher_logs"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
GROUP_A_LOG="${STUDY_ROOT}/launcher_logs/group_a_${LAUNCH_ID}.log"
GROUP_B_LOG="${STUDY_ROOT}/launcher_logs/group_b_${LAUNCH_ID}.log"

shell_join() {
    local rendered
    printf -v rendered '%q ' "$@"
    printf '%s' "${rendered% }"
}

build_window_command() {
    local gpu_identifier="$1"
    local log_path="$2"
    shift 2
    local experiments=("$@")
    local invocation
    local repository
    local visible_gpu
    local log_file

    invocation="$(shell_join \
        "${PYTHON_BINARY}" \
        -m "${RUNNER_MODULE}" \
        run-group \
        --experiments "${experiments[@]}" \
        --seeds "${SEEDS[@]}" \
        --study-root "${STUDY_ROOT}" \
        --epochs 500 \
        --batch-size 100 \
        --five-benzene-batch-size 128 \
        --device cuda:0 \
        --comet-project "${COMET_PROJECT}"
    )"
    repository="$(shell_join "${REPOSITORY_ROOT}")"
    visible_gpu="$(shell_join "${gpu_identifier}")"
    log_file="$(shell_join "${log_path}")"

    printf '%s' \
        "set -o pipefail; cd ${repository}; export PYTHONUNBUFFERED=1; " \
        "export PYTHONPATH=${repository}/src:${repository}\${PYTHONPATH:+:\${PYTHONPATH}}; " \
        'export PYTHONHASHSEED=0; export CUBLAS_WORKSPACE_CONFIG=:4096:8; ' \
        "export CUDA_VISIBLE_DEVICES=${visible_gpu}; " \
        "${invocation} 2>&1 | tee -a ${log_file}; " \
        'status=${PIPESTATUS[0]}; ' \
        "printf '\\nrunner exit status: %s\\n' \"\${status}\" | tee -a ${log_file}; " \
        'exec "${SHELL:-/bin/bash}"'
}

GROUP_A_COMMAND="$(build_window_command "${GROUP_A_GPU}" "${GROUP_A_LOG}" "${GROUP_A[@]}")"
GROUP_B_COMMAND="$(build_window_command "${GROUP_B_GPU}" "${GROUP_B_LOG}" "${GROUP_B[@]}")"

SESSION_CREATED=0
cleanup_failed_launch() {
    if (( SESSION_CREATED == 1 )); then
        tmux kill-session -t "=${TMUX_SESSION}" 2>/dev/null || true
    fi
}
trap cleanup_failed_launch ERR

tmux new-session \
    -d \
    -s "${TMUX_SESSION}" \
    -n bootstrap \
    -c "${REPOSITORY_ROOT}" \
    'exec sleep infinity'
SESSION_CREATED=1
tmux set-environment -t "=${TMUX_SESSION}" COMET_API_KEY "${COMET_API_KEY}"
tmux new-window \
    -d \
    -t "=${TMUX_SESSION}" \
    -n group_a \
    -c "${REPOSITORY_ROOT}" \
    "${GROUP_A_COMMAND}"
tmux new-window \
    -d \
    -t "=${TMUX_SESSION}" \
    -n group_b \
    -c "${REPOSITORY_ROOT}" \
    "${GROUP_B_COMMAND}"
tmux kill-window -t "=${TMUX_SESSION}:bootstrap"

trap - ERR

printf 'tmux session: %s\n' "${TMUX_SESSION}"
printf 'group_a visible GPU: %s\n' "${GROUP_A_GPU}"
printf 'group_b visible GPU: %s\n' "${GROUP_B_GPU}"
printf 'study root: %s\n' "${STUDY_ROOT}"
printf 'group_a log: %s\n' "${GROUP_A_LOG}"
printf 'group_b log: %s\n' "${GROUP_B_LOG}"
printf 'inspect: tmux list-windows -t %s\n' "${TMUX_SESSION}"
