#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIRECTORY}/$(basename -- "${BASH_SOURCE[0]}")"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd)"
RUNNER_MODULE="experiments.gnn.e311_gnn_y12_diagnostic_runner_v2"
MODEL_SEED="${Y_MODEL_SEED:-20260824}"
STUDY_ROOT="${Y_STUDY_ROOT:-/root/autodl-tmp/TFENN_y12_fast_v2_runs}"
COMET_PROJECT="${Y_COMET_PROJECT:-tfenn_e311_gnn_y12_diagnostic_v2}"
PYTHON_COMMAND="${Y_PYTHON:-python}"
SESSION_A="${Y_SESSION_A:-tfenn_y_a}"
SESSION_B="${Y_SESSION_B:-tfenn_y_b}"
DEPENDENCY_TIMEOUT_SECONDS="${Y_DEPENDENCY_TIMEOUT_SECONDS:-259200}"
LAUNCH_ID="${Y_LAUNCH_ID:-}"

fail() {
    printf 'launch error: %s\n' "$1" >&2
    exit 1
}

run_directory() {
    printf '%s/%s/seed_%s' "${STUDY_ROOT}" "$1" "${MODEL_SEED}"
}

resume_path() {
    printf '%s/resume_checkpoint.pt' "$(run_directory "$1")"
}

validate_complete_summary() {
    local experiment_id="$1"
    local path
    path="$(summary_path "${experiment_id}")"
    "${PYTHON_COMMAND}" - "${path}" "${experiment_id}" "${MODEL_SEED}" <<'PY'
import json
import sys
from pathlib import Path

from experiments.gnn import e311_gnn_y12_diagnostic_runner_v2 as runner

path = Path(sys.argv[1]).resolve()
experiment_id = sys.argv[2]
model_seed = int(sys.argv[3])
summary = json.loads(path.read_text(encoding="utf_8"))
target_steps = runner.target_steps_v2(experiment_id)
if summary.get("schema_name") != runner.RESULT_SCHEMA_NAME:
    raise ValueError(f"{experiment_id} result schema is invalid")
if summary.get("runner_variant") != runner.RUNNER_VARIANT:
    raise ValueError(f"{experiment_id} runner variant is invalid")
if summary.get("status") != "complete":
    raise ValueError(f"{experiment_id} summary is not complete")
experiment = summary.get("experiment", {})
if experiment.get("experiment_id") != experiment_id:
    raise ValueError(f"{experiment_id} summary identity is invalid")
if int(summary.get("model_seed", -1)) != model_seed:
    raise ValueError(f"{experiment_id} summary seed is invalid")
training = summary.get("training", {})
if int(training.get("target_steps", -1)) != target_steps:
    raise ValueError(f"{experiment_id} summary target is invalid")
if int(training.get("global_steps_completed", -1)) != target_steps:
    raise ValueError(f"{experiment_id} summary is incomplete")
run_directory = path.parent
for name in (
    "history.csv",
    "status.json",
    "best_checkpoint.pt",
    "final_checkpoint.pt",
    "resume_checkpoint.pt",
):
    if not (run_directory / name).is_file():
        raise FileNotFoundError(run_directory / name)
status = json.loads((run_directory / "status.json").read_text(encoding="utf_8"))
if status.get("status") != "complete":
    raise ValueError(f"{experiment_id} status is not complete")
if status.get("experiment_id") != experiment_id:
    raise ValueError(f"{experiment_id} status identity is invalid")
if int(status.get("model_seed", -1)) != model_seed:
    raise ValueError(f"{experiment_id} status seed is invalid")
if int(status.get("global_step", -1)) != target_steps:
    raise ValueError(f"{experiment_id} status step is invalid")
if int(status.get("target_steps", -1)) != target_steps:
    raise ValueError(f"{experiment_id} status target is invalid")
PY
}

validate_pstar() {
    local path="${STUDY_ROOT}/pstar_seed_${MODEL_SEED}.json"
    "${PYTHON_COMMAND}" - "${path}" "${MODEL_SEED}" <<'PY'
import sys
from pathlib import Path

from experiments.gnn.e311_gnn_y12_diagnostic_runner_v2 import load_pstar_protocol_v2

load_pstar_protocol_v2(
    Path(sys.argv[1]),
    expected_model_seed=int(sys.argv[2]),
)
PY
}

select_or_validate_pstar() {
    local path="${STUDY_ROOT}/pstar_seed_${MODEL_SEED}.json"
    if [[ -f "${path}" ]]; then
        validate_pstar
        printf 'reusing validated Pstar: %s\n' "${path}"
        return
    fi
    "${PYTHON_COMMAND}" -m "${RUNNER_MODULE}" select-pstar \
        --seed "${MODEL_SEED}" \
        --output-root "${STUDY_ROOT}" \
        --device cuda:0 \
        --comet-project "${COMET_PROJECT}"
}

preserve_comet_identity() {
    local experiment_id="$1"
    local source
    local audit_directory
    local destination
    source="$(run_directory "${experiment_id}")/comet.json"
    if [[ ! -f "${source}" ]]; then
        return
    fi
    audit_directory="$(run_directory "${experiment_id}")/resume_audit/${LAUNCH_ID}"
    destination="${audit_directory}/comet_before_resume.json"
    mkdir -p -- "${audit_directory}"
    [[ ! -e "${destination}" ]] \
        || fail "${experiment_id} already has a Comet resume audit for ${LAUNCH_ID}"
    cp -p -- "${source}" "${destination}"
}

run_one() {
    local experiment_id="$1"
    local output
    local path
    local resume_arguments=()
    output="$(run_directory "${experiment_id}")"
    path="$(summary_path "${experiment_id}")"
    if [[ -f "${path}" ]]; then
        validate_complete_summary "${experiment_id}"
        printf 'skipping validated complete run: %s\n' "${experiment_id}"
        return
    fi
    path="$(resume_path "${experiment_id}")"
    if [[ -f "${path}" ]]; then
        preserve_comet_identity "${experiment_id}"
        resume_arguments=(--resume)
        printf 'resuming run: %s\n' "${experiment_id}"
    elif [[ -d "${output}" ]] \
        && [[ -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        fail "${experiment_id} has artifacts but no complete summary or resume checkpoint"
    else
        printf 'starting fresh run: %s\n' "${experiment_id}"
    fi
    "${PYTHON_COMMAND}" -m "${RUNNER_MODULE}" run "${experiment_id}" \
        --seed "${MODEL_SEED}" \
        --output-root "${STUDY_ROOT}" \
        --device cuda:0 \
        --comet-project "${COMET_PROJECT}" \
        "${resume_arguments[@]}"
}

summary_path() {
    printf '%s/%s/seed_%s/summary.json' "${STUDY_ROOT}" "$1" "${MODEL_SEED}"
}

status_path() {
    printf '%s/%s/seed_%s/status.json' "${STUDY_ROOT}" "$1" "${MODEL_SEED}"
}

require_not_failed() {
    local experiment_id="$1"
    local path
    path="$(status_path "${experiment_id}")"
    if [[ -f "${path}" ]] && grep -q '"status": "failed"' "${path}"; then
        fail "${experiment_id} failed before its dependency completed"
    fi
}

worker_marker() {
    local role="$1"
    local state="$2"
    printf '%s/launcher_state/worker_%s_%s.%s' \
        "${STUDY_ROOT}" "${role}" "${LAUNCH_ID}" "${state}"
}

require_worker_available() {
    local role="$1"
    local dependency="$2"
    local failed_marker
    local done_marker
    local session_name
    failed_marker="$(worker_marker "${role}" failed)"
    done_marker="$(worker_marker "${role}" done)"
    if [[ "${role}" == "a" ]]; then
        session_name="${SESSION_A}"
    else
        session_name="${SESSION_B}"
    fi
    if [[ -f "${failed_marker}" ]]; then
        fail "worker ${role} failed while waiting for ${dependency}"
    fi
    if [[ -f "${done_marker}" ]]; then
        fail "worker ${role} ended without producing ${dependency}"
    fi
    if ! tmux has-session -t "=${session_name}" 2>/dev/null; then
        fail "worker ${role} session disappeared while waiting for ${dependency}"
    fi
}

wait_for_summary() {
    local experiment_id="$1"
    local role="$2"
    local path
    local deadline=$((SECONDS + DEPENDENCY_TIMEOUT_SECONDS))
    path="$(summary_path "${experiment_id}")"
    while [[ ! -f "${path}" ]]; do
        require_not_failed "${experiment_id}"
        require_worker_available "${role}" "${experiment_id} summary"
        if (( SECONDS >= deadline )); then
            fail "timed out waiting for ${experiment_id} summary"
        fi
        sleep 15
    done
    validate_complete_summary "${experiment_id}"
}

wait_for_pstar() {
    local path="${STUDY_ROOT}/pstar_seed_${MODEL_SEED}.json"
    local deadline=$((SECONDS + DEPENDENCY_TIMEOUT_SECONDS))
    while [[ ! -f "${path}" ]]; do
        require_not_failed Y02
        require_not_failed Y03
        require_worker_available a "Pstar selection"
        if (( SECONDS >= deadline )); then
            fail "timed out waiting for Pstar selection"
        fi
        sleep 15
    done
    validate_pstar
}

worker_a() {
    run_one Y01
    run_one Y02
    run_one Y05
    wait_for_summary Y03 b
    select_or_validate_pstar
    run_one Y06
    run_one Y07
    run_one Y08
    run_one Y12
}

worker_b() {
    run_one Y03
    run_one Y04
    wait_for_pstar
    run_one Y09
    run_one Y10
    run_one Y11
}

run_worker() {
    local role="$1"
    local failed_marker
    local done_marker
    local worker_complete=0
    [[ -n "${LAUNCH_ID}" ]] || fail "worker launch identity is missing"
    mkdir -p -- "${STUDY_ROOT}/launcher_state"
    failed_marker="$(worker_marker "${role}" failed)"
    done_marker="$(worker_marker "${role}" done)"
    trap 'worker_status=$?; if (( worker_complete == 0 )); then printf "%s\n" "${worker_status}" > "${failed_marker}"; fi' EXIT
    if [[ "${role}" == "a" ]]; then
        worker_a
    else
        worker_b
    fi
    printf '%s\n' 0 > "${done_marker}"
    worker_complete=1
    trap - EXIT
}

if [[ "${Y_WORKER_ROLE:-}" == "a" ]]; then
    run_worker a
    exit 0
fi
if [[ "${Y_WORKER_ROLE:-}" == "b" ]]; then
    run_worker b
    exit 0
fi

command -v tmux >/dev/null 2>&1 || fail "tmux is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
PYTHON_BINARY="$(command -v "${PYTHON_COMMAND}" || true)"
[[ -n "${PYTHON_BINARY}" ]] || fail "Python is unavailable"
[[ -n "${COMET_API_KEY:-}" ]] || fail "COMET_API_KEY must be exported"
[[ -n "${COMET_PROJECT}" ]] || fail "Comet project must not be empty"
[[ -f "${SCRIPT_DIRECTORY}/e311_gnn_y12_diagnostic_core_v2.py" ]] \
    || fail "the Y core is unavailable"
[[ -f "${SCRIPT_DIRECTORY}/e311_gnn_y12_diagnostic_runner_v2.py" ]] \
    || fail "the Y runner is unavailable"

for session_name in "${SESSION_A}" "${SESSION_B}"; do
    if tmux has-session -t "=${session_name}" 2>/dev/null; then
        fail "tmux session already exists: ${session_name}"
    fi
done

mkdir -p -- "${STUDY_ROOT}/launcher_logs"
export PYTHONPATH="${REPOSITORY_ROOT}/src:${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
preflight_path="${STUDY_ROOT}/launcher_logs/preflight_${LAUNCH_ID}.json"
[[ ! -e "${preflight_path}" ]] \
    || fail "preflight record already exists for ${LAUNCH_ID}"
"${PYTHON_BINARY}" -m "${RUNNER_MODULE}" preflight \
    --output-root "${STUDY_ROOT}" \
    --device cuda:0 \
    --disable-comet \
    > "${preflight_path}"

log_a="${STUDY_ROOT}/launcher_logs/session_a_${LAUNCH_ID}.log"
log_b="${STUDY_ROOT}/launcher_logs/session_b_${LAUNCH_ID}.log"

created_sessions=()
cleanup_failed_launch() {
    local session_name
    for session_name in "${created_sessions[@]}"; do
        tmux kill-session -t "=${session_name}" 2>/dev/null || true
    done
}
trap cleanup_failed_launch ERR

create_worker_session() {
    local session_name="$1"
    local role="$2"
    local log_path="$3"
    local command_text

    tmux new-session -d -s "${session_name}" -n bootstrap \
        -c "${REPOSITORY_ROOT}" 'exec sleep infinity'
    created_sessions+=("${session_name}")
    tmux set-environment -t "=${session_name}" COMET_API_KEY "${COMET_API_KEY}"
    tmux set-environment -t "=${session_name}" Y_WORKER_ROLE "${role}"
    tmux set-environment -t "=${session_name}" Y_MODEL_SEED "${MODEL_SEED}"
    tmux set-environment -t "=${session_name}" Y_STUDY_ROOT "${STUDY_ROOT}"
    tmux set-environment -t "=${session_name}" Y_COMET_PROJECT "${COMET_PROJECT}"
    tmux set-environment -t "=${session_name}" Y_PYTHON "${PYTHON_BINARY}"
    tmux set-environment -t "=${session_name}" Y_LAUNCH_ID "${LAUNCH_ID}"
    tmux set-environment -t "=${session_name}" Y_DEPENDENCY_TIMEOUT_SECONDS \
        "${DEPENDENCY_TIMEOUT_SECONDS}"
    tmux set-environment -t "=${session_name}" PYTHONPATH "${PYTHONPATH}"
    tmux set-environment -t "=${session_name}" PYTHONHASHSEED 0
    tmux set-environment -t "=${session_name}" CUBLAS_WORKSPACE_CONFIG :4096:8
    tmux set-environment -t "=${session_name}" CUDA_VISIBLE_DEVICES 0
    printf -v command_text 'exec bash %q >> %q 2>&1' "${SCRIPT_PATH}" "${log_path}"
    tmux new-window -d -t "=${session_name}" -n run \
        -c "${REPOSITORY_ROOT}" "${command_text}"
    tmux kill-window -t "=${session_name}:bootstrap"
}

create_worker_session "${SESSION_A}" a "${log_a}"
create_worker_session "${SESSION_B}" b "${log_b}"

trap - ERR
unset COMET_API_KEY

printf 'session a: %s\n' "${SESSION_A}"
printf 'session b: %s\n' "${SESSION_B}"
printf 'study root: %s\n' "${STUDY_ROOT}"
printf 'session a log: %s\n' "${log_a}"
printf 'session b log: %s\n' "${log_b}"
