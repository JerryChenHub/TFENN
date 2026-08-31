#!/usr/bin/env bash
set -euo pipefail

command_name="${1:?usage: $0 y13|y14|y15 [runner arguments]}"
shift

python -m experiments.gnn.e311_y13_y15_pair_control_runner_v1 "${command_name}" "$@"
