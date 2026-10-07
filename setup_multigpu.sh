#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
GPU_POWER_LIMIT_WATTS=200
API_KEY_FILE="${STRATA_API_KEY_FILE:-${HOME}/.config/strata/api-key}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--gpus GPU_LIST] [--host ADDRESS] [--api-key KEY]

Start one Strata instance per selected GPU. GPU_LIST is a comma-separated
list of GPU numbers from 0 through 3 (for example, --gpus 0,1).
Without --gpus, all four GPUs are started.
Multiple instances get disjoint physical CPU cores, including their SMT siblings.
A single instance keeps the launcher's available CPU cores.

The default host is 0.0.0.0 (LAN and Tailscale access). Set --api-key KEY or
STRATA_API_KEY, or let this script create a key in ${API_KEY_FILE}.
Use --host 127.0.0.1 to limit access to this PC.
EOF
}

gpu_list="0,1,2,3"
host="0.0.0.0"
api_key="${STRATA_API_KEY:-}"
while (( $# > 0 )); do
    case "$1" in
        --gpus)
            if (( $# < 2 )); then
                echo "--gpus requires a comma-separated GPU list, for example --gpus 0,1." >&2
                usage >&2
                exit 2
            fi
            gpu_list="$2"
            shift 2
            ;;
        --gpus=*)
            gpu_list="${1#*=}"
            shift
            ;;
        --host)
            if (( $# < 2 )) || [[ -z "$2" ]]; then
                echo "--host requires an address, for example --host 0.0.0.0." >&2
                usage >&2
                exit 2
            fi
            host="$2"
            shift 2
            ;;
        --host=*)
            host="${1#*=}"
            if [[ -z "${host}" ]]; then
                echo "--host requires an address, for example --host 0.0.0.0." >&2
                exit 2
            fi
            shift
            ;;
        --api-key)
            if (( $# < 2 )) || [[ -z "$2" ]]; then
                echo "--api-key requires a non-empty key." >&2
                usage >&2
                exit 2
            fi
            api_key="$2"
            shift 2
            ;;
        --api-key=*)
            api_key="${1#*=}"
            if [[ -z "${api_key}" ]]; then
                echo "--api-key requires a non-empty key." >&2
                exit 2
            fi
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "${gpu_list}" =~ ^[0-3](,[0-3])*$ ]]; then
    echo "--gpus must be a comma-separated list of GPU numbers from 0 through 3, for example --gpus 0,1." >&2
    exit 2
fi

IFS=',' read -r -a GPUS <<<"${gpu_list}"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "--gpus contains GPU ${gpu} more than once." >&2
        exit 2
    fi
    SEEN_GPUS[${gpu}]=1
done
GPU_COUNT=${#GPUS[@]}
if (( GPU_COUNT == 1 )); then
    INSTANCE_DESCRIPTION="the selected Strata instance"
    INSTANCE_READY_VERB="is"
else
    INSTANCE_DESCRIPTION="all ${GPU_COUNT} selected Strata instances"
    INSTANCE_READY_VERB="are"
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Strata's Python environment was not found at ${PYTHON}." >&2
    echo "Run ./setup.sh --no-start first." >&2
    exit 1
fi

if ! command -v taskset >/dev/null 2>&1; then
    echo "taskset is required to isolate each Strata instance's CPU workers." >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to keep concurrent launches from sharing CPU cores." >&2
    exit 1
fi
exec 9>"${ROOT}/.strata-multigpu.lock"
if ! flock --nonblock 9; then
    echo "Another Strata launcher is assigning CPU cores; wait for its startup to finish." >&2
    exit 1
fi

declare -a CPU_PLAN_ARGS=("${GPU_COUNT}")
for gpu in 0 1 2 3; do
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        continue
    fi
    pid_file="${ROOT}/strata-iq3_s_gpu${gpu}.pid"
    if [[ -f "${pid_file}" ]]; then
        pid="$(<"${pid_file}")"
        if [[ "${pid}" =~ ^[0-9]+$ && -r "/proc/${pid}/cmdline" ]]; then
            cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
            if [[ "${cmdline}" == *"serve/server.py"* &&
                  "${cmdline}" == *"${ROOT}/strata-iq3_s_gpu${gpu}.json"* ]]; then
                CPU_PLAN_ARGS+=(--exclude-pid "${pid}")
            fi
        fi
    fi
done
if ! cpu_plan="$("${PYTHON}" "${ROOT}/serve/cpu_affinity.py" "${CPU_PLAN_ARGS[@]}")"; then
    echo "Could not assign independent CPU cores; no servers were stopped or started." >&2
    exit 1
fi
mapfile -t CPU_SETS <<<"${cpu_plan}"

if [[ -z "${api_key}" && -r "${API_KEY_FILE}" ]]; then
    IFS= read -r api_key <"${API_KEY_FILE}"
fi
if [[ -z "${api_key}" ]]; then
    api_key="$("${PYTHON}" -c 'import secrets; print(secrets.token_hex(32))')"
fi
if [[ -z "${api_key}" || "${api_key}" == *$'\n'* || "${api_key}" == *$'\r'* ]]; then
    echo "The Strata API key must be non-empty and contain no newlines." >&2
    exit 2
fi

api_key_dir="$(dirname -- "${API_KEY_FILE}")"
mkdir -p -- "${api_key_dir}"
(umask 077; printf '%s\n' "${api_key}" >"${API_KEY_FILE}")

declare -a PIDS
declare -a READY
declare -a LOG_FILES
declare -a SERVER_NETWORK_ARGS=(--host "${host}" --api-key "${api_key}")
TAIL_PID=""

stop_log_tail() {
    if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
        kill "${TAIL_PID}" 2>/dev/null || true
        wait "${TAIL_PID}" 2>/dev/null || true
    fi
}

trap stop_log_tail EXIT

for gpu in "${GPUS[@]}"; do
    config="${ROOT}/strata-iq3_s_gpu${gpu}.json"
    pid_file="${ROOT}/strata-iq3_s_gpu${gpu}.pid"

    if [[ ! -f "${config}" ]]; then
        echo "Missing configuration: ${config}" >&2
        exit 1
    fi

    if [[ -f "${pid_file}" ]]; then
        pid="$(<"${pid_file}")"
        if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
            cmdline=""
            if [[ -r "/proc/${pid}/cmdline" ]]; then
                cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
            fi

            if [[ "${cmdline}" != *"serve/server.py"* || "${cmdline}" != *"${config}"* ]]; then
                echo "Refusing to stop PID ${pid}: it is not the expected GPU ${gpu} Strata process." >&2
                echo "Remove ${pid_file} after checking that process manually." >&2
                exit 1
            fi

            echo "GPU ${gpu}: stopping existing Strata PID ${pid}"
            kill "${pid}"

            for _ in {1..30}; do
                if ! kill -0 "${pid}" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done

            if kill -0 "${pid}" 2>/dev/null; then
                echo "GPU ${gpu}: PID ${pid} did not stop cleanly; forcing it to stop"
                kill -KILL "${pid}"
                for _ in {1..10}; do
                    if ! kill -0 "${pid}" 2>/dev/null; then
                        break
                    fi
                    sleep 0.2
                done
            fi

            if kill -0 "${pid}" 2>/dev/null; then
                echo "GPU ${gpu}: PID ${pid} is still present after SIGKILL." >&2
                exit 1
            fi
        fi
        rm -f -- "${pid_file}"
    fi

    port=$((8080 + gpu))
    if ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .; then
        echo "Port ${port} is already in use and no running Strata PID was recorded." >&2
        exit 1
    fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to set each GPU's power limit." >&2
    exit 1
fi

declare -a NVIDIA_SMI=(nvidia-smi)
if (( EUID != 0 )); then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "sudo is required to set each GPU's power limit." >&2
        exit 1
    fi
    NVIDIA_SMI=(sudo nvidia-smi)
fi

echo "Setting power limits on GPUs ${gpu_list} to ${GPU_POWER_LIMIT_WATTS} W..."
for gpu in "${GPUS[@]}"; do
    if ! "${NVIDIA_SMI[@]}" --id="${gpu}" --power-limit="${GPU_POWER_LIMIT_WATTS}"; then
        echo "GPU ${gpu}: could not set the power limit to ${GPU_POWER_LIMIT_WATTS} W; no servers were started." >&2
        exit 1
    fi
done

for index in "${!GPUS[@]}"; do
    gpu="${GPUS[${index}]}"
    port=$((8080 + gpu))
    config="${ROOT}/strata-iq3_s_gpu${gpu}.json"
    output="${ROOT}/strata-iq3_s_gpu${gpu}.out"
    engine_log="${ROOT}/strata-iq3_s-gpu${gpu}.log"
    pid_file="${ROOT}/strata-iq3_s_gpu${gpu}.pid"

    touch "${output}" "${engine_log}"

    nohup taskset --cpu-list "${CPU_SETS[${index}]}" "${PYTHON}" "${ROOT}/serve/server.py" \
        --engine strata \
        --config "${config}" \
        --port "${port}" \
        "${SERVER_NETWORK_ARGS[@]}" \
        9>&- >"${output}" 2>&1 &

    PIDS[${gpu}]=$!
    printf '%s\n' "${PIDS[${gpu}]}" >"${pid_file}"
    echo "GPU ${gpu}: started PID ${PIDS[${gpu}]} on port ${port}; CPUs ${CPU_SETS[${index}]}"
done
flock --unlock 9
exec 9>&-

for gpu in "${GPUS[@]}"; do
    LOG_FILES+=(
        "${ROOT}/strata-iq3_s_gpu${gpu}.out"
        "${ROOT}/strata-iq3_s-gpu${gpu}.log"
    )
done

echo "Streaming server and engine logs; press Ctrl-C to stop watching."
tail -n 20 -F "${LOG_FILES[@]}" &
TAIL_PID=$!

echo "Waiting for ${INSTANCE_DESCRIPTION} to become ready..."
if [[ "${host}" == "0.0.0.0" || "${host}" == "::" ]]; then
    remote_host="<this PC's LAN or Tailscale IP>"
elif [[ "${host}" != "127.0.0.1" && "${host}" != "localhost" && "${host}" != "::1" ]]; then
    remote_host="${host}"
else
    remote_host=""
fi
if [[ -n "${remote_host}" ]]; then
    for gpu in "${GPUS[@]}"; do
        echo "GPU ${gpu} remote API: http://${remote_host}:$((8080 + gpu))/v1 (API key required)"
    done
fi
deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))

while (( SECONDS < deadline )); do
    ready_count=0

    for gpu in "${GPUS[@]}"; do
        if [[ "${READY[${gpu}]:-}" == "1" ]]; then
            ((ready_count += 1))
            continue
        fi

        pid="${PIDS[${gpu}]}"
        port=$((8080 + gpu))

        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "GPU ${gpu}: Strata exited during startup; see strata-iq3_s_gpu${gpu}.out" >&2
            rm -f -- "${ROOT}/strata-iq3_s_gpu${gpu}.pid"
            exit 1
        fi

        if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            READY[${gpu}]=1
            ((ready_count += 1))
            echo "GPU ${gpu}: ready on port ${port}"
        fi
    done

    if (( ready_count == GPU_COUNT )); then
        echo "${INSTANCE_DESCRIPTION^} ${INSTANCE_READY_VERB} ready."
        echo "Continuing to stream logs; the Strata processes remain running after Ctrl-C."
        wait "${TAIL_PID}"
        exit $?
    fi

    sleep 2
done

echo "Timed out after ${STARTUP_TIMEOUT_SECONDS}s waiting for all instances." >&2
echo "The surviving processes were left running; inspect strata-iq3_s_gpu*.out." >&2
exit 1
