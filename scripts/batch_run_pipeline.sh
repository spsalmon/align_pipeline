#!/bin/bash

# Submit one pipeline run per experiment, all sharing the same config. Each
# experiment goes through run_pipeline.sh with -e, which overrides the config's
# experiment_dir, so every run is validated and submitted independently (and gets
# its own temp_files/pipeline_<start_time>_<job_id> dir).
#
# Usage:
#   bash scripts/run_pipeline_batch.sh -c config.yaml /path/to/exp1 /path/to/exp2 ...
#   bash scripts/run_pipeline_batch.sh -c config.yaml -f experiments.txt
#
# The -f list file holds one experiment dir per line; blank lines and lines
# starting with # are ignored. -f and positional experiments can be combined.

usage() {
    echo "Usage: $0 -c <config.yaml> [-f <experiments.txt>] [experiment_dir ...]" >&2
    exit 1
}

CONFIG_FILE=""
EXPERIMENTS=()
while (( $# > 0 )); do
    case "$1" in
        -c|--config)
            [ -n "$2" ] || usage
            CONFIG_FILE="$2"
            shift 2
            ;;
        -f|--experiment_list)
            [ -n "$2" ] || usage
            if [ ! -f "$2" ]; then
                echo "Error: experiment list '$2' not found" >&2
                exit 1
            fi
            while IFS= read -r line || [ -n "$line" ]; do
                line="${line%%#*}"
                line="$(echo "$line" | xargs)"
                [ -n "$line" ] && EXPERIMENTS+=("$line")
            done < "$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            EXPERIMENTS+=("$1")
            shift
            ;;
    esac
done

[ -n "$CONFIG_FILE" ] || usage
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config '$CONFIG_FILE' not found" >&2
    exit 1
fi
if (( ${#EXPERIMENTS[@]} == 0 )); then
    echo "Error: no experiments given" >&2
    usage
fi

# run_pipeline.sh cds to the repo root, so hand it absolute paths.
CONFIG_FILE="$(realpath "$CONFIG_FILE")"
RUN_PIPELINE="$(dirname "$(realpath "$0")")/run_pipeline.sh"

FAILED=()
for experiment in "${EXPERIMENTS[@]}"; do
    echo "=== $experiment ==="
    if [ ! -d "$experiment" ]; then
        echo "Error: experiment dir '$experiment' not found, skipping" >&2
        FAILED+=("$experiment")
        continue
    fi
    if ! bash "$RUN_PIPELINE" -c "$CONFIG_FILE" -e "$(realpath "$experiment")"; then
        FAILED+=("$experiment")
    fi
done

echo
echo "Submitted $(( ${#EXPERIMENTS[@]} - ${#FAILED[@]} ))/${#EXPERIMENTS[@]} experiments."
if (( ${#FAILED[@]} > 0 )); then
    echo "Failed:" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
