#!/bin/bash

# Run everything from the repo root, regardless of where this is called from.
cd "$(dirname "$0")/.."

# Landing zone for the outer job's #SBATCH -o/-e, which sbatch resolves at submit
# time and will not create. The pipeline moves the logs into its run dir.
mkdir -p sbatch_output

# Find the config among the forwarded arguments (same default as the pipeline),
# so we can derive the outer job's sbatch resources from it. Read by
# index rather than shift, so "$@" stays intact for forwarding below.
CONFIG_FILE="./align_pipeline/defaults/configs/config.yaml"
EXPERIMENT_DIR=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        -c|--config)
            if (( i + 1 >= ${#args[@]} )); then
                echo "Error: ${args[$i]} requires a value" >&2
                exit 1
            fi
            CONFIG_FILE="${args[$((i + 1))]}"
            ;;
        -e|--experiment_dir)
            if (( i + 1 >= ${#args[@]} )); then
                echo "Error: ${args[$i]} requires a value" >&2
                exit 1
            fi
            EXPERIMENT_DIR="${args[$((i + 1))]}"
            ;;
    esac
done

# Resolve the python launcher for the pre-flight and the submitted job: an
# exported ALIGN_PYTHON wins, else the config's python_command, else the
# micromamba default. Grepped (not read via python) because python is the very
# thing we are resolving. Exported so the sbatch job inherits the same launcher.
CONFIG_PYTHON=$(grep -E '^[[:space:]]*python_command:' "$CONFIG_FILE" 2>/dev/null | head -1 | sed -E 's/^[^:]*:[[:space:]]*//; s/[[:space:]]*(#.*)?$//; s/^["'\'']//; s/["'\'']$//')
export ALIGN_PYTHON="${ALIGN_PYTHON:-${CONFIG_PYTHON:-$HOME/.local/bin/micromamba run -n align_pipeline python3}}"

# Validate the config here on the login node so a bad one fails fast with the
# error on the terminal, instead of costing a job submission and surfacing in the
# outer job's slurm .err. Pass -e through so validation sees the same experiment
# dir the run will. The pipeline still validates again inside the job.
VALIDATE_ARGS=(-c "$CONFIG_FILE")
[ -n "$EXPERIMENT_DIR" ] && VALIDATE_ARGS+=(-e "$EXPERIMENT_DIR")
if ! $ALIGN_PYTHON -m align_pipeline.run_params --validate "${VALIDATE_ARGS[@]}"; then
    exit 1
fi

# Config-drive the outer job's sbatch resources from sbatch_init. If this yields
# nothing (e.g. non-slurm config), sbatch falls back to the header in the script.
SBATCH_INIT_FLAGS=$($ALIGN_PYTHON -m align_pipeline.run_params --sbatch-init -c "$CONFIG_FILE" 2>/dev/null)

# Pass the resource flags and the forwarded arguments to the SBATCH script.
sbatch $SBATCH_INIT_FLAGS scripts/_init_pipeline.sh "$@"
