#!/bin/bash

# Delete the raw and analysis images of the frames flagged Ignore in the GUI and,
# with MAX_TIME_AFTER_M4, of the frames more than that many time points after M4
# (same rule as the pipeline's rerun mode), and blank those frames in the
# filemaps and per-frame reports (morphology, QC, ...). Annotations are kept. Run
# with DRY_RUN=1 first to see what would be deleted and blanked.
#
# Edit the parameters below, then run:
#   bash tools/clean_experiment.sh
# Command-line flags override them:
#   bash tools/clean_experiment.sh -e /path/to/experiment [-a analysis] [-m N] [-r raw] [-n | --delete]

# ---- Parameters ----
EXPERIMENT_DIR="/path/to/experiment"
ANALYSIS_DIR_NAME="analysis"
RAW_DIR_NAME="raw"
# Leave empty to only delete frames flagged Ignore.
MAX_TIME_AFTER_M4=""
# 1 = only list the files that would be deleted, 0 = delete them.
DRY_RUN=1
# --------------------

usage() {
    echo "Usage: $0 [-e <experiment_dir>] [-a <analysis_dir_name>] [-m <max_time_after_m4>] [-r <raw_dir_name>] [-n|--dry_run|--delete]" >&2
    exit 1
}

while (( $# > 0 )); do
    case "$1" in
        -e|--experiment_dir)
            [ -n "$2" ] || usage
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        -a|--analysis_dir_name)
            [ -n "$2" ] || usage
            ANALYSIS_DIR_NAME="$2"
            shift 2
            ;;
        -m|--max_time_after_m4)
            [ -n "$2" ] || usage
            MAX_TIME_AFTER_M4="$2"
            shift 2
            ;;
        -r|--raw_dir_name)
            [ -n "$2" ] || usage
            RAW_DIR_NAME="$2"
            shift 2
            ;;
        -n|--dry_run)
            DRY_RUN=1
            shift
            ;;
        --delete)
            DRY_RUN=0
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            usage
            ;;
    esac
done

ARGS=(-a "$ANALYSIS_DIR_NAME" -r "$RAW_DIR_NAME")
[ -n "$MAX_TIME_AFTER_M4" ] && ARGS+=(-m "$MAX_TIME_AFTER_M4")
[ "$DRY_RUN" = "1" ] && ARGS+=(--dry_run)

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
~/.local/bin/micromamba run -n align python3 "$SCRIPT_DIR/clean_experiment.py" \
    "$EXPERIMENT_DIR" "${ARGS[@]}"
