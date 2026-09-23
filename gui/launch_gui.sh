#!/usr/bin/env bash

# Run from the repo root so ./gui/run.py resolves.
cd "$(dirname "$0")/.."

# --- Configuration: edit these ---
FILEMAP_PATH="/mnt/towbin.data/shared/kstojanovski/20240202_Orca_10x_yap-1del_col-10-tir_wBT160-186-310-337-380-393_25C_20240202_171239_051/analysis_sacha/report/analysis_filemap.parquet"
OPEN_ANNOTATED=1      # 1 = open annotated if exists, 0 = always open original
RECOMPUTE_VALUES_AT_MOLT=0            # 1 = recompute features at molt, 0 = skip if already computed
PORT=0             # 0 = random available port
HOST="127.0.0.1"
# ---------------------------------

${ALIGN_PYTHON:-$HOME/.local/bin/micromamba run -n align python3} "./gui/run.py" \
    ${FILEMAP_PATH:+--filemap "$FILEMAP_PATH"} \
    ${RECOMPUTE_VALUES_AT_MOLT:+$([ "$RECOMPUTE_VALUES_AT_MOLT" = "1" ] && echo "--recompute")} \
    $([ "$OPEN_ANNOTATED" = "0" ] && echo "--no-annotated") \
    --host "$HOST" \
    --port "$PORT"
