#!/bin/bash
set -euo pipefail

# Build the `align_pipeline` environment WITHOUT ever mutating the live env folder.
#
# Why this exists:
#   `micromamba create/install` updates the env folder IN PLACE. When it has to
#   remove a file that is busy (a lingering background process, a Jupyter kernel,
#   an NFS handle), the transaction fails half-way and leaves the env corrupted —
#   most files deleted, no longer a valid conda prefix — forcing a manual rm -rf.
#
# How this avoids it:
#   Every build goes into a brand-new, never-busy, timestamped prefix
#   (envs/align_pipeline_<timestamp>). A stable `envs/align_pipeline` SYMLINK is then
#   flipped to point at it. micromamba resolves `-n align_pipeline` through the
#   symlink, so all the hardcoded `micromamba run -n align_pipeline ...` calls keep
#   working. A failed build leaves the previous env untouched, and busy files in
#   old versions are never in the deletion path of an update.
#
# Usage: build_env.sh            (build fresh + switch over)

MAMBA="$HOME/.local/bin/micromamba"
ENV_NAME=align_pipeline
LOCK_FILE="$(cd "$(dirname "$0")" && pwd)/conda-linux-64.lock"

ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
ENVS="$ROOT/envs"
STABLE="$ENVS/$ENV_NAME"                           # the name everything else uses
NEW="$ENVS/${ENV_NAME}_$(date +%Y%m%d_%H%M%S)"     # fresh build target

mkdir -p "$ENVS"
echo ">> Building a fresh environment at: $NEW"

# --- Phase 1: conda packages into the NEW prefix (empty dir, nothing to remove) -
"$MAMBA" create -p "$NEW" -f "$LOCK_FILE" -y

# --- Phase 2: pip packages, pinned + hash-verified, from the lock's `# pip` lines
# Kept separate from conda on purpose; pip handles its own files and we never feed
# busy in-place removals through a single combined transaction.
TEMP_PIP_REQS=$(mktemp /tmp/pip_requirements_XXXXXX.txt)
grep '^# pip ' "$LOCK_FILE" | sed 's/^# pip //' > "$TEMP_PIP_REQS"
"$MAMBA" run -p "$NEW" pip install --require-hashes --no-deps -r "$TEMP_PIP_REQS"
rm -f "$TEMP_PIP_REQS"

# --- Register the Jupyter kernel from the NEW prefix ----------------------------
"$MAMBA" run -p "$NEW" python -m ipykernel install --user --name="$ENV_NAME"

# --- Atomically switch `align_pipeline` over to the new prefix ---------------------
if [ -L "$STABLE" ]; then
    # Already a symlink: flip it atomically (create temp link, rename over).
    ln -sfn "$NEW" "${STABLE}.tmp"
    mv -Tf "${STABLE}.tmp" "$STABLE"
elif [ -e "$STABLE" ]; then
    # Legacy real env dir (first migration): rename it aside — a directory rename
    # succeeds even if files inside are busy, since it does not touch them — then
    # point the symlink at the new prefix.
    mv -T "$STABLE" "${STABLE}_legacy_$(date +%Y%m%d_%H%M%S)"
    ln -sfn "$NEW" "$STABLE"
else
    ln -sfn "$NEW" "$STABLE"
fi
echo ">> '$ENV_NAME' now points to: $(readlink -f "$STABLE")"

# Make the kernel reference the STABLE symlink path so it survives future swaps.
KSPEC="$HOME/.local/share/jupyter/kernels/$ENV_NAME/kernel.json"
[ -f "$KSPEC" ] && sed -i "s|$NEW/bin/python|$STABLE/bin/python|g" "$KSPEC"

# --- Best-effort cleanup of old / legacy prefixes no longer in use --------------
# Busy files simply cause a skip; leftovers are harmless (never the live env).
CURRENT="$(readlink -f "$STABLE")"
for d in "$ENVS/${ENV_NAME}_"*; do
    [ -d "$d" ] || continue
    [ "$(readlink -f "$d")" = "$CURRENT" ] && continue
    echo ">> Removing old env (best effort): $d"
    rm -rf "$d" 2>/dev/null || echo "   some files busy — leaving $d to clean up next time"
done

echo ">> Done."
