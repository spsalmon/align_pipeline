# Trade-offs

Places where a choice closed off an alternative. Kept so the reasoning survives
the commits and can be reviewed as a whole.

Split into **A. Introduced by this refactor** — decisions we took, and the ones
to scrutinise — and **B. Inherited and consciously kept** — pre-existing
behaviour we examined and decided not to change, with the reason.

Format: **decision** — why — *cost* — how hard to reverse.
Add an entry whenever a change has a real cost; not for choices that are simply
correct.

---

# A. Introduced by this refactor

## Execution model

**Local backend shells out instead of running in-process.**
The local backend reuses the existing `subprocess` call and only swaps the
interpreter for `sys.executable`, rather than importing the workers and calling
them directly. Chosen for a small diff and one code path shared with slurm.
*Cost: a process launch per block, no shared memory between blocks, and local
debugging still crosses a process boundary.*
Reversible: moderate — `run_command_local` is the only place to change.

**The completion marker is written by the last linker, not by a dedicated job.**
A separate end-of-run slurm job would sit in the queue, so the "finished" signal
could arrive long after the run really ended, and a failed submission would lose
it entirely.
*Cost: if the final job is killed (e.g. time limit) after its block finished but
before the linker ran, a run that did all its work shows no FINISHED marker.*
Reversible: easy.

**The combined log is rebuilt at every link rather than written once at the end.**
A run that dies mid-chain still leaves a readable log.
*Cost: the file is rewritten once per block, and the tail of the block running
while it is written is never included.*
Reversible: easy.

## Layout and packaging

**Flat package layout, not `src/`.**
`src/` would require an install or `PYTHONPATH` for the package to be importable,
coupling every cluster invocation to a correct install. Flat keeps
`python -m towbintools_pipeline...` working from the repo root anywhere.
*Cost: the repo root is importable, so a working tree can silently shadow an
installed copy — the class of bug `src/` exists to prevent.*
Reversible: moderate; touches every invocation path.

**Workers are invoked as modules, not by absolute file path.**
`create_command` runs `python -m towbintools_pipeline.workers.<name>`, so workers
resolve by import and use package-clean imports (`from towbintools_pipeline
import utils`) instead of relying on their own directory being on `sys.path`.
*Cost: the worker's environment must have the package importable — installed, or
launched from the repo root. The outer job already required this, so it is not
new, but a run started from elsewhere without an install would now fail to find
the workers.*
Reversible: easy — `create_command` is the single place.

**Bundled `defaults/` (configs + models) live inside the package, resolved by `__file__`.**
`defaults/` moved from the repo root into `towbintools_pipeline/`, is declared as
package data, and is looked up via `_PIPELINE_DIR` (the package dir from
`__file__`) rather than `importlib.resources`. The flat layout means the package
is always real files on disk (editable, checkout, or a normal wheel), so a plain
path works and keeps model paths as filesystem strings for loaders/subprocess —
`importlib.resources` would only matter for a zipped install, which this project
does not use.
*Cost: the 47 MB default molt-detection checkpoint now ships inside the package,
so a non-editable `pip install` copies it into site-packages. Keeping only the
configs bundled and the models external would avoid this.*
Reversible: easy — `git mv` back and repoint two constants.

**Dependencies are declared in two places, and stay that way.**
`pyproject.toml` is the single source for the local install; the cluster
`environment.yml` is a battle-tested superset (it also carries gui/training/worker
deps, conda-delivered). Sourcing it from pyproject with `- .` was rejected: the
cluster build is a `--require-hashes` install and a local path has no hash.
*Cost: a core dependency bumped in pyproject is not reflected on the cluster until
`environment.yml` is bumped too; the two can drift.*
Reversible: easy — a pyproject-extras approach could revisit de-dup later.

**The pipeline package is installed editable + `--no-deps` on the cluster.**
After the locked env is built, `install_pipeline.sh` runs `pip install -e .
--no-deps` to register the package and its entry point, rather than adding it to
`environment.yml`. Editable tracks the checkout the cluster already runs from;
`--no-deps` keeps pip away from the hash-pinned dependency set.
*Cost: a second install step outside the locked build; a fresh env rebuild must
re-run it (handled in install/update), and a dep pyproject needs but the lock
lacks would install nothing and only surface as an import error at runtime.*
Reversible: easy.

**The installed command dispatches subcommands; an unknown first arg means `run`.**
`towbintools-pipeline` routes `run` / `init-configs` and, for anything else, passes
the arguments straight to the runner — so `towbintools-pipeline -c config.yaml`
(and a bare positional config) keep working after subcommands were added, and
extras can attach as further subcommands.
*Cost: the first argument is matched against subcommand names before being treated
as run args, so a config file literally named `run` or `init-configs` given
positionally would be read as the subcommand. Use `-c` for such a path.*
Reversible: easy.

## Configuration

**Folder refs are resolved by basename to `{analysis_dir_name}/{name}`.**
Lets a mask/source ref be written with or without the analysis-dir prefix and
survive renaming `analysis_dir_name`, instead of repeating the prefix in every
ref. Backward-compatible: the old prefixed form still resolves.
*Cost: `resolve_ref` keys on the basename, so a multi-level ref under the analysis
dir would collapse to its last segment (not used today); first-class absolute /
relative-to-experiment external refs are not modelled.*
Reversible: easy.

**`sbatch_extra_options` accumulate; scalar `sbatch_*` keys replace.**
Prevents a per-block or init section from silently dropping a cluster-wide entry
such as `--account`.
*Cost: a section cannot remove a top-level option. An option that must not apply
everywhere has to be moved down into the sections instead.*
Reversible: easy.

**The python launcher is one config key, bridged to the outer job by a bash grep.**
`python_command` in the main config drives both the workers (via
`get_python_command`) and the outer slurm job: `run_pipeline.sh` greps the key
out of the YAML in pure bash and exports it, because it runs before python (the
thing it is resolving) can parse the config. `TOWBINTOOLS_PYTHON` overrides it
for the bootstrap case where the default can't start python at all. The earlier
`micromamba_path` / `slurm_env_name` slot keys were dropped — `python_command`
is a strict superset, so one full string replaces the template.
*Cost: the outer job's launcher comes from a fragile one-line YAML grep, not a
real parse; an exotic quoted/colon'd value could parse differently there than in
python, diverging outer from inner.*
Reversible: easy.

**`sbatch_init` is a separate section, not a key inside `sbatch_overrides`.**
The outer job is not a building block and is resolved by a different program at a
different time (`run_params.py`, from bash, before the pipeline starts).
*Cost: three config sections to learn instead of two.*
Reversible: easy.

**The outer job's resources are sbatch CLI flags overriding the script header.**
This is what makes adapting to a new cluster a config-only change.
*Cost: the flags come from a Python call made before submission. If that call
fails (bad config, broken env) it silently yields nothing and the job falls back
to the bare header. Mitigated by the pre-flight
`python -m towbintools_pipeline.run_params --sbatch-init -c <config>`.*
Reversible: easy.

**Config validation re-checks per-block list lengths that the parser also checks.**
`validate_config` runs at startup to fail fast and report every problem at once;
its per-block length rule restates the one `parse_building_blocks_config` still
enforces (as a backstop) deeper in the build. Kept as two places rather than
routing the parser through the validator, which would be a larger change to a
working code path.
*Cost: the length/broadcast rule lives in two functions and could drift; a change
to one must be mirrored in the other.*
Reversible: easy — the parser's asserts could later defer to `validate_config`.

**Validation rejects unknown top-level keys against a hand-maintained allowlist.**
`validate_config` flags any key that is not a known global (`GLOBAL_CONFIG_KEYS`),
a per-block option (`OPTIONS_MAP`, derived automatically), or `sbatch_*` (prefix).
This catches typos like `pixlesize` up front instead of silently ignoring them.
*Cost: `GLOBAL_CONFIG_KEYS` is a manual list — adding a new global config key
means registering it here too, or a valid config is rejected. `sbatch_*` keys are
waved through by prefix, so a typo'd `sbatch_*` key is not caught. Hard rejection
(not a warning), so a stray key blocks the run.*
Reversible: easy — relax to a warning, or drop the check.

## Files and provenance

**The pipeline reads its config from the original path; temp and backup are write-only.**
This is what makes a relative `slurm_config:` resolve against the config's own
directory. The previous behaviour copied the config into temp, ran the copy, and
silently failed to find the sibling file.
*Cost: the recorded config is a snapshot of intent taken at start-up, not proof
of what executed; blocks run from a pickled copy.*
Reversible: easy.

**`./temp_files` is now the default in all cases.**
The launcher already used it; direct Python runs used `<experiment>/temp_files`.
Unified on the launcher's behaviour, since the durable outputs and backup are
already external and `-t <path>` covers the home-quota case.
*Cost: the default writes inside the repo, which is not self-evident to a new
user. Mitigated: gitignored and transient.*
Reversible: easy.

**`pipeline_backup/` lives inside `report/`.**
Provenance is kept together with the results it produced.
*Cost: backups from earlier runs remain in the old location; there is no
migration.*
Reversible: easy.

**`cleanup_on_success` deletes a finished run's temp dir (opt-in, default off).**
When set, the final linker removes the run's temp dir after mirroring it into the
backup, to reclaim scratch; it never fires mid-run or on failure.
*Cost: after a cleaned run the raw scratch (per-block logs, pickles) survives only
in the backup, not in temp; on slurm the final linker's own live log is discarded
(its content is already in the combined log).*
Reversible: easy.

**The run directory and log relocation moved from bash into Python.**
Gives the temp path a single definition instead of one in bash and one in argparse.
*Cost: if the environment is broken and Python never starts, the outer job's logs
stay in the repo-root `sbatch_output/` instead of moving into the run directory.
Previously bash moved them first, so they landed correctly even then.*
Reversible: easy.

**Local run directories are keyed on a start timestamp with second resolution.**
Slurm uses the job id; local runs previously had no unique id at all and silently
overwrote each other's backup.
*Cost: two local runs started within the same second would collide. Start-up cost
makes this practically unreachable, but it is not impossible.*
Reversible: easy.

**The full building-block config dump was removed from the terminal log.**
It printed the entire resolved config for every block, dwarfing everything else.
*Cost: that detail is no longer in the terminal output — it is still snapshotted
into the run backup.*
Reversible: trivial (one line).

## Process

**Comments are capped at ~2 lines and state what the code does, not why.**
Rationale lives in commit messages, `DOCS_TODO.md` and this file.
*Cost: reading the source alone does not explain why a choice was made.*

**Changes are stacked PRs merged bottom-up.**
Keeps each step reviewable in isolation.
*Cost: the stack is long; a change low in it means rebasing everything above.*

**`gui/`, `training/` and `tools/` were left untouched.**
Scope control — core pipeline first.
*Cost: they may not work against the new layout until adapted.*

**The auto git-update prompt was removed from the launcher.**
`run_pipeline.sh` no longer fetches and offers to hard-reset the working tree to
the remote before each run; updating is now the explicit
`scripts/update_pipeline.sh`.
*Cost: a user on an out-of-date checkout is no longer told so at launch.*
Reversible: easy.

---

# B. Inherited and consciously kept

Pre-existing behaviour we looked at and decided not to change (yet). Listed
because each still carries its cost, and because a reviewer should know these
were examined rather than missed.

**The block chain self-propagates through the linker.**
Each worker's job script has the linker command appended, so a block submits its
own successor. Kept: replacing it would mean rewriting the execution model.
*Cost: no central scheduler sees the whole pipeline. Nothing can report overall
status, cancel the remainder, or notice the chain died — a run simply stops, and
only the logs say where. This is the single largest structural limitation.*
Reversible: hard.

**Generated job scripts, `-J` names and logs are named after the block, not the worker.**
Kept deliberately: a block can map to several workers (`segmentation` →
`segmentation_non_learning` or `segmentation_learning_based`), `sbatch_overrides`
is keyed by block name, and the progress log prints block names. The new logging
was made consistent with this rather than against it.
*Cost: `batch/segmentation.sh` does not match `segmentation_learning_based`, so
the filename alone does not reveal which implementation ran — the command inside
the file does.*
Reversible: easy.

**Blocks of the same type overwrite each other's generated script.**
Two `segmentation` blocks both write `batch/segmentation.sh`.
*Cost: harmless at run time (each is rewritten immediately before its submission),
but `batch/` only ever shows the last version of each type, so it cannot be used
to reconstruct what a specific block instance ran.*
Reversible: easy.

**Generated job scripts carry commented-out lock-file and thread-pinning experiments.**
Left verbatim in `create_sbatch_file` rather than deleted, since they record
something the original author tried.
*Cost: every generated script contains several lines of dead commented code.*
Reversible: trivial.
