# Docs rewrite — things to cover

Running notes of things to include when the docs (README + book) are rewritten
at the end of the refactor. Add to this as we change things; don't edit the
docs piecemeal in the meantime.

Decisions that cost something go in `TRADEOFFS.md` instead — that file is the
list to hand to a reviewer, this one feeds the docs. Forward-looking engineering
TODOs and the future-work index live in `OUTLOOK.md`.

## Repo structure
- `align_pipeline/` = core pipeline package (`python -m
  align_pipeline...`), with `workers/` (one worker per block) and
  `defaults/` = bundled `config/` + `models/` (fallbacks/examples, shipped as
  package data and resolved package-relative). `scripts/` = automation bash.
  `tools/` = one-off data-conversion helpers (auxiliary). `examples/custom_scripts/`
  = user extension-point templates. `training/` and `gui/` are separate auxiliary
  trees. Add a repo map to the README.
- Run/install now via scripts/: `bash scripts/run_pipeline.sh`,
  `bash scripts/install_pipeline.sh`.

## Repo map & deployment lifecycle (turn into a README diagram at the docs step)
Four tiers:
- PACKAGE: `align_pipeline/` (core + `workers/` + `defaults/`) -- the
  installable, self-contained pipeline.
- DEPLOYMENT: `env/` and `scripts/`, which pair up. `env/` = define & build the
  conda environment (spec + generated locks + `build_env.sh`/`generate_lock.sh`);
  `scripts/` = operate the pipeline. Flow:
    `env/environment.yml` --generate_lock.sh--> `env/conda-linux-64.lock`
      --build_env.sh--> the `align_pipeline` env
    `scripts/install_pipeline.sh` orchestrates that build (+ `pip install -e .
      --no-deps` once PR C lands, to register the package + entry point)
    `scripts/run_pipeline.sh` -> `scripts/_init_pipeline.sh` -> the package
      (submits the self-propagating slurm chain)
- EXTRAS: `tools/`, `training/`, `gui/` (each owns its own launch scripts),
  `analysis_and_plots/`, `examples/`.
- META: `README`, `pyproject.toml`, `book/` (docs), `tests/`, TRADEOFFS/DOCS_TODO.
- `scripts/` by lifecycle: SETUP (`install_pipeline` cluster,
  `install_pipeline_local` local, `update_pipeline`) | RUN (`run_pipeline` ->
  `init_pipeline`) | MAINTAIN (`cleanup_temp_files`).

## Doc path fixes for the rewrite (files moved/renamed in the layout PR)
- `book/usage/UsingGUI.md`: `bash launch_gui.sh` -> `bash gui/launch_gui.sh`.
- `book/getting_started/Update.md`: `./requirements/conda-lock.yml` ->
  `./env/conda-lock.yml`.

## Installation / environment
- Local install without micromamba, any OS: run `bash scripts/install_pipeline_local.sh`
  (creates the env + editable install via `conda run`), then `conda activate
  align_pipeline_local`. Manual equivalent: `conda env create -f
  env/environment_local.yml`, `conda activate align_pipeline_local`,
  `pip install -e ".[dev]"`. Three install paths: cluster
  (`install_pipeline.sh`), local (`install_pipeline_local.sh`), manual package.
- `lxml` is needed to read OME-TIFF metadata cleanly (otherwise a warning).
- Cluster path (micromamba + conda-lock + `scripts/install_pipeline.sh`): builds
  the locked env, then registers the pipeline package into it with
  `pip install -e . --no-deps` (so `align_pipeline` imports from any cwd and
  the `align_pipeline` command exists). `update_pipeline.sh` reinstalls it
  only when it rebuilds the env; a pipeline-only update relies on the editable
  install already tracking the checkout.
- `pip install -e ".[dev]"` (via pyproject.toml) installs the pipeline as a
  package so it runs from anywhere (no repo-root / PYTHONPATH). Local install is
  two steps from the repo root: `conda env create -f
  env/environment_local.yml`, then `pip install -e ".[dev]"`. (conda
  resolves a pip `-e .` relative to the yml's dir, so we don't put it in the yml.)
- Explain the dependency layering:
  1. pyproject.toml = what the pipeline needs (abstract deps, single source).
  2. environment*.yml = how to build an env (conda-vs-pip delivery choice).
  3. conda-lock.yml / conda-linux-64.lock = exact pinned solve (cluster).

## Running the pipeline
- New `backend` config option: `slurm` (default, submits jobs) vs `local` (runs
  in-process, no slurm/micromamba).
- Local run, from the repo root:
  `python -m align_pipeline.init_pipeline -c <config> --temp_dir <dir>`.
- Installed console command (equivalent, works from any cwd once pip-installed):
  `align_pipeline -c <config> --temp_dir <dir>` (entry point ->
  `init_pipeline:main`). The `-m` form stays the fallback and is still what the
  cluster launcher uses until the package is installed there (env consolidation).
- `experiment_dir` can be given with `--experiment_dir` (overrides the config).
- Temp/output location precedence: `--temp_dir` flag > `temp_dir` config key >
  default `./temp_files` (in-repo, gitignored). Durable outputs and the backup
  live under `experiment_dir`; the outer launcher makes a repo-root
  `sbatch_output/` landing zone that the pipeline relocates into the run dir (and
  `cleanup_on_success` can remove).

## Which script does what
- `scripts/run_pipeline.sh` — the user entry point, runs on the login node:
  creates the repo-root `sbatch_output/` landing zone, finds the config among the
  arguments, resolves the python launcher, derives the outer job's sbatch flags
  from `sbatch_init`, submits. No longer auto-updates from git; to update, run
  `scripts/update_pipeline.sh` (`--pipeline-only` to skip the env rebuild).
- `scripts/_init_pipeline.sh` — the submitted job, named after the module it
  launches; the leading `_` marks it internal (run_pipeline.sh submits it, the
  user never runs it). A `#SBATCH` header (which must live in a file for sbatch),
  the env activation, and `"$@"` passed straight through. Nothing else belongs here.
- Everything else is Python. `setup_run_dir()` resolves the per-run directory
  (`pipeline_<jobid>` under slurm, `pipeline_<timestamp>` otherwise) and, under
  slurm, moves the launcher's logs into it — so the temp path has a single
  definition instead of one in bash and one in argparse. Tradeoff: if the env is
  broken and Python never starts, those logs stay in the repo-root
  `sbatch_output/` instead of moving into the run dir.
- Only `batch/` and `sbatch_output/` are slurm-specific; every other write
  (outputs, report, backup, provenance, pickles) happens on both backends.
- `align_pipeline/workers/` holds one worker per block implementation,
  named after its block (`straightening`, `morphology_computation`,
  `segmentation_non_learning`/`segmentation_learning_based`, ...). Each
  `BuildingBlock` stores its `worker_module` and `create_command` runs it with
  `python -m align_pipeline.workers.<name>`, so workers resolve by import
  rather than by a working-directory-relative path. Custom blocks run a
  user-supplied `custom_script_path` (file, `.py` or `.sh`) instead. Generated
  job scripts and logs stay named after the BLOCK (what `sbatch_overrides` keys
  on and the progress log prints); a block can map to several workers.

## Log output
- The run prints a numbered plan of the blocks up front, then a
  `### Starting block 3/12 straightening ###` / `### Finished block ... ###` pair
  around each one, and a closing line. Same on both backends: on slurm the
  linker is appended to the worker's job script, so the finish line and the next
  block's submission land in that block's sbatch `.out` file. A
  `Submitting <block> to slurm ...` line precedes sbatch's own
  "Submitted batch job <id>" so the job id has context.
- Prints that precede a subprocess are flushed, otherwise Python's block
  buffering (stdout redirected to a file under slurm) reorders them after the
  worker's output.
- Slurm log layout: every job writes into `<temp>/pipeline_<id>/sbatch_output/`,
  the outer one as `init_pipeline-<id>.out/.err` (so it sorts first), each block as
  `<block>-<id>.out/.err`. The linker joins them per stream into
  `<temp>/pipeline_<id>/pipeline-<id>.out` (and `.err`) — one file to read the
  whole chain — with a `===== <file> =====` header per section. Originals are
  kept, and the combined files sync into the run backup.
- The combined files are rebuilt at every link, not just at the end, so a run
  that dies mid-chain still leaves a readable log. Their last line says which:
  `PIPELINE FINISHED -- all N blocks completed` only ever gets written by the
  link that found no next block, so anything else (`pipeline still running --
  k/N blocks done so far`) means the run stopped without reaching the end.
- Deliberately NOT a separate end-of-run slurm job: it would sit in the queue,
  so the completion marker could arrive long after the run actually ended, and
  a failed submission would lose the marker entirely. The final link writes it
  instead. The prints it makes are flushed before the concatenation, so they do
  land in the combined log.

## Code conventions (for the contributing/docs section)
- Section dividers in a module are `# ---- Title ----` (capitalised, spaces around
  the dashes), two blank lines before, two after.
- Imports are grouped stdlib / third-party / first-party (`align_pipeline`),
  one blank line between groups. (`workers/straightening.py` keeps a `# noqa: E402`
  block because it sets an OpenBLAS env var before importing.)
- Multiple names from the same module go in one `from x import a, b, c` statement,
  wrapped in parentheses (one name per line, trailing comma) when it exceeds the
  line length -- the isort/black default, not one-import-per-line.
- Two blank lines between top-level functions/classes, one between methods
  (already holds package-wide; verified, no exceptions).
- Module docstrings on the orchestration entry points (init_pipeline, block_linker,
  building_blocks, run_params); the workers are short enough to read directly.
- Scope: these conventions were applied to the core package `align_pipeline/`
  (incl. `workers/`) and the `tests/` suite. NOT yet applied to the extras --
  `tools/`, `gui/`, `training/`, `examples/custom_scripts/` -- which are deferred
  to the non-core PR (F); bring them in line then. (`scripts/` is bash, N/A.)

## Testing
- `python -m pytest tests/ -v` runs the local-backend smoke test (synthetic
  images, no bundled data). Document how to add more tests.
- Unit tests cover the pure helpers (config validation + parsing, output naming,
  folder refs, input selection, slurm resolution, logging) alongside the local
  e2e smoke test, and are grouped into labelled sections in the test file.
- Known untested area: `get_experiment_time_from_filemap` (the T0 / incremental
  ExperimentTime logic). The e2e test runs with `get_experiment_time=False`, so
  this path has no coverage; a faithful test needs a filemap with acquisition
  dates and exercises the recursive recompute branch. Worth adding later.
- CI (`.github/workflows/tests.yml`): on every push and pull request, a GitHub
  Actions job does `pip install -e ".[dev]"` + `pytest` on Ubuntu / Python 3.12
  — the same pip path as the local install (`environment_local.yml`), so a red
  build means a fresh local install would fail too. Deliberately NOT the cluster
  micromamba/conda-lock build (that is a hash-pinned Linux artifact for
  production, overkill for testing logic). pyproject deps are unpinned, so an
  upstream release can turn CI red on its own; a version matrix and dependency
  pinning are deferred (widen the workflow's single 3.12 to a matrix later if the
  product owner wants multi-version support).

## Config
- Document `backend` (`slurm` vs `local`).
- `n_jobs` (main config, all backends) = worker compute parallelism (joblib
  `--n_jobs`). Falls back to `sbatch_cpus` and vice versa, so setting one is
  enough.
- SLURM resources live in a separate file `configs/slurm_config.yaml`, pointed
  to by `slurm_config:` in the main config (relative to it) and merged in at
  startup for the slurm backend only. Inline `sbatch_*` keys override the file.
- Each standard sbatch directive (`-c`, `-t`, `--mem`, gpu gres) is emitted only
  when set, so one can be dropped (e.g. omit `sbatch_memory`, use
  `--mem-per-cpu` instead). `sbatch_extra_options` is a list of raw sbatch
  option strings rendered verbatim as `#SBATCH <option>` lines — cluster-specific
  directives (`--account`, `--mem-per-cpu`, `--partition`, custom gres) are now
  config-only, no edits to `align_pipeline/utils.py`.
- Per-block SLURM resources: the top-level `sbatch_*` keys are the shared
  default for every worker block. Override per block type under
  `sbatch_overrides` (keyed by block name, e.g. `segmentation`), merged over the
  default. The outer/orchestrator job's resources go under `sbatch_init`.
  Per-instance (a specific occurrence) overrides are still future work.
- Merge rule inside the slurm config: scalar `sbatch_*` keys in a section
  (`sbatch_overrides.<block>`, `sbatch_init`) REPLACE the top-level default,
  while `sbatch_extra_options` entries are APPENDED to it. So cluster-wide
  invariants (`--account`, `--mem-per-cpu`) belong at the top level and always
  apply; a section can add to them but not drop them (move an option down into
  the sections if it must not apply everywhere).
- `sbatch_init` is deliberately NOT a key inside `sbatch_overrides`: the outer
  job is not a building block, it's resolved by a different program at a
  different time (`run_params.py`, from bash, before the pipeline starts), and
  it would collide with the block-name namespace.
- The outer/orchestrator job's resources come from `sbatch_init`. `run_pipeline.sh`
  turns them into sbatch CLI flags (via `python -m align_pipeline.run_params
  --sbatch-init`) which override `_init_pipeline.sh`'s minimal header. So a new
  cluster is adjusted entirely in the config now — cluster-specific outer
  directives (`--account`, a custom `--gres` like the old `pipelinecapacity`
  throttle, `--mem-per-cpu`) go under `sbatch_init.sbatch_extra_options`.
- `run_pipeline.sh` forwards `-e/--experiment_dir` and `-t/--temp_dir` through to
  the pipeline (previously only `-c`).
- Env launcher is decoupled, one knob: `python_command` in the main config sets
  the command that launches python everywhere. `run_pipeline.sh` greps it from
  the config (pure bash — python is what it is resolving) and exports it as
  `TOWBINTOOLS_PYTHON` for the pre-flight and the submitted job; the workers read
  it via `get_python_command`. Exporting `TOWBINTOOLS_PYTHON` directly still wins
  (bootstrap escape hatch when the default can't even start python). Unset
  everywhere = the micromamba default, so behaviour is unchanged.
- Temp working dir defaults to in-repo `./temp_files` (gitignored, transient).
  Decision: keep this default rather than auto-placing it next to the experiment
  — the durable outputs and backup are already external, and `-t <path>` covers
  the large-experiment / home-quota case (e.g. put temp on the data storage). Not
  worth the bash complexity of resolving the experiment dir before Python runs.
- Two ways to clear scratch:
  - `cleanup_on_success` config key (default false): on SUCCESSFUL completion the
    final linker deletes THIS run's whole temp dir (`temp_files/pipeline_<id>/` —
    pickles, batch, per-block + combined logs) AND the empty repo-root
    `sbatch_output/` landing zone. Safe: `sync_backup_folder` mirrors the full
    temp dir into `pipeline_backup/` first, and results live under `analysis/`, so
    only redundant scratch is removed. Never fires mid-run or on failure (nothing
    resumable is touched). Reclaims scratch, keeps the durable backup.
  - `scripts/cleanup_temp_files.sh` — manual sledgehammer: clears the DEFAULT
    `temp_files/` (all runs) + repo-root `sbatch_output/`. A custom `temp_dir`
    must be cleared by hand.

## Run backup / provenance
- The pipeline snapshots the config(s) used and a `git_info.txt` (git
  branch/commit/status + interpreter/package versions) into the run's temp dir,
  which syncs into the backup. Done in Python, so local runs are recorded too
  (previously only the sbatch launcher did this, slurm-only).
- The backup lives at `<experiment>/analysis/report/pipeline_backup/pipeline_<id>/`
  (in subdirectory mode: `report/<subdir>/pipeline_backup/pipeline_<id>/`).
- Temp and backup are write-only records: the pipeline reads its config from the
  original path given with `-c`, never from the temp copy. This is why a
  relative `slurm_config:` resolves against the original config's directory. (The
  old launcher copied the config into temp and ran that copy, so the sibling
  `slurm_config.yaml` was silently not found.)

## Config
- `analysis_dir_name` (default `analysis`) is honored everywhere, including the
  prefix-stripping in output naming. The `report/` subfolder name is currently
  fixed (no config key).
- `backend`: `slurm` (default) vs `local`. Config-only (no CLI flag); a commented
  hint is in the default config.
- Folder references (masks/sources) are resolved by `resolve_ref`: a directory
  ref normalizes to `{analysis_dir_name}/{name}` whether written with or without
  the prefix, so `ch2_seg` == `analysis/ch2_seg` and refs survive renaming
  `analysis_dir_name`. `raw` and absolute paths pass through. Applies to
  directory refs only, NOT report-column refs like `molt_detection_columns`. The
  shipped config now uses the prefix-free form.

## Config validation
- The config is checked once at startup (`validate_config`, before any run dir
  or job is created) and every problem is reported together, not one at a time.
  Checks: required keys (`experiment_dir`, `building_blocks`), known block names
  (with a hint that `classification` became `quality_control`), per-block option
  lists holding one value per block of that type (or a single broadcast value),
  the enumerated `backend` / `report_format` values, unknown/typo'd top-level
  keys, and existence of the input paths given in the config.
- Unknown-key check: any top-level key that is not a known global
  (`GLOBAL_CONFIG_KEYS`), a per-block option (`OPTIONS_MAP`), or an `sbatch_*`
  key is rejected — so `pixlesize`, `experimnt_dir`, etc. fail up front instead
  of being silently ignored. `sbatch_*` keys pass by prefix (cluster-specific,
  free-form). Runs on the raw config before any internal keys are injected.
- Input-path existence: `experiment_dir` must be a directory, and the file
  options (`model_path`, `qc_model_path`, `molt_detection_model_path`,
  `custom_script_path`) must point at existing files. Only inputs the user
  supplies — folders an earlier block produces are deliberately NOT checked (they
  do not exist yet at validation time).
- The per-block length rule mirrors the one `parse_building_blocks_config` still
  enforces later; validation front-loads it so the whole config fails fast and
  at once. See the TRADEOFFS entry on the small duplication.
- A bad config ABORTS the run: `validate_config` raises, nothing catches it, the
  process exits non-zero before any run dir/backup/job exists (on slurm the outer
  job fails and the block chain never starts).
- `building_blocks` rule is permissive by design — "every entry is a KNOWN type",
  NOT "one of each". No required block, no ordering/dependency check, duplicates
  are fine (several `segmentation` blocks is normal).
- Still NOT validated (confirm with the product owner before hardening further):
  - inter-block dependencies / order (e.g. a `morphology_computation` whose mask
    is produced by an earlier `segmentation`) — a wrong order just yields "no
    input files" at run time, not an upfront error.
  - folder-ref existence for INTERMEDIATE refs — a mask/source ref pointing at a
    folder no earlier block produces is not caught (it cannot be: the folder is
    created during the run). Only user-supplied input paths are existence-checked.
- Follow-up (nice-to-have): also run `validate_config` in the login-node pre-flight
  (`run_pipeline.sh` already loads the config there for `sbatch_init`), so a bad
  config fails BEFORE the sbatch submission with the error printed straight to the
  terminal. Today validation runs inside the submitted job and aborts before the
  run dir / log relocation, so the error lands in the outer job's slurm err file in
  the repo-root landing zone (`sbatch_output/pipeline-<jobid>.err`) -- correct, but
  easy to miss, and it still cost a job submission. Confirmed on the cluster:
  the error is there and no folders are created.

## CLI / commands
- The installed command is a subcommand dispatcher (`align_pipeline/cli.py`):
  - `align_pipeline run [config] [-c CONFIG] [-e ...] [-t ...]` — run the
    pipeline. The config may be positional or `-c` (`-c` wins if both are given).
  - `align_pipeline init-configs [DIR] [--force]` — copy the bundled default
    `config.yaml` + `slurm_config.yaml` into DIR (default cwd), so a user can start
    from them without digging into the installed package. Configs only (the bundled
    models are large; it prints where they live); skips existing files unless
    `--force`. Both files are copied because the main config references the slurm
    one by a relative path.
- Aliases: `tt-pipeline` and `ttp` are shorter console names for the same command.
- Back-compat: with no recognised subcommand the arguments go straight to `run`,
  so the old `align_pipeline -c config.yaml` still works, as does
  `python -m align_pipeline.init_pipeline ...` (the cluster launcher path,
  untouched). New extras (e.g. a `start-gui`) slot in as further subcommands, each
  lazy-importing its module so `run` never pays for their dependencies.

## CLI flags — the rule
- Precedence is uniform: CLI flag > config key > default. The config is the only
  required input (positional or `-c/--config`) and has no config-key counterpart
  (by definition).
- Flags cover what varies per invocation or per machine (`-e/--experiment_dir`
  where the data is, `-t/--temp_dir` where scratch goes); everything scientific
  stays in the config. Deliberately no flag for `analysis_dir_name` — it is an
  output-layout choice, and today changing it also means rewriting every
  `analysis/...` reference in the config, so a flag would be a half-measure.

## Known cleanups to mention / finish before docs
- Document custom blocks. The `CustomBuildingBlock.create_command` bug (missing
  `config` param, plus a doubled `run -n align_pipeline python3` launcher) is fixed;
  custom blocks now work on both backends.
- The outer orchestrator job's resources are now config-driven: `run_pipeline.sh`
  passes sbatch CLI flags built from `sbatch_init`, overriding the minimal header
  in `_init_pipeline.sh`. The env launcher is now overridable too (see
  `TOWBINTOOLS_PYTHON` under "Which script does what").
