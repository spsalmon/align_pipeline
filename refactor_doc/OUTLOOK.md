# Outlook — future work & deferred engineering

Forward-looking notes: what is still to come, and the engineering/design cleanups
consciously deferred. The high-level "what's next" lives in
[`REFACTOR_OVERVIEW.md`](REFACTOR_OVERVIEW.md) — its three major follow-ups are the
top three below. Decisions that cost something are in
[`TRADEOFFS.md`](TRADEOFFS.md); the user-docs feedstock is in
[`DOCS_TODO.md`](DOCS_TODO.md). Section names in quotes below refer to DOCS_TODO
sections.

## Future work (roughly priority-ordered)
Single index of what is still to come; details live in DOCS_TODO where noted.
Higher items first; the top three are the overview's major follow-ups, the rest
are smaller. (DONE: config-validation of input paths + unknown-key rejection —
see "Config validation"; the `init-configs` scaffolding command and the
subcommand dispatcher — see "CLI / commands".)
1. **Extras adaptation (PR F)** — bring `tools/`, `gui/`, `training/`,
   `examples/custom_scripts/` onto the new layout + code conventions (see the
   "Code conventions" scope note). Deferred until the core path is agreed.
2. **Docs rewrite (PR G, last)** — README + `book/` overhaul, driven from
   DOCS_TODO. Only after the core overhaul is agreed with the product owner.
3. **"Real" API** — an object-oriented / stepwise outer orchestration in Python
   (drive blocks one at a time, opt-in linking) alongside the current config-driven
   run. Larger design effort; optional.
4. **Publish to PyPI** — currently installed from the repo/checkout only; publishing
   would make `pip install align_toolbox-pipeline` work directly. Easy later step.
5. **Config validation, further** — optional warning-level "contents reasonable"
   checks (e.g. `experiment_dir` contains a `raw/`), and running `validate_config`
   in the login-node pre-flight (see the follow-up under "Config validation").
6. **(lower) Warning-log volume** — some warnings can fire once per image and blow
   up the `.out` file(s). Idea: keep a list of ignorable warnings in a repo file
   (adjustable, but out of the user's config surface) and filter those when logging.
7. **(lower) Output-filename suffix** — optional (bool, default off) suffix on output
   names. Deferred/parked: the data handling / read-in is changing soon, so not worth
   doing against the current scheme.

## Deferred engineering cleanup
- Cluster dep de-duplication was CONSIDERED and dropped: the cluster build is a
  hash-pinned `--require-hashes --no-deps` install from `conda-linux-64.lock`, so
  a `- .` (unhashable local path) does not fit, and `environment.yml` is a
  superset of pyproject (gui/training/cellpose/bioio, conda-delivered). Instead
  the package is installed separately, editable + `--no-deps`, by
  `install_pipeline.sh`/`update_pipeline.sh` after the env build — lock untouched.
  A future pyproject-extras approach (gui/training groups) could revisit de-dup.

## Deferred design / cleanup (later PRs)
- Folder inputs, further: `resolve_ref` covers name-only refs (done). Still open
  if needed: first-class (a) absolute and (c) relative external-directory refs
  (today an absolute path passes through, but there is no relative-to-experiment
  form) — add only if cross-experiment refs are actually wanted.
- Naming: `analysis_dir_name` / `analysis_subdir` really denote the OUTPUT
  directory. Renaming the KEY is a breaking config change, so it stays deferred
  and separate from the (now-done) ref decoupling.
