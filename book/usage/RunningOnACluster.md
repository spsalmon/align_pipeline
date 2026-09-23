# Running on a cluster

With `backend: "slurm"` (the default), the pipeline does not run your analysis
itself: it submits each building block as a separate SLURM job, and each job
submits the next one when it is done. A pipeline of 12 blocks is therefore 13
jobs, not one.

You do not need to know any of this to use the pipeline — but you do need to tell
SLURM how much memory, how many CPUs and how much time each step gets.

## The SLURM configuration file

These resource requests live in their own file, next to your main configuration:

```yaml
# in your main config
slurm_config: "slurm_config.yaml"
```

The path is relative to the main configuration file. `align init-configs` copies both files together for this reason.

The file looks like this:

```yaml
sbatch_memory: 64G
sbatch_time: 0-48:00:00
sbatch_cpus: 32
sbatch_gpus: "rtx6000:1"
```

These are the **defaults applied to every block**. GPUs are only actually
requested for the blocks that can use one (segmentation, molt detection, or a
custom script that asks for it).

Any `sbatch_*` key written directly in the main configuration overrides the file.

## Giving one kind of block different resources

Most steps do not need 32 CPUs and a GPU. Override them per block type under
`sbatch_overrides`, listing only what differs:

```yaml
sbatch_overrides:
  segmentation:
    sbatch_memory: 32G
    sbatch_gpus: "rtx6000:1"
  morphology_computation:
    sbatch_cpus: 8
    sbatch_gpus: null
```

The keys are block names (`segmentation`, `straightening`, ...). Everything not
listed is taken from the defaults above.

## The orchestrator job

One extra job runs the pipeline itself: it reads the configuration, builds the
filemap and submits the first block. It does not do image analysis, so it usually
needs far less than a block. Its resources go under `sbatch_init`:

```yaml
sbatch_init:
  sbatch_cpus: 4
  sbatch_memory: 8G
  sbatch_time: 0-12:00:00
```

`sbatch_init` is deliberately separate from `sbatch_overrides` — the orchestrator
is not a building block.

## Cluster-specific options

Clusters differ: some need an account, a partition, memory per CPU rather than
total memory, or a custom resource. Anything SLURM understands can be passed
through verbatim:

```yaml
sbatch_extra_options:
  - "--account=gratis"
  - "--mem-per-cpu=4G"
```

Each entry becomes one `#SBATCH <entry>` line in the generated job script. This
means **moving the pipeline to a different cluster is a change to your
configuration file, never a change to the pipeline code**.

Two rules are worth knowing:

- The standard keys (`sbatch_memory`, `sbatch_cpus`, `sbatch_time`,
  `sbatch_gpus`) are only sent to SLURM when you set them. Drop `sbatch_memory`
  and use `--mem-per-cpu` in `sbatch_extra_options` instead, if that is what your
  cluster wants.
- `sbatch_extra_options` entries written at the top level are **added to**, not
  replaced by, the ones in `sbatch_overrides` and `sbatch_init`. Cluster-wide
  settings like `--account` therefore belong at the top level, where they always
  apply. If an option must *not* apply everywhere, move it down into the sections
  that need it.

## Using a different environment manager

By default the pipeline launches its jobs with micromamba. If your cluster uses
conda, or a plain virtual environment, set `python_command` in your main
configuration:

```yaml
python_command: "conda run -n align python"
```

This one key controls how Python is started everywhere, both for the orchestrator
and for every block.
