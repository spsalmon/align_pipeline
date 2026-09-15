# Monitoring a run

## What the pipeline prints

At startup, the run prints the full list of steps it is going to perform, then a
marker around each one:

```
### Starting block 3/12 straightening ###
...
### Finished block 3/12 straightening ###
```

The number tells you which step you are on, and blocks of the same type are told
apart by their position. The output is the same whether you are running on a
cluster or on your own machine.

On the cluster, each block also prints `Submitting <block> to slurm ...` just
before SLURM answers with its job id, so the id always has a name attached to it.

## Where the logs are

Each run gets its own directory, named after the SLURM job id (or after the start
time when running locally):

```
temp_files/pipeline_1234567/
├── pipeline-1234567.out     <- everything, in order  ← start here
├── pipeline-1234567.err
├── sbatch_output/           <- one log per job
│   ├── init_pipeline-1234567.out
│   ├── segmentation-1234568.out
│   └── straightening-1234569.out
├── batch/                   <- the generated job scripts
└── pickles/                 <- internal state
```

**`pipeline-<id>.out` is the file to read.** It concatenates the logs of every job
in the order they ran, with a `===== <file> =====` header before each section, so
the whole run can be followed in one place. It is rebuilt after every step, so a
run that stops halfway still leaves a readable log.

`temp_files/` is inside the pipeline folder unless you set `temp_dir` or pass
`-t`.

## Did the run finish?

The last line of the combined log answers this:

- `PIPELINE FINISHED -- all N blocks completed` — the run reached the end.
- `pipeline still running -- k/N blocks done so far` — the run had not finished
  when that line was written. If nothing is in the queue any more, this is where
  it stopped.

## When something goes wrong

Because each block is a separate cluster job, a block that fails simply stops the
chain: nothing submits the next step, and the run ends quietly. To find out what
happened:

1. Open `pipeline-<id>.out` and look at the last `### Starting block ... ###`
   line — that is the step that did not finish.
2. Open the matching `.err` file in `sbatch_output/` for the real error message.
3. The generated job script for that block is in `batch/`, if you want to see the
   exact command that ran.

A run can simply be restarted with the same command. With the `rerun_*` options
set to `False`, the images that were already processed are skipped, so the
pipeline picks up roughly where it stopped.

```{note}
A mistake in the configuration is caught **before** anything is submitted, and is
printed straight to your terminal. If you got as far as having log files, the
configuration was fine.
```

## Cleaning up

The scratch directory is not needed once a run has finished — a full copy of it
is kept in the experiment's `analysis/report/pipeline_backup/` folder. Two ways to clear
it:

- Set `cleanup_on_success: True` in your configuration, and each successful run
  deletes its own scratch directory automatically.
- Run `bash scripts/cleanup_temp_files.sh` to clear the default `temp_files/`
  folder of all runs at once. A custom `temp_dir` has to be cleared by hand.

Neither ever touches your results.
