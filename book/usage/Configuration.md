# Configuration

Everything the pipeline does is described in a single YAML configuration file.
YAML is a human readable format: you can read it, and so can the computer. This
page is the reference for the **general** options; the parameters of each analysis
step are documented in the
[building blocks](https://spsalmon.github.io/align_pipeline/building-blocks/buildingblock/)
section.

Get a working configuration to start from with:

```bash
align_pipeline init-configs ~/my_configs
```

## Experiment and outputs

| Key | Default | Meaning |
| --- | --- | --- |
| `experiment_dir` | *required* | Root of your experiment. |
| `raw_dir_name` | `"raw"` | Subdirectory of `experiment_dir` holding the raw images. |
| `analysis_dir_name` | `"analysis"` | Subdirectory where all outputs are written. Change it to run several analyses of the same experiment side by side. |
| `report_format` | `"csv"` | `"csv"` or `"parquet"`. Parquet is much smaller, CSV is easier to open in Excel. |
| `pixelsize` | *required for morphology* | Physical size of a pixel in µm. |
| `get_experiment_time` | `True` | Extract the real acquisition time of each image from the metadata. Slow, but very useful downstream. |
| `time_regex` | `'Time(\d+)'` | Regular expression extracting the time index from the file name. |
| `point_regex` | `'Point(\d+)'` | Same, for the position (individual) index. |
| `overwrite_annotated_filemap` | `False` | If an annotated filemap (produced by the GUI) exists, use it as the starting point instead of the plain one. |

`experiment_dir` can also be given on the command line with
`-e / --experiment_dir`, which overrides the file. This lets one configuration be
reused across experiments.

## What to run

| Key | Meaning |
| --- | --- |
| `building_blocks` | The ordered list of analysis steps. |
| `rerun_<block>` | If `False` (the default), already-processed images are skipped. If `True`, everything is recomputed. |

The block names are `segmentation`, `straightening`, `morphology_computation`,
`quality_control`, `fluorescence_quantification`, `molt_detection` and `custom`.

## Where it runs

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `"slurm"` | `"slurm"` submits each step as a cluster job; `"local"` runs them one after the other on the current machine. |
| `n_jobs` | `sbatch_cpus`, else 1 | How many images are processed in parallel inside a step. |
| `slurm_config` | `"slurm_config.yaml"` | Path to the cluster resource file, **relative to the configuration file**. Only used with the SLURM backend. |
| `python_command` | micromamba (SLURM) / current interpreter (local) | The command used to launch Python everywhere. Set it if you use conda or a plain virtual environment instead of micromamba, e.g. `"conda run -n align_pipeline python"`. |

Cluster resources (memory, CPUs, GPUs, time) live in the SLURM configuration file
— see [running on a cluster](https://spsalmon.github.io/align_pipeline/usage/runningonacluster/).

## Temporary files and cleanup

| Key | Default | Meaning |
| --- | --- | --- |
| `temp_dir` | `./temp_files` | Scratch directory for the run: generated job scripts, logs, and intermediate state. Also settable with `-t / --temp_dir`. |
| `cleanup_on_success` | `False` | Delete this run's scratch directory once the pipeline has completed successfully. |

The results of a run are **never** in the temporary directory — they are in your
experiment's analysis directory, and a full copy of the run's scratch (logs,
configuration, provenance) is kept in the experiment as well. See
[pipeline output](https://spsalmon.github.io/align_pipeline/usage/pipelineoutput/).

Point `temp_dir` at your data storage if your home directory has a small quota.

## Referring to folders

Many block options point at a folder produced by an earlier block: the masks used
for straightening, the images used for quality control, and so on. These are
written **by name**, without the analysis directory prefix:

```yaml
straightening_masks: [ "ch2_seg" ]
```

`ch2_seg` and `analysis/ch2_seg` mean the same thing, so old configurations keep
working — but the short form is preferred, because it survives renaming
`analysis_dir_name`. Two references are special:

- `raw` always means the raw images.
- An absolute path is used exactly as written.

## Writing options for several blocks at once

Every block option is a list. If you have several blocks of the same type, an
option with **one element** is used by all of them; an option with **as many
elements as blocks** is distributed among them in order:

```yaml
building_blocks:
  - "segmentation"
  - "segmentation"
segmentation_method: [ "deep_learning" ]          # both blocks
segmentation_channels: [ [ 1 ], [ 0 ] ]           # first block, then second
```

Any other length is an error. Use `null` to leave an option empty.

## Checks made before the run starts

The configuration is validated up front, before any job is submitted or any
folder is created. If something is wrong, the run stops immediately and **all**
the problems are reported together. The checks are:

- required keys (`experiment_dir`, `building_blocks`) are present;
- every entry of `building_blocks` is a known block name;
- every per-block option has one value per block, or a single value to share;
- `backend` and `report_format` have allowed values;
- there are no unknown or misspelled top-level keys (`pixlesize` is rejected
  rather than silently ignored);
- the input files and folders you provided exist — `experiment_dir`, and the
  model and script paths.

Folders that an earlier block will *produce* are deliberately not checked: they
do not exist yet when the configuration is validated. A reference pointing at a
folder nothing produces shows up at run time as "no input files".
