# Running your first pipeline

A pipeline run needs two things: an **experiment directory** containing your raw
images, and a **configuration file** describing what to do with them.

## 1. Get a configuration to start from

The pipeline ships with a working example configuration. Copy it into a folder of
your choice with:

```bash
align_pipeline init-configs ~/my_configs
```

This writes two files:

- `config.yaml` — the analysis configuration (this is the one you will edit)
- `slurm_config.yaml` — how much memory, CPU and time each step gets on the
  cluster. You can ignore it until you need to change it, see
  [running on a cluster](https://spsalmon.github.io/align_pipeline/usage/runningonacluster/).

Both are copied because the main configuration refers to the SLURM one by name.

```{tip}
Never edit the bundled configuration inside the pipeline folder
(`align_pipeline/defaults/configs/`) — it is reset to its upstream version
every time you update. Always work on a copy.
```

## 2. Understand what you are editing

The pipeline is configured with YAML files. The basic assumption is that your
images follow the naming scheme `TimeX_PointY_(...).tiff`, where `Time` refers to
the index in your time loop and `Point` is the unique identifier of the position
(one individual worm, if each position contains one worm). All images (planes,
channels, etc.) for a given position at a given time should be in the same
OME-TIFF file. For more details, see [pipeline input](https://spsalmon.github.io/align_pipeline/usage/pipelineinput/).

Let's break the configuration down.

### Where the data is

```yaml
experiment_dir: "/mnt/towbin.data/shared/spsalmon/pipeline_test_folder/"
analysis_dir_name: "analysis"
raw_dir_name: "raw"
report_format: "parquet"
pixelsize: [ 0.65 ]
get_experiment_time: True
time_regex: 'Time(\d+)'
point_regex: 'Point(\d+)'
```

- **experiment_dir**: the root of your experiment.
- **analysis_dir_name**: the directory where all the analysis files (segmentation
  masks, quantifications, etc.) will be saved. Change it to run several different
  analyses of the same experiment.
- **raw_dir_name**: the directory where your raw images are saved.
- **report_format**: either `"csv"` or `"parquet"`. Parquet files are much smaller
  than CSVs (useful for big experiments) but less convenient to edit.
- **pixelsize**: physical size of a pixel in µm (depends on your microscope,
  camera and objective).
- **get_experiment_time**: if True, the actual acquisition time of each image is
  extracted from the metadata. This takes a while but is very useful downstream.
- **time_regex** / **point_regex**: the regular expressions used to extract the
  time and point indices from the file names. The defaults work for names like
  `TimeX_PointY_(...).tiff`. The part in brackets is what gets extracted.

If you have different imaging modalities during your timelapse (say, a picture of
each worm every 10 minutes and a Z-stack every hour), split them into different
raw directories (e.g. `raw` and `raw_stack`), run a pipeline for each, and merge
the results at the end by joining the two dataframes.

### Where it runs

```yaml
backend: "slurm"
n_jobs: 32
slurm_config: "slurm_config.yaml"
```

- **backend**: `"slurm"` (the default) submits each step as a cluster job.
  `"local"` runs the steps one after the other on the machine you are on — use
  this if you don't have a cluster.
- **n_jobs**: how many images are processed in parallel inside a step.
- **slurm_config**: the file holding the cluster resource requests. Only used with
  the SLURM backend, and the path is relative to your configuration file.

### What it does

```yaml
building_blocks:
  - "segmentation"
  - "segmentation"
  - "straightening"
  - "straightening"
  - "straightening"
  - "straightening"
  - "morphology_computation"
  - "morphology_computation"
  - "fluorescence_quantification"
  - "quality_control"
  - "quality_control"
  - "molt_detection"

rerun_segmentation: [ False ]
rerun_straightening: [ False ]
rerun_morphology_computation: [ False ]
rerun_fluorescence_quantification: [ False ]
rerun_quality_control: [ False ]
rerun_molt_detection: [ False ]
```

- **building_blocks**: the list of atomic tasks you want the pipeline to perform.
  Here: 2 segmentations, 4 straightenings, and so on. They run in the order they
  are written.
- **rerun_...**: if False, images that were already processed are skipped and only
  the missing ones are processed. For blocks producing a single report file (like
  `morphology_computation`), the whole block is skipped if that file already
  exists. If True, everything is reprocessed.

Then come the parameters of each block. They are described in detail in the
[building blocks](https://spsalmon.github.io/align_pipeline/building-blocks/buildingblock/)
section. Here is the configuration of the two segmentation blocks:

```yaml
# segmentation parameters
segmentation_column: [ "raw" ]
segmentation_name_suffix: [ null ]
segmentation_method: [ "deep_learning" ]
segmentation_channels: [ [ 1 ], [ 0 ] ]

# deep learning segmentation parameters
model_path: [ "/mnt/.../body/best_light.ckpt", "/mnt/.../pharynx/best_light.ckpt" ]
batch_size: [ 4 ]
```

Every option in the full list of configuration keys is described in
[configuration](https://spsalmon.github.io/align_pipeline/usage/configuration/).

## 3. Run it

Save your configuration anywhere you like — a single folder centralising all your
configurations is a good idea. Assuming you saved it as
`~/my_configs/my_experiment.yaml`:

**On the cluster:**

```bash
cd ~/align_pipeline # or wherever you put the pipeline folder
bash scripts/run_pipeline.sh -c ~/my_configs/my_experiment.yaml
```

**On your own machine** (with `backend: "local"` in the config), from anywhere:

```bash
align_pipeline run ~/my_configs/my_experiment.yaml
```

The `-c` argument specifies the configuration to run. Two more optional arguments
are useful:

- `-e / --experiment_dir` — run the same configuration on another experiment,
  without editing the file.
- `-t / --temp_dir` — where the pipeline puts its temporary files (defaults to
  `temp_files/` inside the pipeline folder).

So, to analyse a second experiment with the exact same settings:

```bash
bash scripts/run_pipeline.sh -c ~/my_configs/my_experiment.yaml -e /path/to/other_experiment
```

## 4. Watch it

The configuration is checked before anything is submitted, so a typo or a missing
file is reported immediately, in your terminal, with every problem listed at once.

Once the run starts, it prints its list of steps and then a marker around each
one, so you can always tell where it is. See
[monitoring a run](https://spsalmon.github.io/align_pipeline/usage/monitoringruns/)
for where the logs live and what to do when something goes wrong.

That's it! Once you are happy with your configuration, analysing a new experiment
is a single command.
