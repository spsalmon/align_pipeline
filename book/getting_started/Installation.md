# Installing the pipeline

The pipeline can be installed in two ways:

- **On a computing cluster** (the usual case in the lab). Uses micromamba and a
  locked environment, so everyone runs exactly the same package versions.
- **On your own machine**, with conda and no micromamba. Useful for testing a
  config, running small experiments, or working without a cluster.

Both give you the same pipeline. Pick one.

## On a cluster (Linux)

1. Clone the pipeline repository:

```bash
cd
git clone https://github.com/spsalmon/align_pipeline.git
```

2. Install micromamba. **BE CAREFUL, THE PIPELINE EXPECTS YOU TO CHOOSE THE BASE
OPTIONS AND YES ALL THE TIME.** Then restart your shell. Skip this step if you
already have micromamba.

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
```

3. Run the installation script:

```bash
cd ~/align_pipeline
bash scripts/install_pipeline.sh
```

This creates a micromamba environment called `ALIGN` containing all the
required packages, and registers the pipeline itself inside it so that it can be
run from anywhere.

That's it. Head to [running your first pipeline](https://spsalmon.github.io/align_pipeline/getting-started/runningfirstpipeline/),
or read how to [update the pipeline](https://spsalmon.github.io/align_pipeline/getting-started/update/).

## On your own machine

This path only needs `conda` (Anaconda, Miniconda or Miniforge) and works on any
operating system. It does not need micromamba and does not need a cluster.

1. Clone the repository (see above).

2. Run the local installation script:

```bash
cd ~/align_pipeline
bash scripts/install_pipeline_local.sh
```

3. Activate the environment:

```bash
conda activate align_local
```

If you prefer to do it by hand, the script is equivalent to:

```bash
cd ~/align_pipeline
conda env create -f env/environment_local.yml
conda activate align_local
pip install -e ".[dev]"
```

When running on your own machine, set `backend: "local"` in your configuration
so the pipeline runs the blocks one after the other instead of submitting them
to SLURM. See [configuration](https://spsalmon.github.io/align_pipeline/usage/configuration/).

## Windows

The easiest way to get things to work on Windows is to use the Windows Subsystem
for Linux (WSL): follow the [installation instructions](https://learn.microsoft.com/en-us/windows/wsl/install).

1. In a PowerShell terminal run as administrator:

```PowerShell
wsl --install -d Ubuntu
wsl --set-default-version 2
```

2. To use WSL, in a terminal, run:

```PowerShell
wsl
```

3. Follow the Linux instructions above from inside WSL.

Alternatively, the local install works directly on Windows with conda, as long as
you run the commands from a bash shell (Git Bash).

## Checking that it worked

Once the environment is active, this should print the available commands:

```bash
align
```

`align_pipeline` (as well as the legacy `tt-pipeline` and `ttp`) is an alias for
exactly the same command.
