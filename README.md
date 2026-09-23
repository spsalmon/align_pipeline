# ALIGN

ALIGN (Automated Longitudinal Image-based Growth aNalyzer) is a pipeline for processing and analyzing time lapse microscopy experiments. It implements many of the functions present in the ALIGN package and bundles them with a pipelining tool to easily and reproducibly process large longitudinal experiments.

A detailed documentation of the pipeline can be found here : <https://spsalmon.github.io/align_pipeline/>
The documentation for the package used as a backbone for the pipeline can be found here : <https://align-toolbox.readthedocs.io/en/latest/align_toolbox.html>

## RTFM

## How to install ?

You will find detailed explanations on how to install, update and use the pipeline here : <https://spsalmon.github.io/align_pipeline/getting-started/installation/>

Short version, on a cluster:

```bash
git clone https://github.com/spsalmon/align_pipeline.git
cd align_pipeline
bash scripts/install_pipeline.sh
```

On your own machine (conda, no cluster needed):

```bash
bash scripts/install_pipeline_local.sh
conda activate align_local
```

## Running the pipeline

You will find a detailed explanation on how to run the pipeline here : <https://spsalmon.github.io/align_pipeline/getting-started/runningfirstpipeline/>

Short version:

```bash
align init-configs ~/my_configs   # get a config to start from
# edit ~/my_configs/config.yaml
bash scripts/run_pipeline.sh -c ~/my_configs/config.yaml
```

Without a cluster, set `backend: "local"` in the config and run it directly:

```bash
align run ~/my_configs/config.yaml
```

## Updating the pipeline

You will find a detailed explanation on how to update the pipeline here : <https://spsalmon.github.io/align_pipeline/getting-started/update/>

```bash
bash scripts/update_pipeline.sh                  # code + environment
bash scripts/update_pipeline.sh --pipeline-only  # code only, much faster
```

## What is in this repository

```
align_pipeline/   The pipeline itself (installable python package)
  workers/                one worker per analysis step
  defaults/               bundled example configs + default models
scripts/                Operate the pipeline: install, update, run, cleanup
env/                    Define and build the conda environment
gui/                    The Shiny annotation GUI
training/               Retrain the segmentation, QC and molt detection models
tools/                  One-off helpers (ND2/SQUID/MATLAB conversion, ...)
custom_workers/         Example scripts for the "custom" building block
analysis_and_plots/     Notebooks for downstream analysis and plotting
configs/                Working configurations
book/                   Source of the documentation website
tests/                  Automated tests
```

The four things to know:

- **`align_pipeline/`** is the pipeline. It is installed as a package, so
  the `align` command (alias `align_pipeline`) works from any directory.
- **`scripts/` and `env/`** are the glue that installs and runs it. `env/` builds
  the environment, `scripts/` operates the pipeline.
- **`gui/`, `training/`, `tools/`** are extras, each with its own launch scripts.
- Everything else is documentation, configurations and tests.

## How to set up Visual Studio Code ? (for members of the IZB)

1. Download VS Code : <https://code.visualstudio.com/download>
2. Install it like you would install any software.
3. Inside of VS Code, open a terminal and run :

```bash
code --install-extension ms-vscode-remote.remote-ssh
```

Now, click on the remote explorer icon that should be on the left of the window and click on the + to add a new remote.
Enter the command you usually use to ssh into the cluster using PuTTY, for example:

```bash
ssh username@izblisbon.unibe.ch
```

Obviously, change username to your username (first letter of your first name + last name, eg : spsalmon)

Optionnal, but **HIGHLY** recommended. Open the Windows command line (cmd). Run :

```bash
ssh-keygen
```

- Select all the default options, except if you are extremely paranoid and want to set a passphrase.
  Go into the folder where the file was saved, it should be something like Users/username/.ssh/

- Open the file **id_rsa.pub** using the notepad or any text editing software.
  Copy the entire content of the file.

- In VS Code, go to your home folder : /home/username/

- Go into the .ssh folder

- If it doesn't exist, create a file named **authorized_keys**

- Paste the content of the **id_rsa.pub** file that you copied earlier into this file

- You will now be able to connect to the cluster without having to type your password

If you want to code using Python, you should run the following commands, while connected inside of VS Code, while being connected to your session on the cluster.

```bash
code --install-extension ms-python.python
```

```bash
code --install-extension ms-toolsai.jupyter
```

```bash
code --install-extension ms-python.vscode-pylance
```
