# Updating the pipeline

## Fast explanation

A script does everything described below. To update the pipeline:

1. Deactivate the environment, and close the scripts, notebooks and terminals
   where it is activated.
2. Run the update script:

```bash
cd ~/align_pipeline # (or wherever you installed the pipeline)
bash scripts/update_pipeline.sh
```

Optionally, you can run:

```bash
bash scripts/update_pipeline.sh --pipeline-only
```

to update only the pipeline code and not the underlying packages. This is much
faster, and is what you want most of the time.

```{note}
Running the pipeline no longer updates it for you. Since the refactor,
`scripts/run_pipeline.sh` never touches your working copy — updating is always
an explicit choice.
```

## Longer explanation

Updating is done in two parts.

First, the repository is updated from the source on GitHub. This can be done in
two ways:

1. using `git pull`
2. by resetting the repository to the upstream content:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

If what you want is to reset the repository to how it is on GitHub, use option 2.
**This is what the update script does**, so any local modification you made to the
pipeline files will be lost.

Then, the packages are updated. For consistency's sake, the pipeline uses conda
lock files. They ensure that the package versions you end up with are exactly the
same as what was tested during development. To update your environment according
to the lock file, run:

```bash
cd ~/align_pipeline
micromamba run -n align conda-lock install --name align ./env/conda-lock.yml
```

Finally, the pipeline package itself is registered in the fresh environment:

```bash
micromamba run -n align pip install -e . --no-deps
```

This last step is only needed when the environment was rebuilt. The pipeline is
installed in "editable" mode, which means it always follows the files in your
`align_pipeline` folder — a `--pipeline-only` update therefore needs
nothing more than the `git` step.

## Troubleshooting

If for some reason you end up with a broken environment, you can always delete it
and create it again. To do so, run:

```bash
micromamba env remove -n align
cd ~/align_pipeline
bash scripts/install_pipeline.sh
```

You may also want to update the ALIGN package itself manually. To do so,
run:

```bash
micromamba run -n align pip install -U align_toolbox
```

## Updating a local install

On a local (conda) install, update the repository and reinstall the package:

```bash
cd ~/align_pipeline
git pull
conda activate align_local
pip install -e ".[dev]"
```
