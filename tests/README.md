# Tests

## Local pipeline smoke test

`test_local_pipeline.py` runs the pipeline end-to-end with `backend: local`
(no slurm, no micromamba). It generates tiny synthetic images at runtime,
runs threshold segmentation followed by area morphology, and checks that the
masks and the morphology report are produced.

Requires an environment with the pipeline dependencies (e.g. `ALIGN`,
`tifffile`, `pyyaml`). Run from the repo root:

```bash
python -m pytest tests/ -v
```

The test skips itself automatically if the dependencies are not installed.
