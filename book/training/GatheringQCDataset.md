# Gathering a Quality Control Dataset

The quality-control (QC) classifier decides, for each worm image, whether the
segmentation is a valid **worm**, an **egg**, or an **error**. Training it starts
by building a labelled image dataset drawn from many already-analysed
experiments.

The gathering step (`get_qc_data.py`) scans the storage cluster for experiments
that match your criteria, turns each into candidate samples, balances them by
class, and copies the selected images and masks into a dataset directory ready
for annotation and training.

The script lives in `training/classification/`, with example configs under
`training/classification/configs/`.

## Two modes

- **Labelled mode** (`class_proportions` is set): the script uses the existing
  growth curves and QC annotations to propose class labels. It fits a smooth
  growth curve per point, flags timepoints whose residual exceeds a quantile
  `threshold` as `potential_error`, keeps confirmed `error` and `egg`
  annotations, and subsamples `good` points. Images are then sampled to match
  `class_proportions`.
- **Unlabelled mode** (`class_proportions: null`): the script simply samples
  valid images (non-NaN, non-zero volume where available) to reach the requested
  size, labelling everything `unknown` for manual annotation later.

## Example configuration

An example configuration lives under
`training/classification/configs/qc_dataset_config.yaml`; the block below is
an illustrative, trimmed version.

```yaml
storage_path: "/mnt/towbin.data/shared"
valid_subdirectories: ["plenart", "kstojanovski"]

database_path: "/mnt/towbin.data/shared/spsalmon/towbinlab_classification_database/datasets/10x_pharynx_qc"

database_configs: {
  'pharynx': {
      'size': 2000,
      'channel': [0],
      'strains': ["186", "160", "125", "318", "446"],
      'magnifications': ["10x"],
  },
}

class_proportions: {"error": 0.0, "potential_error": 0.5, "good": 0.5, "egg": 0.0}
lambda: 0.0075
threshold: 0.90

extra_adulthood_time: 40

valid_scopes: ["squid", "ti2"]
scopes_alt_names: {
  'crest': ['Crest', 'crest', 'CREST'],
  'squid': ['Squid', 'squid', 'SQUID'],
  'ti2': ['Ti2', 'ti2', 'TI2', 'orca', 'Orca', 'ORCA'],
  'ziva': ['ziva', 'ZIVA', 'Ziva']
}

keywords_to_exclude: ["exclude", "fail", "failure", "crash"]
keywords_to_include: []

experiments_to_consider: []
experiments_to_always_include: []
experiments_to_exclude: []
```

- **storage_path** : the root directory containing one subdirectory per
  experimentalist.
- **valid_subdirectories** : the experimentalist subdirectories to scan.
- **database_path** : where the dataset is written. One subdirectory is created
  per key in `database_configs`, each with `images/` and `masks/`.
- **database_configs** : one entry per sub-dataset to build. Each entry accepts:
  - **size** : the target number of images for that sub-dataset.
  - **channel** : the raw channel(s) to extract (0-indexed).
  - **strains** : optional list of strain numbers to restrict to.
  - **magnifications** : optional list of magnification strings to restrict to
    (e.g. `["10x"]`); case variations are generated automatically. Note the
    key is plural and takes a list — a singular `magnification: "10x"` entry is
    not read and silently disables the filter, so check older configs for it.
  - **stage_proportions** : optional. Only its presence matters (its per-stage
    values are not otherwise used): setting it restricts the scan to
    GUI-annotated filemaps (`analysis_filemap_annotated`) instead of all
    filemaps.
- **class_proportions** : the target proportion of each class
  (`error`, `potential_error`, `good`, `egg`). Set to `null` for unlabelled
  mode.
- **lambda** : smoothing strength of the growth-curve smoother used to compute
  residuals (labelled mode only).
- **threshold** : residual quantile above which a timepoint is flagged
  `potential_error` (labelled mode only).
- **extra_adulthood_time** : how many frames after M4 to keep (points annotated
  with M4). Limits how far into adulthood images are sampled (labelled mode
  only).
- **valid_scopes** : only keep experiments whose directory name mentions one of
  these microscopes.
- **scopes_alt_names** : maps each canonical scope name to the spelling
  variations found in directory names.
- **keywords_to_exclude** / **keywords_to_include** : drop or force-keep
  experiments whose directory name contains these keywords.
- **experiments_to_consider** : if non-empty, scan only these exact experiment
  paths instead of walking `valid_subdirectories`.
- **experiments_to_always_include** / **experiments_to_exclude** : force-include
  or force-exclude experiments by name.

## Running

This step reads many filemaps across the cluster, so it is submitted as a SLURM
job:

```bash
cd ~/align_pipeline/training/classification
bash run_gather_dataset.sh -c configs/qc_dataset_config.yaml
```

The output dataset directory is what you point the training step at. See
[training a quality control model](https://spsalmon.github.io/align_pipeline/training/trainingqcmodels).
