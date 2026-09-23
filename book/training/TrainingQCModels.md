# Training Quality Control Models

Once you have a labelled QC dataset (see
[gathering a quality control dataset](https://spsalmon.github.io/align_pipeline/training/gatheringqcdataset)),
`train_qc_xgb_model.py` trains an **XGBoost** classifier that labels each
segmented worm as a valid worm, an egg, or an error.

The script computes shape/intensity features for every annotated image with
`compute_qc_features` (from the `ALIGN` library), optionally tunes the
XGBoost hyperparameters with Bayesian optimization, fits the model with
balanced class weights, and saves a small model bundle that the pipeline's
`quality_control` block consumes.

The script lives in `training/classification/`, with an example config under
`training/classification/configs/qc_training_config.yaml`.

## Dataset directory layout

The `dataset_path` you point at must contain:

- `project.yaml` — with a `classes` list.
- `images/` and `masks/` — the extracted crops.
- `annotations/annotations.csv` — with at least `ImagePath` and `Class`
  columns, produced by the annotation tool.

Extracted features are cached as `features.csv` (and `processed_annotations.csv`)
inside `dataset_path`, so re-running is fast unless `rerun_feature_extraction`
is set.

## Example configuration

```yaml
dataset_path: "/mnt/towbin.data/shared/spsalmon/towbinlab_classification_database/datasets/10x_pharynx_qc/pharynx/"
output_path: "/mnt/towbin.data/shared/spsalmon/towbinlab_classification_database/models_new/10x_pharynx_qc/"

model_name: "qc_xgb_model.pkl"
qc_classifier_name: "qc_xgb_model.json"
egg_classifier_name: "egg_xgb_model.json"

optimize_hyperparameters: true
mask_only: false
train_egg_detector: false
rerun_feature_extraction: true

n_points: 10
n_iter: 50

test_split_ratio: 0.2
random_state: 42
```

- **dataset_path** : the dataset directory described above.
- **output_path** : where the trained model bundle and classifiers are written.
  A copy of the config is saved here too.
- **model_name** : the joblib bundle the pipeline loads. It records the
  classifier file name(s), the class list, and whether the model is
  `mask_only`.
- **qc_classifier_name** / **egg_classifier_name** : file names of the raw
  XGBoost classifiers inside `output_path` (`egg_classifier_name` is only
  written when `train_egg_detector` is true).
- **optimize_hyperparameters** : if true, run Bayesian optimization over the
  XGBoost hyperparameters before fitting the final model; otherwise fit with
  defaults.
- **mask_only** : compute features from the mask alone (ignore the intensity
  image). The value is stored in the bundle so inference uses the same inputs.
- **train_egg_detector** : if true, also train a separate egg vs. non-egg
  classifier and fold the remaining classes into `worm`/`error` for the QC
  model. Off by default.
- **rerun_feature_extraction** : recompute features even if `features.csv`
  already exists.
- **n_points** / **n_iter** : the Bayesian optimization budget (initial random
  points and optimization iterations). Only used when
  `optimize_hyperparameters` is true.
- **test_split_ratio** : the fraction of samples held out for validation.
- **random_state** : the seed used for the train/validation split.

## Running

XGBoost training and the Bayesian search are CPU-only, so no GPU is requested:

```bash
cd ~/align_pipeline/training/classification
bash run_training.sh -c configs/qc_training_config.yaml
```

The script prints a validation F1 score and a full classification report when
it finishes.

## Using the trained model in the pipeline

Training writes a bundle at `<output_path>/<model_name>` (e.g.
`qc_xgb_model.pkl`). Point the `quality_control` building block's
`qc_model_path` at it, alongside the images/masks it should classify:

```yaml
building_blocks:
  - "quality_control"

qc_masks: [ "analysis/ch2_seg_str" ]
qc_images: [ [ "analysis/ch2_raw_str", null ] ]
qc_model_path: [ "/path/to/qc_xgb_model.pkl" ]
```

There is no separate "method" option: the block always loads the joblib
bundle at `qc_model_path` and uses whichever classifier(s) it points to. See
`OPTIONS_MAP["quality_control"]` in `align_pipeline/building_blocks.py` for
the full list of accepted keys (`qc_masks`, `qc_images`, `qc_model_path`,
`qc_import_eggs_from`, `rerun_quality_control`).

The default checkpoints shipped with the pipeline live under
`align_pipeline/defaults/models/10x_body_qc/` and
`align_pipeline/defaults/models/10x_pharynx_qc/`. See the
[Quality Control building block](https://spsalmon.github.io/align_pipeline/building-blocks/qualitycontrol)
documentation for the full list of options.
