# Training Molt Detection

The molt detection model finds the developmental events (hatch and the four molts
M1–M4) of each worm from its growth curve. It is a **1D keypoint detection** model:
instead of looking at images, it takes the per-worm time series (log body volume and
its instantaneous growth rate) and predicts, for each molt, a heatmap of *where* it
happens along the series.

Training happens in two steps, each driven by its own YAML configuration file:

1. **Gathering the dataset** — aggregate the already-annotated growth curves from many
   experiments into a set of training pickles (`get_molt_detection_data.py`).
2. **Training the model** — train the keypoint detection model on those pickles
   (`train_molt_detection.py`).

Both scripts live in `training/molt_detection/`, with example configs in
`training/molt_detection/configs/`.

## Gathering the dataset

This step scans the storage cluster for experiments that have been segmented,
straightened, and **manually annotated** with their molts (via the GUI), and turns each
valid worm into a training sample. The ground-truth molt frames are converted into
Gaussian "keymaps" that the model learns to reproduce.

An example configuration is kept up to date under
`training/molt_detection/configs/molt_dataset_config.yaml`.

```yaml
storage_cluster_path: "/mnt/towbin.data/shared"
experimentalists: ["kstojanovski", "plenart", "igheor", "spsalmon"]
scopes: ["ti2", "orca", "lipsi", "crest", "squid"]

min_year: 2024
magnifications: ["10x", "10X"]

worm_type_column: "ch2_seg_str_worm_type"
volume_column: "ch2_seg_str_volume"

sigma_frames: 2.0
keymap_type: "gaussian"

min_series_length: 200
max_series_length: 1000

output_dir: "./molt_detection_dataset"
```

- **storage_cluster_path** : the root directory that contains one subdirectory per
  experimentalist.
- **experimentalists** : the experimentalist subdirectories to scan for experiments.
- **scopes** : only keep experiments whose directory name mentions one of these
  microscopes (case-insensitive variations are handled automatically).
- **min_year** : only keep experiments whose directory name starts with a year greater
  than or equal to this value.
- **magnifications** : only keep experiments whose directory name contains one of these
  magnifications. Molt detection is trained on low-magnification whole-worm imaging.
- **worm_type_column** : the quality-control column used to clean the growth curve
  before computing features. If the column is not found, the script falls back to
  `ch2_seg_str_qc`.
- **volume_column** : the body-volume column used to build the features (log volume and
  log-volume growth rate).
- **sigma_frames** : the width (in frames) of the Gaussian bump placed at each ground
  truth molt when building the target keymaps.
- **keymap_type** : the type of ground-truth keymap. Used to name the output file
  (`y_<keymap_type>_molt_detection.pickle`).
- **min_series_length** / **max_series_length** : only keep worms whose time series
  length falls within these bounds. This removes truncated or abnormally long series.
- **output_dir** : where to write the three dataset pickles
  (`X_molt_detection.pickle`, `y_<keymap_type>_molt_detection.pickle`,
  `keypoints_molt_detection.pickle`).

This step walks the whole storage cluster and reads many filemaps, so it is submitted as
a SLURM job:

```bash
cd ~/align_pipeline/training/molt_detection
bash run_gather_dataset.sh -c configs/molt_dataset_config.yaml
```

Gathering the dataset is a prerequisite for training: the training script checks for the
dataset pickles and refuses to start if they are missing.

## Training the model

Once the dataset is gathered, training reads the pickles and fits the keypoint detection
model. An example configuration is kept up to date under
`training/molt_detection/configs/molt_training_config.yaml`.

```yaml
dataset_dir: "./molt_detection_dataset"
keymap_type: "gaussian"
# features_pickle: null
# heatmaps_pickle: null
# keypoints_pickle: null

save_dir: "./model_checkpoints"
model_name: "molt_detection_model"

input_channels: 2
n_classes: 4
activation: "sigmoid"
learning_rate: 1.0e-4

enforce_divisibility_by: 32
resize_method: "pad"

batch_size: 64
max_epochs: 50
num_workers: 32
accumulate_grad_batches: 1
train_test_split_ratio: 0.2
save_best_k_models: 1
swa_lrs: 1.0e-2
random_state: 42
```

- **dataset_dir** : the directory the gathering step wrote to. The three pickles are
  resolved automatically from it, using `keymap_type` to find the target file. Instead
  of `dataset_dir`, you may set the three explicit pickle paths (`features_pickle`,
  `heatmaps_pickle`, `keypoints_pickle`).
- **save_dir** / **model_name** : the trained model is saved to
  `<save_dir>/<model_name>/`, together with a copy of the config, the best checkpoint(s),
  and a lightweight checkpoint (`best_light.ckpt`) suitable for deployment.
- **input_channels** : the number of feature channels per sample. The gathering step
  produces two (log volume and log-volume growth rate), so this should be `2`.
- **n_classes** : the number of molts to detect. Should be `4` (M1–M4); hatch is derived
  separately downstream.
- **activation** : the output activation of the model.
- **learning_rate** : the learning rate to use for training.
- **enforce_divisibility_by** : the model requires series lengths divisible by this
  value; shorter series are padded up to the next multiple.
- **resize_method** : how variable-length series are made batchable. `pad` pads to the
  longest series in the batch and feeds the model a validity mask.
- **batch_size** : the batch size to use for training.
- **max_epochs** : the maximum number of epochs to train for.
- **num_workers** : the number of workers to use for loading the data.
- **accumulate_grad_batches** : the number of batches to accumulate gradients over before
  an optimizer step (an effective way to increase the batch size without using more
  memory).
- **train_test_split_ratio** : the fraction of samples held out for validation.
- **save_best_k_models** : the number of best checkpoints to keep (by validation loss).
- **swa_lrs** : the learning rate used by Stochastic Weight Averaging.
- **random_state** : the seed used for the train/validation split, for reproducibility.

Once your configuration is finished, submit the training as a SLURM job:

```bash
cd ~/align_pipeline/training/molt_detection
bash run_training.sh -c configs/molt_training_config.yaml
```

## Using the trained model in the pipeline

Training produces a lightweight checkpoint at
`<save_dir>/<model_name>/best_light.ckpt`. To use it in the pipeline, point the
`molt_detection` building block at it:

```yaml
building_blocks:
  - "molt_detection"

molt_detection_method: ["deep_learning"]
molt_detection_model_path: ["/path/to/best_light.ckpt"]
molt_detection_columns: [["ch2_seg_str_volume"]]
```

The default checkpoint shipped with the pipeline lives inside the package, at
`align_pipeline/defaults/models/molt_detection_model.ckpt`, and is used when
`molt_detection_model_path` is left unset. See the
[Molt Detection building block](https://spsalmon.github.io/align_pipeline/building-blocks/moltdetection)
documentation for the full list of options.

The two notebooks in `training/molt_detection/utils/` help you inspect the gathered
dataset (`explore_molt_detection_data.ipynb`) and evaluate a trained model on the
held-out split (`test_molt_detection.ipynb`).
