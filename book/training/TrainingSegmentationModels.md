# Training Segmentation Models

Once again, this is configured through a YAML configuration file. You will find an example (kept up to date with every update) under `training/segmentation/configs/segmentation_training_config.yaml`. Let's break it down.

```yaml
training_dataframes:
  - '/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/align/germline/training_dataframe.csv'

validation_dataframes:
  - '/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/align/germline/validation_dataframe.csv'

test_dataframes:
  - '/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/align/germline/test_dataframe.csv'

# if you don't have dataframes yet, you may input directories directly
# image_directories: ["/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/datasets/chamber_segmentation/brightfield/good_images"]
# mask_directories: ["/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/datasets/chamber_segmentation/brightfield/binarized_and_cleaned_masks"]

save_dir: '/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/germline/'
model_name: 'align_medium'
continue_training_from_checkpoint: null
```

- **dataframes** : path to the training, validation, and test dataframes containing the paths to the images and associated ground truth masks.
- **directories** : if you don't have dataframes yet, you can directly input directories containing the raw images and the associated masks. The pipeline will then automatically generate the dataframes for you. Those dataframes can then be reused for future trainings.
- **save_dir** : where to save the trained model.
- **model_name** : the name of the model, which will be used to name the saved checkpoint files.
- **continue_training_from_checkpoint** : initialize the model with weights from a previous checkpoint. This is useful if you want to continue training a model that was already trained for some epochs, or if you want to fine-tune a model that was trained on a different dataset. Importantly, the checkpoint you load MUST have the exact same architecture as the model you currently train (same number of classes, number of input channels, encoder, etc.)

```yaml
n_classes: 1
channels_to_segment: [0]

architecture: 'UnetPlusPlus'
encoder: 'efficientnet-b5'
pretrained_weights: 'imagenet'
```

- **n_classes** : the number of classes to segment. Should be 1 for binary segmentation, and >1 for multi-class segmentation (background is not counted as a class).
- **channels_to_segment** : the channels to segment. For example, if you have multichannel images and you only want to segment the first channel, you would set this to [0]. If you want to segment the first and third channels, you would set this to [0, 2].
- **architecture** : the architecture of the model. See [SMP documentation](https://smp.readthedocs.io/en/latest/models.html) for the available architectures.
- **encoder** : the encoder to use for the model. See [SMP documentation](https://smp.readthedocs.io/en/latest/encoders.html) for the available encoders.
- **pretrained_weights** : the pretrained weights to use for the encoder (e.g. `imagenet`). See [SMP documentation](https://smp.readthedocs.io/en/latest/encoders.html) for the available pretrained weights for each encoder.

```yaml
learning_rate: 1.0e-4
normalization_parameters: {'type': 'percentile', 'lo': 1, 'hi': 99, 'axis': [-2, -1]}

train_on_tiles: True
tiler_config: {'tile_size':[1024, 1024], 'tile_step':[256, 256]}

loss: 'FocalTversky'
value_to_ignore: -1
```

- **learning_rate** : the learning rate to use for training.
- **normalization_parameters** : the parameters to use for normalizing the images. `type` can be either `percentile` or `mean_std` or `data_range`. If `percentile`, in normalized image, 1 will correspond to the `hi`th percentile of the original image and 0 will correspond to the `lo`th percentile of the original image. If `mean_std`, the image will be normalized so that it has the specified mean and standard deviation. If `data_range`, values will be simply rescaled to be between 0 and 1.
- **train_on_tiles** : whether to train on tiles of the images instead of the full images. This can be useful if the images are very large and cannot fit into memory, or if the images in your training set vary in size.
- **tiler_config** : the parameters to use for tiling the images if `train_on_tiles` is set to True. `tile_size` is the size of the tiles, and `tile_step` is the step size between tiles (i.e. how much the tiles overlap).
- **loss** : the loss function to use for training. Available options are `BCE` and `FocalTversky` (for binary segmentation), `CrossEntropy` and `MultiClassFocalLoss` (for multi-class segmentation).
- **value_to_ignore** : the value in the ground truth masks to ignore during training. This can be useful if you have pixels in your ground truth masks that are not annotated (e.g. because they are ambiguous or because they contain artifacts). Those pixels will be ignored during the calculation of the loss and will not contribute to the training of the model.

```yaml
max_epochs: 500
batch_size: 8
accumulate_grad_batches: 3
num_workers: 24
deterministic: True # slower training but ensures reproducibility

save_best_k_models: 1
train_val_split_ratio: 0.20
train_test_split_ratio: 0.1
```

- **max_epochs** : the maximum number of epochs to train the model for.
- **batch_size** : the batch size to use for training.
- **accumulate_grad_batches** : the number of batches to accumulate gradients for before performing an optimizer step. This can be useful if you want to effectively increase the batch size without increasing the memory usage (e.g. if you want to use a batch size of 24 but can only fit 8 images in memory, you can set `batch_size` to 8 and `accumulate_grad_batches` to 3).
- **num_workers** : the number of workers to use for loading the data. This can speed up data loading, especially if you have a large dataset and a fast storage system.
- **deterministic** : whether to Pytorch Lightning's deterministic mode, which ensures reproducibility at the cost of slower training.
- **save_best_k_models** : the number of best models to save during training. The best models are determined based on the validation loss. If set to 1, only the best model will be saved. If set to 3, the three best models will be saved, etc.
- **train_val_split_ratio** : the ratio of the training set to use for validation if no separate validation dataframe is provided. For example, if set to 0.2, 20% of the training data will be used for validation and 80% will be used for training.
- **train_test_split_ratio** : the ratio of the training set to use for testing if no separate test dataframe is provided.

## Running the training

Once you configuration is finished, you can run the training with the following command:

```bash
cd ~/align_pipeline/training/segmentation
bash run_training.sh -c /path/to/your/config.yaml
```
