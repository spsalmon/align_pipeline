import argparse
import os
import pickle
import shutil

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
import yaml
from align_toolbox.deep_learning.architectures import KeypointDetection1DModel
from align_toolbox.deep_learning.utils.dataset import KeypointDetection1DTrainingDataset
from align_toolbox.deep_learning.utils.util import create_lightweight_checkpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


def resolve_dataset_paths(config):
    """Resolve the three dataset pickle paths from the config.

    Either the explicit ``features_pickle`` / ``heatmaps_pickle`` /
    ``keypoints_pickle`` paths are used, or they are inferred from ``dataset_dir``
    using the naming convention written by ``get_molt_detection_data.py``.
    """
    features_pickle = config.get("features_pickle", None)
    heatmaps_pickle = config.get("heatmaps_pickle", None)
    keypoints_pickle = config.get("keypoints_pickle", None)

    if features_pickle and heatmaps_pickle and keypoints_pickle:
        return features_pickle, heatmaps_pickle, keypoints_pickle

    dataset_dir = config.get("dataset_dir", None)
    if dataset_dir is None:
        raise ValueError(
            "You must provide either 'dataset_dir' or the three explicit pickle "
            "paths ('features_pickle', 'heatmaps_pickle', 'keypoints_pickle')."
        )

    keymap_type = config.get("keymap_type", "gaussian")
    features_pickle = os.path.join(dataset_dir, "X_molt_detection.pickle")
    heatmaps_pickle = os.path.join(
        dataset_dir, f"y_{keymap_type}_molt_detection.pickle"
    )
    keypoints_pickle = os.path.join(dataset_dir, "keypoints_molt_detection.pickle")
    return features_pickle, heatmaps_pickle, keypoints_pickle


def check_dataset_exists(features_pickle, heatmaps_pickle, keypoints_pickle):
    """Make sure the dataset pickles exist before training.

    If any is missing, point the user at the dataset gathering step instead of
    letting training fail later with an opaque FileNotFoundError.
    """
    missing = [
        path
        for path in (features_pickle, heatmaps_pickle, keypoints_pickle)
        if not os.path.isfile(path)
    ]
    if missing:
        missing_str = "\n  ".join(missing)
        raise FileNotFoundError(
            "Could not find the molt detection dataset. Missing file(s):\n  "
            f"{missing_str}\n\n"
            "Gather the dataset first by running:\n"
            "  ~/.local/bin/micromamba run -n align python3 "
            "get_molt_detection_data.py -c configs/molt_dataset_config.yaml\n\n"
            "and make sure this config's 'dataset_dir' (or the explicit pickle "
            "paths) matches the gathering config's 'output_dir'."
        )


if __name__ == "__main__":
    config_file = get_args().config
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    features_pickle, heatmaps_pickle, keypoints_pickle = resolve_dataset_paths(config)
    check_dataset_exists(features_pickle, heatmaps_pickle, keypoints_pickle)

    save_dir = config.get("save_dir", "./model_checkpoints")
    model_name = config.get("model_name", "molt_detection_model")

    input_channels = config.get("input_channels", 2)
    n_classes = config.get("n_classes", 4)
    activation = config.get("activation", "sigmoid")
    learning_rate = config.get("learning_rate", 1e-4)

    enforce_divisibility_by = config.get("enforce_divisibility_by", 32)
    resize_method = config.get("resize_method", "pad")

    batch_size = config.get("batch_size", 64)
    max_epochs = config.get("max_epochs", 50)
    num_workers = config.get("num_workers", 32)
    accumulate_grad_batches = config.get("accumulate_grad_batches", 1)
    train_test_split_ratio = config.get("train_test_split_ratio", 0.2)
    save_best_k_models = config.get("save_best_k_models", 1)
    swa_lrs = config.get("swa_lrs", 1e-2)
    random_state = config.get("random_state", 42)

    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # copy the config file to the model save directory
    shutil.copy(config_file, model_save_dir)

    X = pickle.load(open(features_pickle, "rb"))
    y = pickle.load(open(heatmaps_pickle, "rb"))
    keypoints_all = pickle.load(open(keypoints_pickle, "rb"))

    y_train, y_test = train_test_split(
        y, test_size=train_test_split_ratio, random_state=random_state
    )
    X_raw_train, X_raw_test, keypoints_train, keypoints_test = train_test_split(
        X, keypoints_all, test_size=train_test_split_ratio, random_state=random_state
    )

    model = KeypointDetection1DModel(
        input_channels=input_channels,
        n_classes=n_classes,
        learning_rate=learning_rate,
        activation=activation,
    )

    train_dataset = KeypointDetection1DTrainingDataset(
        inputs=X_raw_train,
        heatmap_targets=y_train,
        index_targets=keypoints_train,
        enforce_divisibility_by=enforce_divisibility_by,
        resize_method=resize_method,
    )

    val_dataset = KeypointDetection1DTrainingDataset(
        inputs=X_raw_test,
        heatmap_targets=y_test,
        index_targets=keypoints_test,
        enforce_divisibility_by=enforce_divisibility_by,
        resize_method=resize_method,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
    )

    device = "gpu" if torch.cuda.is_available() else "cpu"

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=model_save_dir, save_top_k=save_best_k_models, monitor="val_loss"
    )
    swa_callback = callbacks.StochasticWeightAveraging(swa_lrs=swa_lrs)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        strategy="auto",
        callbacks=[checkpoint_callback, swa_callback],
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=0.5,
        detect_anomaly=False,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    print(checkpoint_callback.best_model_path)

    create_lightweight_checkpoint(
        input_path=checkpoint_callback.best_model_path,
        output_path=os.path.join(model_save_dir, "best_light.ckpt"),
    )
