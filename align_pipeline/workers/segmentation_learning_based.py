import logging
import os
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
from align_toolbox.deep_learning.deep_learning_tools import (
    load_segmentation_model_from_checkpoint,
)
from align_toolbox.deep_learning.utils.augmentation import (
    get_prediction_augmentation_from_model,
)
from align_toolbox.deep_learning.utils.dataset import (
    SegmentationPredictionDataset,
    StackPredictionDataset,
)
from align_toolbox.foundation import image_handling
from cv2 import resize
from joblib import Parallel, delayed, parallel_config
from tifffile import imwrite
from torch.utils.data import DataLoader

from align_pipeline import utils

logging.basicConfig(level=logging.INFO)


def prefetch_images_cellpose(raw_paths, channel, prefetch=8):
    q = Queue(maxsize=prefetch)

    def producer():
        for raw_path in raw_paths:
            raw = image_handling.read_tiff_file(raw_path, channels_to_keep=[channel])
            raw = np.expand_dims(raw, axis=-1)
            q.put((raw_path, raw))
        q.put(None)

    Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def reshape_images_to_original_shape(images, original_shapes, padded_or_cropped="pad"):
    reshaped_images = []
    for image, original_shape in zip(images, original_shapes):
        if padded_or_cropped == "pad":
            reshaped_image = image_handling.crop_to_dim_equally(
                image, original_shape[-2], original_shape[-1]
            )
        elif padded_or_cropped == "crop":
            reshaped_image = image_handling.pad_to_dim_equally(
                image, original_shape[-2], original_shape[-1]
            )
        reshaped_images.append(reshaped_image)
    return reshaped_images


def predict_batch(model, images, image_shapes, device, n_classes, scale_factor=1.0):
    if not isinstance(images, torch.Tensor):
        images = torch.from_numpy(images)
    images = images.to(device)

    with torch.no_grad():
        predictions = model(images)

    predictions = predictions.cpu().numpy()
    predictions = np.squeeze(predictions)

    if predictions.ndim < 3:
        predictions = np.expand_dims(predictions, axis=0)

    if n_classes > 1 and predictions.ndim < 4:
        predictions = np.expand_dims(predictions, axis=0)

    if n_classes > 1:
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions > 0.5
    predictions = predictions.astype(np.uint8)

    if predictions.ndim == 2:
        predictions = np.expand_dims(predictions, axis=0)

    predictions = reshape_images_to_original_shape(
        predictions, image_shapes, padded_or_cropped="pad"
    )

    if scale_factor != 1.0:
        predictions = [
            resize(
                prediction,
                (
                    int(prediction.shape[0] * 1 / scale_factor),
                    int(prediction.shape[1] * 1 / scale_factor),
                ),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
            for prediction in predictions
        ]
    return predictions


def save_prediction(prediction, output_path, z_dim=None, t_dim=None):
    if z_dim is None or t_dim is None:
        imwrite(output_path, prediction, compression="zlib")
    else:
        if t_dim > 1 and z_dim > 1:
            axes = "TZYX"
        elif t_dim > 1:
            axes = "TYX"
        elif z_dim > 1:
            axes = "ZYX"
        else:
            axes = "ZYX"  # Default to Z-stack

        metadata = {"axes": axes}
        imwrite(
            output_path, prediction, compression="zlib", ome=True, metadata=metadata
        )


# ---- Segmentation methods ----


def deep_learning_segmentation(
    block_config,
    input_files,
    output_files,
    segmentation_channels,
    n_jobs,
    z_dim,
    t_dim,
    is_stack=False,
):
    if block_config["model_path"] is None:
        raise ValueError(
            "model_path must be set in the block_config file for deep learning segmentation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_segmentation_model_from_checkpoint(block_config["model_path"]).to(
        device
    )
    n_classes = model.n_classes

    enforce_n_channels = block_config.get("enforce_n_channels", None)
    scale_factor = block_config.get("scale_factor", 1.0)
    preprocessing_fn = get_prediction_augmentation_from_model(
        model, enforce_n_channels=enforce_n_channels
    )

    batch_size = block_config["batch_size"]
    model.eval()

    if not is_stack:
        dataset = SegmentationPredictionDataset(
            input_files,
            segmentation_channels,
            preprocessing_fn,
            scale_factor=scale_factor,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_jobs // 2,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                image_paths, images, image_shapes, invalid_indices = batch
                batch_size = len(image_paths) + len(invalid_indices)

                if len(image_paths) == 0:
                    # if all images in the batch failed to load, create a batch of black masks
                    predictions = np.zeros((batch_size, (10, 10)), dtype=np.uint8)
                else:
                    if images.ndim == 3:
                        images = images.unsqueeze(0)

                    predictions = predict_batch(
                        model,
                        images,
                        image_shapes,
                        device,
                        n_classes,
                        scale_factor=scale_factor,
                    )

                    # insert black masks for any images that failed to load
                    if len(invalid_indices) > 0:
                        predictions = np.array(predictions)
                        for j in invalid_indices:
                            predictions = np.insert(
                                predictions,
                                j,
                                np.zeros_like(predictions[0]),
                                axis=0,
                            )
                        predictions = list(predictions)

                with parallel_config(backend="threading", n_jobs=n_jobs // 2):
                    Parallel()(
                        delayed(save_prediction)(prediction, output_path)
                        for prediction, output_path in zip(predictions, output_files)
                    )

                # remove the output paths that have been processed
                output_files = output_files[batch_size:]
    else:
        # it's suboptimal, but for now we treat each image individually
        for input_file, output_file in zip(input_files, output_files):
            dataset = StackPredictionDataset(
                input_file[0],
                channels=segmentation_channels,
                transform=preprocessing_fn,
                enforce_divisibility_by=32,
                pad_or_crop="pad",
                scale_factor=scale_factor,
            )

            stack_shape = dataset.stack_shape
            original_shapes = [stack_shape] * batch_size

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_jobs // 2,
                pin_memory=True,
            )

            segmented_planes = []
            for batch in dataloader:
                predictions = predict_batch(
                    model,
                    batch,
                    original_shapes,
                    device,
                    n_classes,
                    scale_factor=1.0,
                )
                segmented_planes.extend(predictions)

            # in the 3D case it's more elegant to rescale the whole stack at once
            resized_planes = []
            if scale_factor != 1.0:
                for plane in segmented_planes:
                    resized_plane = resize(
                        plane,
                        (
                            int(plane.shape[0] * 1 / scale_factor),
                            int(plane.shape[1] * 1 / scale_factor),
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    resized_planes.append(resized_plane)
                segmented_planes = np.stack(resized_planes, axis=0)
            else:
                segmented_planes = np.stack(segmented_planes, axis=0)

            save_prediction(
                segmented_planes,
                output_file,
                z_dim=z_dim,
                t_dim=t_dim,
            )


def cellpose_segmentation(
    block_config,
    input_files,
    output_files,
    segmentation_channels,
    z_dim,
    t_dim,
    stitch_3D=True,
):
    # Imported here so cellpose is only required when this method is used.
    from cellpose import models

    gpu = torch.cuda.is_available()
    batch_size = block_config.get("batch_size", 8)
    model = models.CellposeModel(gpu=gpu, pretrained_model=block_config["model_path"])
    for (_, image), output_path in zip(
        prefetch_images_cellpose(input_files, channel=segmentation_channels),
        output_files,
    ):
        masks, _, _ = model.eval(
            image,
            z_axis=(0 if z_dim > 1 else None),
            channel_axis=-1,
            do_3D=False,
            batch_size=batch_size,
            stitch_threshold=(0.25 if stitch_3D else 0.0),
        )
        save_prediction(masks.astype(np.uint16), output_path, z_dim=z_dim, t_dim=t_dim)


def main(input_pickle, output_pickle, block_config, n_jobs):
    """Main function."""
    block_config = utils.load_pickles(block_config)[0]
    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    segmentation_channels = block_config.get("segmentation_channels", None)

    is_stack, (z_dim, t_dim) = image_handling.check_if_stack(
        input_files[0][0], channels_to_keep=segmentation_channels
    )

    print(
        f'Predicting on {"stacks" if is_stack else "individual images"} with segmentation channels: {segmentation_channels}'
    )

    assert not (
        t_dim > 1 and z_dim > 1
    ), "4D images with both time and z dimensions are not supported yet."

    if block_config["segmentation_method"] == "deep_learning":
        deep_learning_segmentation(
            block_config,
            input_files,
            output_files,
            segmentation_channels,
            n_jobs,
            z_dim,
            t_dim,
            is_stack,
        )
    if block_config["segmentation_method"] == "cellpose":
        cellpose_segmentation(
            block_config,
            input_files,
            output_files,
            segmentation_channels,
            n_jobs,
            z_dim,
            t_dim,
            is_stack,
        )


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.block_config, args.n_jobs)
