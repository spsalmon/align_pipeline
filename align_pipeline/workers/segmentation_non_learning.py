import logging
import os

import numpy as np
from align_toolbox.foundation.image_handling import check_if_stack, read_tiff_file
from align_toolbox.segmentation.segmentation_tools import (
    get_segmentation_function,
    segment_image,
)
from joblib import Parallel, delayed, parallel_config
from tifffile import imwrite

from align_pipeline import utils

logging.basicConfig(level=logging.INFO)


def segment_and_save(
    image_path,
    output_path,
    method,
    pixelsize,
    channels=None,
    is_stack=False,
    **kwargs,
):
    """Segment image and save to output_path."""
    try:
        image = read_tiff_file(image_path, channels_to_keep=channels).squeeze()
        if not is_stack:
            mask = segment_image(
                image,
                method,
                pixelsize=pixelsize,
                is_stack=False,
                **kwargs,
            )
            metadata = None
        else:
            segmentation_function = get_segmentation_function(
                method,
                pixelsize=pixelsize,
                **{k: v for k, v in kwargs.items() if k != "n_jobs"},
            )
            with parallel_config(backend="loky", n_jobs=kwargs.get("n_jobs", -1)):
                results = Parallel()(
                    delayed(segmentation_function)(plane) for plane in image
                )
            mask = np.array(results)

            t_dim = kwargs.get("t_dim", 1)
            z_dim = kwargs.get("z_dim", 1)

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
            output_path,
            mask.astype(np.uint8),
            compression="zlib",
            ome=True,
            metadata=metadata,
        )
    except Exception as e:
        logging.error(f"Caught exception while segmenting {image_path}: {e}")
        return False


def main(input_pickle, output_pickle, block_config, n_jobs):
    """Main function."""
    block_config = utils.load_pickles(block_config)[0]
    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    segmentation_channels = block_config.get("segmentation_channels", None)
    segmentation_method = block_config.get("segmentation_method", None)

    is_stack, (z_dim, t_dim) = check_if_stack(
        input_files[0][0], channels_to_keep=segmentation_channels
    )

    assert not (
        t_dim > 1 and z_dim > 1
    ), "4D images with both time and z dimensions are not supported yet."
    if not is_stack:
        with parallel_config(backend="loky", n_jobs=n_jobs):
            Parallel()(
                delayed(segment_and_save)(
                    input_file,
                    output_path,
                    method=segmentation_method,
                    channels=segmentation_channels,
                    is_stack=is_stack,
                    pixelsize=block_config["pixelsize"],
                    gaussian_filter_sigma=block_config["gaussian_filter_sigma"],
                )
                for input_file, output_path in zip(input_files, output_files)
            )
    else:
        for input_file, output_path in zip(input_files, output_files):
            segment_and_save(
                input_file,
                output_path,
                method=segmentation_method,
                channels=segmentation_channels,
                is_stack=is_stack,
                pixelsize=block_config["pixelsize"],
                gaussian_filter_sigma=block_config["gaussian_filter_sigma"],
                n_jobs=n_jobs,
                z_dim=z_dim,
                t_dim=t_dim,
            )


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.block_config, args.n_jobs)
