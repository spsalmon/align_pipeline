import logging
import os

import polars as pl
from align_toolbox.foundation import image_handling, worm_features
from align_toolbox.foundation.file_handling import extract_time_point, write_filemap
from joblib import Parallel, delayed, parallel_config

from align_pipeline import utils


def compute_morphological_features_from_file_path(
    straightened_mask_path,
    pixelsize,
    features,
    time_regex=r"Time(\d+)",
    point_regex=r"Point(\d+)",
):
    """Compute the volume of a straightened mask."""
    try:
        time, point = extract_time_point(
            straightened_mask_path, time_regex, point_regex
        )
    except Exception as e:
        print(f"Time regex: {time_regex}, Point regex: {point_regex}")
        logging.warning(
            f"Could not extract time and point from {straightened_mask_path}: {e}"
        )
        return None

    str_mask = image_handling.read_tiff_file(straightened_mask_path)

    features = worm_features.compute_mask_morphological_features(
        str_mask, pixelsize, features
    )

    features["Time"] = time
    features["Point"] = point

    return features


def main(input_pickle, output_file, block_config, config, n_jobs):
    """Main function."""
    block_config = utils.load_pickles(block_config)[0]
    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]
    logging.info(f"Computing volume for {len(input_files)} files.")

    time_regex = config.get("time_regex", r"Time(\d+)")
    point_regex = config.get("point_regex", r"Point(\d+)")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with parallel_config(backend="loky", n_jobs=n_jobs):
        morphological_features = Parallel()(
            delayed(compute_morphological_features_from_file_path)(
                input_file,
                block_config["pixelsize"],
                block_config["morphological_features"],
                time_regex,
                point_regex,
            )
            for input_file in input_files
        )

    # filter out None values (files where time and point could not be extracted)
    morphological_features = [f for f in morphological_features if f is not None]

    morphological_features_dataframe = pl.DataFrame(morphological_features)

    # rename columns to match the rest of the pipeline
    output_file_basename = os.path.basename(output_file).split("_morphology")[0]
    for feature in block_config["morphological_features"]:
        morphological_features_dataframe = morphological_features_dataframe.rename(
            {
                feature: f"{output_file_basename}_{feature}",
            },
        )
    write_filemap(morphological_features_dataframe, output_file)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.block_config, args.config, args.n_jobs)
