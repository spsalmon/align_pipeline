import os

import numpy as np
import polars as pl
from align_toolbox.foundation import image_handling
from align_toolbox.foundation.file_handling import extract_time_point, write_filemap
from align_toolbox.quantification import compute_fluorescence_in_mask
from joblib import Parallel, delayed, parallel_config

from align_pipeline import utils


def quantify_fluorescence_from_file_path(
    source_image_path,
    source_image_channel,
    mask_path,
    aggregations=["sum"],
    background_aggregation="median",
    time_regex=r"Time(\d+)",
    point_regex=r"Point(\d+)",
):
    """Quantify the fluorescence of an image inside a mask."""

    if not isinstance(aggregations, list):
        aggregations = [aggregations]

    try:
        time, point = extract_time_point(source_image_path, time_regex, point_regex)
    except ValueError:
        print(f"Could not extract time and point from {source_image_path}.")
        return None

    source_image = image_handling.read_tiff_file(
        source_image_path, channels_to_keep=[source_image_channel]
    )
    mask = image_handling.read_tiff_file(mask_path)

    measurements = {
        "Time": time,
        "Point": point,
    }

    try:
        results = compute_fluorescence_in_mask(
            source_image,
            mask,
            aggregations=aggregations,
            background_aggregation=background_aggregation,
        )

    except Exception as e:
        print(f"Error processing {source_image_path} with mask {mask_path}: {e}")
        results = {agg: np.nan for agg in aggregations}

    measurements.update(results)

    return measurements


def main(input_pickle, output_file, block_config, config, n_jobs):
    """Main function."""
    block_config = utils.load_pickles(block_config)[0]
    config = utils.load_pickles(config)[0]

    time_regex = config.get("time_regex", r"Time(\d+)")
    point_regex = config.get("point_regex", r"Point(\d+)")

    input_files = utils.load_pickles(input_pickle)[0]
    source_files = [f["source_image_path"] for f in input_files]
    mask_files = [f["mask_path"] for f in input_files]
    aggregations = block_config["fluorescence_quantification_aggregations"]
    background_aggregation = block_config["fluorescence_background_aggregation"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with parallel_config(backend="loky", n_jobs=n_jobs):
        fluo = Parallel()(
            delayed(quantify_fluorescence_from_file_path)(
                source_file,
                block_config["fluorescence_quantification_source"][1],
                mask_file,
                aggregations=aggregations,
                background_aggregation=background_aggregation,
                time_regex=time_regex,
                point_regex=point_regex,
            )
            for source_file, mask_file in zip(source_files, mask_files)
        )

    # filter out None results
    fluo = [f for f in fluo if f is not None]
    fluo_dataframe = pl.DataFrame(fluo)

    output_file_basename = os.path.basename(output_file).split(".")[0]
    fluo_source = output_file_basename.split("_")[0]
    mask_column = output_file_basename.split("_on_")[-1]
    for agg in aggregations:
        # rename columns to match the rest of the pipeline
        fluo_dataframe = fluo_dataframe.rename(
            {
                agg: f"{fluo_source}_fluo_{agg}_on_{mask_column}",
            },
        )

    write_filemap(fluo_dataframe, output_file)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.block_config, args.config, args.n_jobs)
