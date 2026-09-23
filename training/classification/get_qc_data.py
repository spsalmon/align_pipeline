import argparse
import os
import shutil

import numpy as np
import polars as pl
import yaml
from align_toolbox.data_analysis.time_series import (
    correct_series_with_classification,
    smooth_series_classified,
)
from align_toolbox.foundation.file_handling import read_filemap
from align_toolbox.foundation.image_handling import read_tiff_file
from tifffile import imwrite
from tqdm import tqdm

np.random.seed(387799)

# Schema of the gathered classification database. Every code path that returns a
# database must use it, including the "skip this experiment" paths: the caller
# vstacks the result, and an empty DataFrame without this schema raises a
# ShapeError instead of contributing zero rows.
DATABASE_SCHEMA = {
    "Class": pl.Utf8,
    "Image": pl.Utf8,
    "Microscope": pl.Utf8,
    "Point": pl.Int64,
    "Stage": pl.Utf8,
    "Mask": pl.Utf8,
}


def process_strains(strains):
    # to be correct the strain number needs to be followed by a separator like -, _ or space
    correct_strains = []
    if strains is None:
        return correct_strains
    for strain in strains:
        correct_strains.append(strain + "-")
        correct_strains.append(strain + "_")
        correct_strains.append(strain + " ")
    return correct_strains


def get_analysis_filemap(experiment_path, get_annotated_only=False):
    """Return the newest analysis filemap for an experiment, or None.

    Scans every ``analysis*/report`` directory and returns the most recently
    modified filemap. When ``get_annotated_only`` is set, only GUI-annotated
    filemaps (``analysis_filemap_annotated``) are considered.
    """
    keyword = "analysis_filemap_annotated" if get_annotated_only else "analysis_filemap"

    analysis_dirs = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d)) and "analysis" in d
    ]
    report_dirs = [
        report
        for d in analysis_dirs
        if os.path.isdir(report := os.path.join(d, "report"))
    ]

    candidates = []
    for report_dir in report_dirs:
        candidates.extend(
            os.path.join(report_dir, f) for f in os.listdir(report_dir) if keyword in f
        )
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return None


def find_all_relevant_filemaps(
    experiment_directories,
    experiments_to_always_include,
    keywords_to_include,
    experiments_to_exclude,
    valid_scopes_expanded,
    keywords_to_exclude,
    database_config,
    get_annotated_only=False,
):
    filemaps = []
    strains = process_strains(database_config.get("strains", []))
    magnifications = database_config.get("magnifications", [])
    processed_magnifications = []
    for mag in magnifications:
        processed_magnifications.append(mag)
        processed_magnifications.append(mag.lower())
        processed_magnifications.append(mag.upper())

    magnifications = list(set(processed_magnifications))

    for exp in experiment_directories:
        experiment_name = os.path.basename(os.path.normpath(exp))

        # check if the experiment is in the list of experiments to always include
        if experiment_name in experiments_to_always_include:
            filemap = get_analysis_filemap(exp, get_annotated_only=get_annotated_only)
            if filemap:
                filemaps.append(filemap)
                continue

        if any(keyword in experiment_name for keyword in keywords_to_include):
            filemap = get_analysis_filemap(exp, get_annotated_only=get_annotated_only)
            if filemap:
                filemaps.append(filemap)
                continue

        # check if the experiment is in the list of experiments to exclude
        if experiment_name in experiments_to_exclude:
            continue

        if not any(mag in experiment_name for mag in magnifications) and magnifications:
            continue

        if not any(scope in experiment_name for scope in valid_scopes_expanded):
            continue

        if any(keyword in experiment_name for keyword in keywords_to_exclude):
            continue

        if not any(strain in experiment_name for strain in strains) and strains:
            continue

        filemap = get_analysis_filemap(exp, get_annotated_only=get_annotated_only)

        if filemap:
            filemaps.append(filemap)

    return filemaps


def get_images_from_filemap(
    filemap_path,
    database_config,
    valid_scopes_expanded,
    variation_to_unified_scope_name,
    channel,
    threshold=0.95,
    lmbda=0.0075,
    extra_adulthood_time=40,
    get_classes=True,
):
    experiment_name = filemap_path.split("/")[-4]
    filemap_df = read_filemap(filemap_path)

    # if Ignore is in the columns, remove all rows where Ignore is True
    if "Ignore" in filemap_df.columns:
        filemap_df = filemap_df.filter(~pl.col("Ignore"))

    # first check if the filemap has the required columns
    worm_type_columns = [column for column in filemap_df.columns if "qc" in column]

    channel_names = []
    for channel_number in channel:
        channel_names.append(f"ch{channel_number + 1}")
    channel_name = "_".join(channel_names)

    if (
        "ExperimentTime" not in filemap_df.columns
        or filemap_df["ExperimentTime"].is_null().all()
    ):
        # assume that ExperimentTime is in seconds and is equal to Time / 6 (10 min intervals)
        # cast time to float
        filemap_df = filemap_df.with_columns(pl.col("Time").cast(pl.Float64))
        filemap_df = filemap_df.with_columns(
            (pl.col("Time") / 6).alias("ExperimentTime").cast(pl.Float64)
        )
    else:
        # cast ExperimentTime to float
        filemap_df = filemap_df.with_columns(pl.col("ExperimentTime").cast(pl.Float64))
        filemap_df = filemap_df.with_columns(
            (pl.col("ExperimentTime") / 3600).alias("ExperimentTime")
        )

    database = pl.DataFrame(schema=DATABASE_SCHEMA)

    seg_columns = [
        column
        for column in filemap_df.columns
        if ((f"{channel_name}_seg" in column) and ("str" in column))
    ]
    straigthened_raw = f"{channel_name}_raw_str"
    straigthened_raw_columns = [
        column for column in filemap_df.columns if (straigthened_raw in column)
    ]

    # these have to come before the rename and the indexing below
    if len(straigthened_raw_columns) == 0:
        print(
            f"Skipping experiment {experiment_name} as it does not have straightened raw images for channel {channel_name}."
        )
        return database

    if len(seg_columns) == 0:
        print(
            f"Skipping experiment {experiment_name} as it does not have straightened segmentation masks for channel {channel_name}."
        )
        return database

    # remove the previous raw column if it exists
    if "raw" in filemap_df.columns:
        filemap_df = filemap_df.drop("raw")
    filemap_df = filemap_df.rename({straigthened_raw_columns[0]: "raw"})
    volume_columns = [
        column
        for column in filemap_df.columns
        if ((channel_name in column) and ("volume" in column))
    ]

    rows = []

    filemaps_of_points = filemap_df.partition_by("Point")

    microscope = [scope for scope in valid_scopes_expanded if scope in experiment_name][
        0
    ]
    microscope = variation_to_unified_scope_name.get(microscope, microscope)

    print(f"Microscope found: {microscope} in experiment {experiment_name}")

    print(f"Get classes: {get_classes} for experiment {experiment_name}")
    seg_column = seg_columns[0]
    if get_classes:
        if len(worm_type_columns) == 0:
            # make a fake qc column with all good if no qc column exists, so we can still use the data for training but just label everything as good
            filemap_df = filemap_df.with_columns(pl.lit("worm").alias("placeholder_qc"))
            worm_type_columns = ["placeholder_qc"]

        if len(volume_columns) == 0:
            print(
                f"Skipping experiment {experiment_name} as it does not have volume columns."
            )
            return database

        worm_type_column = worm_type_columns[0]

        for filemap in filemaps_of_points:
            if "M4" in filemap.columns:
                # if M4 is annotated, remove rows where Time > M4 + extra_adulthood_time
                m4 = filemap.select(pl.col("M4")).row(0)[0]
                if m4 is not None and not np.isnan(m4):
                    filemap = filemap.filter(
                        pl.col("Time") <= m4 + extra_adulthood_time
                    )

            # remove rows with time too close to the molts
            # we need to do this because the series is not smooth around the molts
            molts = ["M1", "M2", "M3", "M4"]
            for molt in molts:
                if molt in filemap.columns:
                    molt_time = filemap.select(pl.col(molt)).row(0)[0]
                    if molt_time is not None and not np.isnan(molt_time):
                        filemap = filemap.filter(
                            (pl.col("Time") < molt_time - 4)
                            | (pl.col("Time") > molt_time + 4)
                        )

            point = filemap.select(pl.col("Point")).row(0)[0]
            volume = filemap.select(pl.col(volume_columns[0])).to_numpy().squeeze()
            # skip point if less than 10 data points
            if np.sum(~np.isnan(volume)) < 10:
                continue
            worm_types = filemap.select(pl.col(worm_type_column)).to_numpy().squeeze()
            corrected_volume = correct_series_with_classification(volume, worm_types)

            experiment_time = (
                filemap.select(pl.col("ExperimentTime")).to_numpy().squeeze()
            )

            smoothed_volume = smooth_series_classified(
                volume,
                experiment_time,
                qc=worm_types,
                lmbda=lmbda,
            )

            normalized_residuals = (
                corrected_volume - smoothed_volume
            ) / smoothed_volume

            confirmed_errors = np.where(worm_types == "error")[0]
            threshold_value = np.quantile(np.abs(normalized_residuals), threshold)
            potential_errors = np.where(np.abs(normalized_residuals) > threshold_value)[
                0
            ]
            potential_good = np.where(np.abs(normalized_residuals) <= threshold_value)[
                0
            ]
            # remove confirmed errors from potential errors
            potential_errors = np.setdiff1d(potential_errors, confirmed_errors)
            # remove confirmed errors from potential good
            potential_good = np.setdiff1d(potential_good, confirmed_errors)

            # drop 95% of potential good to reduce dataset size
            if len(potential_good) > 0:
                potential_good = np.random.choice(
                    potential_good, size=int(len(potential_good) * 0.05), replace=False
                )
            eggs = np.where(worm_types == "egg")[0]

            # Vectorized creation of all rows
            indices = np.concatenate(
                [potential_errors, potential_good, eggs, confirmed_errors]
            )
            classes = np.concatenate(
                [
                    np.full(len(potential_errors), "potential_error"),
                    np.full(len(potential_good), "good"),
                    np.full(len(eggs), "egg"),
                    np.full(len(confirmed_errors), "error"),
                ]
            )

            images = filemap.select(pl.col("raw")).to_numpy().squeeze()[indices]
            # replace external.data/TowbinLab by towbin.data/shared
            images = [img.replace("TowbinLab", "shared") for img in images]
            images = [img.replace("external.data", "towbin.data") for img in images]

            masks = filemap.select(pl.col(seg_column)).to_numpy().squeeze()[indices]
            masks = [m.replace("TowbinLab", "shared") for m in masks]
            masks = [m.replace("external.data", "towbin.data") for m in masks]

            # only keep if both image and mask exist
            valid_indices = [
                i
                for i, (img, msk) in enumerate(zip(images, masks))
                if os.path.exists(img) and os.path.exists(msk)
            ]
            classes = [classes[i] for i in valid_indices]
            images = [images[i] for i in valid_indices]
            masks = [masks[i] for i in valid_indices]
            filemap_rows = [
                {
                    "Class": clss,
                    "Image": img,
                    "Microscope": microscope,
                    "Point": point,
                    "Stage": "unknown",
                    "Mask": mask,
                }
                for clss, img, mask in zip(classes, images, masks)
                # for clss, img in zip(classes, images)
            ]
            rows.extend(filemap_rows)
        database = pl.DataFrame(
            rows,
            schema=DATABASE_SCHEMA,
        )
    else:
        for filemap in filemaps_of_points:
            point = filemap.select(pl.col("Point")).row(0)[0]

            # if we have volume information, we can already remove all images where it's nan or 0
            if len(volume_columns) > 0:
                filemap = filemap.filter(pl.col(volume_columns[0]).is_not_nan())
                filemap = filemap.filter(pl.col(volume_columns[0]) != 0)

            images = filemap.select(pl.col("raw")).to_numpy().squeeze()
            images = [img.replace("TowbinLab", "shared") for img in images]
            images = [img.replace("external.data", "towbin.data") for img in images]

            masks = filemap.select(pl.col(seg_column)).to_numpy().squeeze()
            masks = [m.replace("TowbinLab", "shared") for m in masks]
            masks = [m.replace("external.data", "towbin.data") for m in masks]

            valid_indices = [
                i
                for i, (img, msk) in enumerate(zip(images, masks))
                if os.path.exists(img) and os.path.exists(msk)
            ]
            images = [images[i] for i in valid_indices]
            masks = [masks[i] for i in valid_indices]

            filemap_rows = [
                {
                    "Class": "unknown",
                    "Image": img,
                    "Microscope": microscope,
                    "Point": point,
                    "Stage": "unknown",
                    "Mask": mask,
                }
                for img, mask in zip(images, masks)
            ]
            rows.extend(filemap_rows)
        database = pl.DataFrame(
            rows,
            schema=DATABASE_SCHEMA,
        )

    return database


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config_path = args.config

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    storage_path = config.get("storage_path", "")
    valid_subdirectories = config.get("valid_subdirectories", [])

    database_path = config.get("database_path", None)
    database_configs = config.get("database_configs", {})
    extra_adulthood_time = config.get("extra_adulthood_time", 0)
    valid_scopes = config.get("valid_scopes", [])
    scopes_alt_names = config.get("scopes_alt_names", {})
    keywords_to_exclude = config.get("keywords_to_exclude", [])
    keywords_to_include = config.get("keywords_to_include", [])
    experiments_to_consider = config.get("experiments_to_consider", [])
    experiments_to_always_include = config.get("experiments_to_always_include", [])
    experiments_to_exclude = config.get("experiments_to_exclude", [])
    class_proportions = config.get(
        "class_proportions",
        None,
    )

    get_classes = class_proportions is not None

    print(f"Extracting classes: {get_classes}")

    lmbda = config.get("lambda", 0.0075)
    threshold = config.get("threshold", 0.95)
    valid_scopes_expanded = []
    for scope in valid_scopes:
        if scope in scopes_alt_names:
            valid_scopes_expanded.extend(scopes_alt_names[scope])
        else:
            valid_scopes_expanded.append(scope)

    os.makedirs(database_path, exist_ok=True)

    # Create mapping from variations to standard names
    variation_to_unified_scope_name = {}
    for microscope, variations in scopes_alt_names.items():
        for variation in variations:
            variation_to_unified_scope_name[variation] = microscope

    os.makedirs(database_path, exist_ok=True)
    for sub_db in database_configs.keys():
        sub_db_dir = os.path.join(database_path, sub_db)
        os.makedirs(sub_db_dir, exist_ok=True)
        os.makedirs(os.path.join(sub_db_dir, "images"), exist_ok=True)
        # os.makedirs(os.path.join(sub_db_dir, "masks"), exist_ok=True)

    # get all experiment directories
    experiment_directories = []
    valid_subdirectories = [
        os.path.join(storage_path, sub_dir) for sub_dir in valid_subdirectories
    ]
    if experiments_to_consider:
        experiment_directories = experiments_to_consider
    else:
        for exp_dir in valid_subdirectories:
            experiment_directories.extend(
                [
                    os.path.join(exp_dir, d)
                    for d in os.listdir(exp_dir)
                    if os.path.isdir(os.path.join(exp_dir, d))
                ]
            )

    experiment_directories.extend(experiments_to_always_include)
    experiment_directories = list(set(experiment_directories))

    # filter experiment directories based on the criteria and get their filemaps
    for database_name, database_config in database_configs.items():
        channel = database_config.get("channel", [0])
        get_annotated_only = database_config.get("stage_proportions", None) is not None
        filemaps = find_all_relevant_filemaps(
            experiment_directories,
            experiments_to_always_include,
            keywords_to_include,
            experiments_to_exclude,
            valid_scopes_expanded,
            keywords_to_exclude,
            database_config,
            get_annotated_only=get_annotated_only,
        )

        database = pl.DataFrame(
            schema=DATABASE_SCHEMA,
        )
        for filemap in filemaps:
            # try:
            database = database.vstack(
                get_images_from_filemap(
                    filemap,
                    database_config,
                    valid_scopes_expanded,
                    variation_to_unified_scope_name=variation_to_unified_scope_name,
                    channel=channel,
                    lmbda=lmbda,
                    threshold=threshold,
                    extra_adulthood_time=extra_adulthood_time,
                    get_classes=get_classes,
                )
            )
            # except Exception as e:
            #     print(f"Error processing filemap {filemap}: {e}")
            #     continue

        total_images = database_config.get("size", 1000)
        if get_classes:
            # calculate number of images per class
            class_counts = {}

            for cls, prop in class_proportions.items():
                class_counts[cls] = int(total_images * prop)
            # sample images per class
            sampled_database = pl.DataFrame(
                schema=DATABASE_SCHEMA,
            )
            for cls, count in class_counts.items():
                class_images = database.filter(pl.col("Class") == cls)
                n = min(count, class_images.height)
                class_images = class_images.sample(n, with_replacement=False)
                sampled_database = sampled_database.vstack(class_images)

            # sort the dataset by class to group similar images together
            sampled_database = sampled_database.sort("Class")
        else:
            sampled_database = database.sample(total_images, with_replacement=False)

        # create output names
        def get_output_name(i, row):
            return f'image_{i}_{row["Class"]}_{row["Microscope"]}.tiff'

        sampled_database = sampled_database.with_columns(
            pl.arange(0, sampled_database.height).alias("Index")
        )
        sampled_database = sampled_database.with_columns(
            pl.struct(["Index", "Class", "Microscope"])
            .map_elements(
                lambda x: get_output_name(x["Index"], x), return_dtype=pl.String
            )
            .alias("OutputName")
        )
        sampled_database = sampled_database.select(pl.exclude("Index", "Point"))
        sampled_database.write_csv(
            os.path.join(
                database_path,
                database_name,
                f"{database_name}_classification_filemap.csv",
            )
        )

        # copy the config file to the database directory
        shutil.copy(
            config_path,
            os.path.join(database_path, database_name, f"{database_name}_config.yaml"),
        )

        images_output_dir = os.path.join(database_path, database_name, "images")
        masks_output_dir = os.path.join(database_path, database_name, "masks")
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(masks_output_dir, exist_ok=True)
        for row in tqdm(
            sampled_database.iter_rows(named=True), total=sampled_database.height
        ):
            image_path = row["Image"]
            mask_path = row["Mask"]
            image = read_tiff_file(image_path, channels_to_keep=channel)
            mask = read_tiff_file(mask_path)
            image_name = row["OutputName"]
            # save the image
            imwrite(
                os.path.join(images_output_dir, image_name), image, compression="zlib"
            )
            imwrite(
                os.path.join(masks_output_dir, image_name), mask, compression="zlib"
            )
