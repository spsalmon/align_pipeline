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


def find_column(columns, name):
    """Return the filemap column called ``name``, or None.

    Path columns carry the analysis directory as a prefix (e.g.
    ``analysis/ch2_seg_str``), so a ``/name`` suffix also matches. Matching is
    exact otherwise: a substring match would let ``ch2_raw_str`` pick up
    ``ch1_ch2_raw_str``.
    """
    for column in columns:
        if column == name or column.endswith(f"/{name}"):
            return column
    return None


def fix_storage_path(path):
    """Point old storage mounts at the current one. Missing paths stay None."""
    if path is None:
        return None
    return path.replace("TowbinLab", "shared").replace("external.data", "towbin.data")


def path_exists(path):
    # filemaps have null paths where a block produced no output for a frame
    return path is not None and os.path.exists(path)


def resolve_channel_names(database_config, n_filemaps, organ_channels):
    """Return the channel name (e.g. ``ch2``) to use for each filemap.

    A database either sets ``channel`` (0-indexed channel list, the same for
    every experiment) or ``organ``, looked up per experiment in
    ``organ_channels``. The latter handles organs that sit in a different
    channel depending on the scope. ``organ_channels`` follows the plotting
    convention: a single mapping applied to all filemaps, or one per filemap.
    """
    organ = database_config.get("organ", None)
    if organ is None:
        channel = database_config.get("channel", [0])
        channel_name = "_".join(f"ch{channel_number + 1}" for channel_number in channel)
        return [channel_name] * n_filemaps

    if organ_channels is None:
        raise ValueError(
            f"Database config uses organ '{organ}' but no organ_channels were given."
        )
    if isinstance(organ_channels, dict):
        organ_channels = [organ_channels]
    if len(organ_channels) == 1:
        organ_channels = organ_channels * n_filemaps
    elif len(organ_channels) != n_filemaps:
        raise ValueError(
            f"organ_channels has {len(organ_channels)} entries but there are "
            f"{n_filemaps} filemaps. Provide one mapping, or one per filemap."
        )

    channel_names = []
    for i, mapping in enumerate(organ_channels):
        if organ not in mapping:
            raise ValueError(
                f"Organ '{organ}' missing from organ_channels entry {i}: {mapping}"
            )
        channel_names.append(mapping[organ])
    return channel_names


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
    channel_name,
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

    # first check if the filemap has the required columns. Filemaps usually hold
    # several organs, so the QC column has to be the one of this channel's masks.
    seg_name = f"{channel_name}_seg_str"
    qc_column = find_column(filemap_df.columns, f"{seg_name}_qc")
    worm_type_columns = [qc_column] if qc_column is not None else []

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

    seg_column = find_column(filemap_df.columns, seg_name)
    seg_columns = [seg_column] if seg_column is not None else []
    straigthened_raw_column = find_column(filemap_df.columns, f"{channel_name}_raw_str")
    straigthened_raw_columns = (
        [straigthened_raw_column] if straigthened_raw_column is not None else []
    )

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
    volume_column = find_column(filemap_df.columns, f"{seg_name}_volume")
    volume_columns = [volume_column] if volume_column is not None else []

    rows = []

    filemaps_of_points = filemap_df.partition_by("Point")

    microscopes = [scope for scope in valid_scopes_expanded if scope in experiment_name]
    # explicitly listed filemaps are not filtered by scope, so there may be no match
    microscope = microscopes[0] if microscopes else "unknown"
    microscope = variation_to_unified_scope_name.get(microscope, microscope)

    print(
        f"Microscope found: {microscope} in experiment {experiment_name}, "
        f"using channel {channel_name}"
    )

    print(f"Get classes: {get_classes} for experiment {experiment_name}")
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
            images = [fix_storage_path(img) for img in images]

            masks = filemap.select(pl.col(seg_column)).to_numpy().squeeze()[indices]
            masks = [fix_storage_path(m) for m in masks]

            # only keep if both image and mask exist
            valid_indices = [
                i
                for i, (img, msk) in enumerate(zip(images, masks))
                if path_exists(img) and path_exists(msk)
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
            images = [fix_storage_path(img) for img in images]

            masks = filemap.select(pl.col(seg_column)).to_numpy().squeeze()
            masks = [fix_storage_path(m) for m in masks]

            valid_indices = [
                i
                for i, (img, msk) in enumerate(zip(images, masks))
                if path_exists(img) and path_exists(msk)
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


def sample_database(database, total_images, class_proportions, scope_proportions):
    """Sample ``total_images`` rows of the gathered database.

    ``class_proportions`` and ``scope_proportions`` (keyed by the unified
    microscope name) are optional. When both are set, each (scope, class) pair
    gets ``total_images * scope_prop * class_prop`` images. Scopes missing from
    ``scope_proportions`` are dropped. A group with fewer images than requested
    contributes all of its images.
    """
    if scope_proportions:
        groups = [
            (pl.col("Microscope") == scope, prop, scope)
            for scope, prop in scope_proportions.items()
        ]
    else:
        groups = [(pl.lit(True), 1.0, None)]

    if class_proportions:
        groups = [
            (
                scope_filter & (pl.col("Class") == cls),
                scope_prop * class_prop,
                (scope, cls),
            )
            for scope_filter, scope_prop, scope in groups
            for cls, class_prop in class_proportions.items()
        ]

    sampled_database = pl.DataFrame(schema=DATABASE_SCHEMA)
    for group_filter, prop, group in groups:
        count = int(total_images * prop)
        group_images = database.filter(group_filter)
        if group is not None and group_images.height < count:
            print(
                f"Only {group_images.height} images available for {group}, "
                f"{count} requested."
            )
        n = min(count, group_images.height)
        sampled_database = sampled_database.vstack(
            group_images.sample(n, with_replacement=False)
        )
    return sampled_database


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
    # explicit list of filemaps, used instead of scanning storage_path
    filemap_paths = config.get("filemap_paths", [])
    organ_channels = config.get("organ_channels", None)
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
    if filemap_paths:
        pass
    elif experiments_to_consider:
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
        if filemap_paths:
            filemaps = filemap_paths
        else:
            get_annotated_only = (
                database_config.get("stage_proportions", None) is not None
            )
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
        channel_names = resolve_channel_names(
            database_config, len(filemaps), organ_channels
        )

        database = pl.DataFrame(
            schema=DATABASE_SCHEMA,
        )
        for filemap, channel_name in zip(filemaps, channel_names):
            # try:
            database = database.vstack(
                get_images_from_filemap(
                    filemap,
                    database_config,
                    valid_scopes_expanded,
                    variation_to_unified_scope_name=variation_to_unified_scope_name,
                    channel_name=channel_name,
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
        sampled_database = sample_database(
            database,
            total_images,
            class_proportions,
            database_config.get("scope_proportions", None),
        )
        if get_classes:
            # sort the dataset by class to group similar images together
            sampled_database = sampled_database.sort("Class")

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
            # the straightened raw image only holds the selected channel(s)
            image = read_tiff_file(image_path)
            mask = read_tiff_file(mask_path)
            image_name = row["OutputName"]
            # save the image
            imwrite(
                os.path.join(images_output_dir, image_name), image, compression="zlib"
            )
            imwrite(
                os.path.join(masks_output_dir, image_name), mask, compression="zlib"
            )
