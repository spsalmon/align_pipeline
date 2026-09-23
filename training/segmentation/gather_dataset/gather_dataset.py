import argparse
import os
import shutil

import numpy as np
import polars as pl
import yaml
from align_toolbox.foundation.image_handling import read_tiff_file
from joblib import Parallel, delayed
from tifffile import imwrite
from tqdm import tqdm

np.random.seed(387799)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


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
    directories = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
    ]

    analysis_directories = [d for d in directories if "analysis" in d]
    report_directories = [os.path.join(d, "report") for d in analysis_directories]

    report_directories = [d for d in report_directories if os.path.isdir(d)]

    filemap_files = []
    for report_dir in report_directories:
        files = [os.path.join(report_dir, f) for f in os.listdir(report_dir)]
        if get_annotated_only:
            files = [f for f in files if "analysis_filemap_annotated" in f]
        else:
            files = [f for f in files if "analysis_filemap" in f]

        # mat_files = [f for f in files if f.endswith(".mat")]
        # return the filemap that was created last
        if files:
            filemap_files.sort(key=lambda x: os.path.getctime(x))
            filemap_files.append(os.path.join(report_dir, files[-1]))

    if filemap_files:
        filemap_files.sort(key=lambda x: os.path.getctime(x))
        return os.path.join(report_dir, filemap_files[-1])
    # check for converted experiments
    # elif mat_files:
    #     filemap_files = [f for f in files if "analysis_filemap" in f]
    #     if filemap_files:
    #         filemap_files.sort(key=lambda x: os.path.getctime(x))
    #         filemap = pd.read_csv(filemap_files[-1], low_memory=False)
    #         if "HatchTime" in filemap.columns and "raw" in filemap.columns:
    #             return os.path.join(report_dir, filemap_files[-1])
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


def pick_within_larval_stage(filemap, ls_beg, ls_end, n_picks=1):
    try:
        if np.isnan(ls_beg) or np.isnan(ls_end):
            return []

        filemap_of_stage = filemap.filter(
            (pl.col("Time") >= ls_beg) & (pl.col("Time") <= ls_end)
        )

        if filemap_of_stage.height > 0:
            picks = min(n_picks, filemap_of_stage.height)
            picked_filemap = filemap_of_stage.sample(picks, with_replacement=False)
            picked_images = picked_filemap.select(pl.col("raw")).to_series().to_list()

            return picked_images
        else:
            return []
    except Exception as e:
        print(
            f"Error in picking image within larval stage: {ls_beg}, {ls_end}. Error: {e}"
        )
        return []


def get_images_from_filemap(
    filemap_path,
    database_config,
    valid_scopes_expanded,
    extra_adulthood_time=40,
    n_picks=10,
):
    experiment_name = filemap_path.split("/")[-4]
    filemap_df = pl.read_csv(filemap_path)

    filemaps_of_points = filemap_df.partition_by("Point")
    stage_proportions = database_config.get("stage_proportions", None)
    microscope = [scope for scope in valid_scopes_expanded if scope in experiment_name][
        0
    ]

    print(f"Microscope found: {microscope} in experiment {experiment_name}")

    microscope = variation_to_unified_scope_name.get(microscope, microscope)
    database = pl.DataFrame()
    for filemap in filemaps_of_points:
        # remove rows where raw is null
        filemap = filemap.filter(pl.col("raw").is_not_null())
        if filemap.height == 0:
            continue
        point = filemap.select(pl.col("Point")).row(0)[0]
        if stage_proportions is None:
            # if no stage proportions are defined
            rows = []
            images = (
                filemap.sample(min(n_picks, filemap.height), with_replacement=False)
                .select(pl.col("raw"))
                .to_series()
                .to_list()
            )
            for img in images:
                img = img.replace(
                    "external.data/TowbinLab", "towbin.data/shared"
                )  # fixes the path for old experiments
                row = {
                    "Point": point,
                    "Image": img,
                    "Stage": "unknown",
                    "Microscope": microscope,
                    "Experiment": experiment_name,
                }
                rows.append(row)

            database = database.vstack(pl.DataFrame(rows))
            continue
        if "larva" in stage_proportions.keys() and "adult" in stage_proportions.keys():
            # handle experiments where only m4 is annotated
            m4 = filemap.select(pl.col("M4")).row(0)[0]

            if m4 is None or np.isnan(m4):
                continue

            times = [0, m4, m4 + extra_adulthood_time]
            stages = ["larva", "adult"]

        elif (
            "L1" in stage_proportions.keys()
            and "L2" in stage_proportions.keys()
            and "L3" in stage_proportions.keys()
            and "L4" in stage_proportions.keys()
            and "adult" in stage_proportions.keys()
        ):
            # handle experiments where all larval stages are annotated

            hatch_time, m1, m2, m3, m4 = filemap.select(
                pl.cols(["HatchTime", "M1", "M2", "M3", "M4"])
            ).row(0)

            times = [0, hatch_time, m1, m2, m3, m4, m4 + extra_adulthood_time]
            stages = ["egg", "L1", "L2", "L3", "L4", "adult"]

        for i, stage in enumerate(stages):
            stage_beg = times[i]
            stage_end = times[i + 1]

            images = pick_within_larval_stage(
                filemap, stage_beg, stage_end, n_picks=n_picks
            )
            rows = []
            if images:
                for img in images:
                    img = img.replace(
                        "external.data/TowbinLab", "towbin.data/shared"
                    )  # fixes the path for old experiments
                    row = {
                        "Point": point,
                        "Image": img,
                        "Stage": stage,
                        "Microscope": microscope,
                        "Experiment": experiment_name,
                    }
                    rows.append(row)

            database = database.vstack(pl.DataFrame(rows))

    return database


def calculate_image_combinations(database_size, scope_proportions, stage_proportions):
    """Calculate number of images needed for each scope and stage combination."""
    combinations = {}

    # Calculate for each scope and stage combination
    for scope, scope_prop in scope_proportions.items():
        combinations[scope] = {}
        if not stage_proportions:
            n_images = int(database_size * scope_prop)
            combinations[scope]["unknown"] = n_images
        else:
            for stage, stage_prop in stage_proportions.items():
                n_images = int(database_size * scope_prop * stage_prop)
                combinations[scope][stage] = n_images

    return combinations


def process_row(row, output_dir, channel):
    try:
        image_path = row["Image"]
        image = read_tiff_file(image_path)
        channel = channel[0] if isinstance(channel, list) else channel
        if channel >= image.shape[0]:
            channel = image.shape[0] - 1
        image = image[channel, :, :]  # select the channel
        image_name = row["OutputName"]
        # save the image
        imwrite(os.path.join(output_dir, image_name), image, compression="zlib")
    except Exception as e:
        print(f"Error processing image {row['Image']}: {e}")


def extract_images(database, output_dir, channel):
    Parallel(n_jobs=1)(
        delayed(process_row)(row, output_dir, channel)
        for _, row in tqdm(database.iterrows())
    )


config_path = get_args().config
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
n_picks_per_stage = config.get("n_picks_per_stage", 10)

valid_scopes_expanded = []
for scope in valid_scopes:
    if scope in scopes_alt_names:
        valid_scopes_expanded.extend(scopes_alt_names[scope])
    else:
        valid_scopes_expanded.append(scope)

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
    os.makedirs(os.path.join(sub_db_dir, "masks"), exist_ok=True)

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

    database = pl.DataFrame()
    for filemap in filemaps:
        database = database.vstack(
            get_images_from_filemap(
                filemap,
                database_config,
                valid_scopes_expanded,
                extra_adulthood_time=extra_adulthood_time,
                n_picks=n_picks_per_stage,
            )
        )

    combinations = calculate_image_combinations(
        database_size=database_config.get("size", database.height),
        scope_proportions=database_config.get("scope_proportions", {}),
        stage_proportions=database_config.get("stage_proportions", {}),
    )

    dataset = pl.DataFrame()
    for scope, stages in combinations.items():
        for stage, n_images in stages.items():
            try:
                combination_images = database.filter(
                    (pl.col("Microscope") == scope) & (pl.col("Stage") == stage)
                )
                n = min(n_images, combination_images.height)
                combination_images = combination_images.sample(
                    n, with_replacement=False
                )

                dataset = dataset.vstack(combination_images)
            except Exception as e:
                print(
                    f"Warning: Could not sample {n_images} images for {scope}, stage {stage}, error: {e}"
                )
                continue

    dataset = dataset.sample(fraction=1, shuffle=True)  # shuffle the dataset

    def get_output_name(i, row):
        return f'image_{i}_{row["Microscope"]}_{row["Stage"]}.tif'

    dataset = dataset.with_columns(pl.arange(0, dataset.height).alias("Index"))

    dataset = dataset.with_columns(
        pl.struct(["Index", "Microscope", "Stage"])
        .map_elements(lambda x: get_output_name(x["Index"], x), return_dtype=pl.String)
        .alias("OutputName")
    )

    dataset = dataset.select(pl.exclude("Index", "Point"))
    dataset.write_csv(
        os.path.join(
            database_path, database_name, f"{database_name}_dataset_filemap.csv"
        )
    )
    dataset = dataset.to_pandas()

    extract_images(
        dataset,
        output_dir=os.path.join(database_path, database_name, "images"),
        channel=database_config.get("channel", 0),
    )

shutil.copy(config_path, os.path.join(database_path, "dataset_config.yaml"))
