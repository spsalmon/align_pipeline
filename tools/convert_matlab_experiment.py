import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from align_toolbox.foundation.file_handling import add_dir_to_experiment_filemap

sys.path.append(str(Path(__file__).parent.parent))
from align_pipeline.utils import (  # noqa: E402
    get_experiment_time_from_filemap_parallel,
)

KEY_CONVERSION_MAP = {
    "vol": "volume",
    "len": "length",
    "strClass": "qc",
    "ecdys": "ecdysis",
}


def convert_matlab_experiment(
    experiment_dir,
    matlab_report_dir,
    matlab_report_files,
    column_names,
    add_raw=False,
    analysis_to_add=None,
):
    matlab_report_files = [
        os.path.join(matlab_report_dir, f) for f in matlab_report_files
    ]
    dfs = []
    for matlab_report_file, column_name in zip(matlab_report_files, column_names):
        matlab_report = sio.loadmat(matlab_report_file, chars_as_strings=False)

        # Convert keys
        new_matlab_report = {}
        for key, value in matlab_report.items():
            if key.startswith("__"):
                continue
            try:
                new_key = KEY_CONVERSION_MAP.get(key)
                new_matlab_report[new_key] = value
            except Exception as e:
                print(f"Error converting key {key}: {e}")
                continue

        # convert qc from [' ', 'w', 'o', 'e'] to ['worm', 'egg', 'error']
        new_matlab_report["qc"] = np.where(
            new_matlab_report["qc"] == "w",
            "worm",
            new_matlab_report["qc"],
        )
        new_matlab_report["qc"] = np.where(
            new_matlab_report["qc"] == "e", "egg", new_matlab_report["qc"]
        )
        new_matlab_report["qc"] = np.where(
            new_matlab_report["qc"] == "o",
            "error",
            new_matlab_report["qc"],
        )
        new_matlab_report["qc"] = np.where(
            new_matlab_report["qc"] == " ",
            "error",
            new_matlab_report["qc"],
        )

        point = np.arange(0, new_matlab_report["volume"].shape[0])
        time = np.arange(0, new_matlab_report["volume"].shape[1])

        # combine each point with every time
        time_point = np.array(np.meshgrid(point, time)).T.reshape(-1, 2)

        time = time_point[:, 1]
        point = time_point[:, 0]

        # put everything into a nice dataframe
        data = {
            "Point": point,
            "Time": time,
            f"{column_name}_volume": new_matlab_report["volume"].flatten(),
            f"{column_name}_length": new_matlab_report["length"].flatten(),
            f"{column_name}_worm_type": new_matlab_report["qc"].flatten(),
        }

        df = pd.DataFrame(data)

        try:
            ecdysis = new_matlab_report["ecdysis"]
            hatch, M1, M2, M3, M4 = np.split(ecdysis, 5, axis=1)

            for point, molts in enumerate(ecdysis):
                HatchTime, M1, M2, M3, M4 = molts
                # convert to int if not nan
                if not np.isnan(HatchTime):
                    HatchTime = int(HatchTime) - 1
                if not np.isnan(M1):
                    M1 = int(M1) - 1
                if not np.isnan(M2):
                    M2 = int(M2) - 1
                if not np.isnan(M3):
                    M3 = int(M3) - 1
                if not np.isnan(M4):
                    M4 = int(M4) - 1

                df.loc[df["Point"] == point, "HatchTime"] = HatchTime
                df.loc[df["Point"] == point, "M1"] = M1
                df.loc[df["Point"] == point, "M2"] = M2
                df.loc[df["Point"] == point, "M3"] = M3
                df.loc[df["Point"] == point, "M4"] = M4
        except KeyError:
            print("No molts found in the matlab report")

        dfs.append(df)

    # combine the two dataframes by adding the columns of the second dataframe to the first dataframe
    df = pd.concat(dfs, axis=1)

    # remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # add the raw data to the experiment filemap
    if add_raw:
        df = add_dir_to_experiment_filemap(
            df, os.path.join(experiment_dir, "raw"), "raw"
        )

    # add the analysis data to the experiment filemap
    if analysis_to_add is not None:
        for key, value in analysis_to_add.items():
            df = add_dir_to_experiment_filemap(
                df, os.path.join(experiment_dir, "analysis", key), f"analysis/{value}"
            )

    # save the dataframe to a csv file
    save_path = os.path.join(matlab_report_dir, "analysis_filemap.csv")
    df.to_csv(save_path, index=False)

    return df, save_path


experiment_dir = "/mnt/towbin.data/shared/igheor/20230914_Ti2_10x_vhp-1_338_344_186_160_20230914_172832_008"
matlab_report_dir = os.path.join(experiment_dir, "analysis/report/")
old_matlab_report_files = [
    "ch1_il_strS_cor_sec2_validlength.mat",
    "ch2_sobel_str_molts_nw_cor_sec2_validlength.mat",
]
old_matlab_report_files = [
    os.path.join(matlab_report_dir, f) for f in old_matlab_report_files
]
better_column_names = ["ch1_seg_str", "ch2_seg_str"]

extract_experiment_time = False
add_raw = False
analysis_to_add = None

# analysis_to_add = {
#     'ch1_il' : 'ch1_seg',
#     'ch2_sobel' : 'ch2_seg',
#     # 'ch1_il_strS' : 'ch1_seg_str',
#     # 'ch2_sobel_str' : 'ch2_seg_str',
# }

experiment_filemap, experiment_filemap_path = convert_matlab_experiment(
    experiment_dir,
    matlab_report_dir,
    old_matlab_report_files,
    better_column_names,
    add_raw=add_raw,
    analysis_to_add=analysis_to_add,
)

# remove duplicate columns that have the same name and the same values
if "ExperimentTime" not in experiment_filemap.columns and extract_experiment_time:
    experiment_filemap["ExperimentTime"] = get_experiment_time_from_filemap_parallel(
        experiment_filemap
    )
    experiment_filemap.to_csv(experiment_filemap_path, index=False)
