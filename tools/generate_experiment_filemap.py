import os

from align_toolbox.foundation.file_handling import (
    add_dir_to_experiment_filemap,
    get_dir_filemap,
)

experiment_path = "path/to/experiment"
raw_dir_name = "raw"
dir_names_to_add = ["analysis/ch2_seg"]


raw_dir = os.path.join(experiment_path, raw_dir_name)
dir_to_add = [os.path.join(experiment_path, d) for d in dir_names_to_add]
report_directory = os.path.join(experiment_path, "analysis", "report")

os.makedirs(report_directory, exist_ok=True)

filemap = get_dir_filemap(raw_dir)
# change name of ImagePath column to raw
filemap = filemap.rename(columns={"ImagePath": raw_dir_name})
# add a worm type column filled with 'worm'
filemap["ch2_seg_str_worm_type"] = "worm"
# add a volume column filled with 2
filemap["ch2_seg_str_volume"] = 2

for d in dir_to_add:
    add_dir_to_experiment_filemap(filemap, d)

filemap.to_csv(os.path.join(report_directory, "analysis_filemap.csv"), index=False)
