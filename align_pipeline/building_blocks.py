"""Building blocks of the pipeline: one class per analysis step (segmentation,
straightening, morphology, quality control, molt detection, fluorescence, and a
user-supplied custom block). Validates the config, parses it into per-block
configurations, and builds the block objects that init_pipeline runs.
"""

import os
from abc import ABC, abstractmethod

import numpy as np
from align_toolbox.foundation.file_handling import add_dir_to_experiment_filemap

from align_pipeline.utils import (
    create_linker_command,
    get_input_and_output_files,
    get_output_name,
    get_python_command,
    pickle_objects,
    resolve_ref,
    run_command,
)

# Resolve bundled scripts/models relative to this package, so the pipeline
# works regardless of the current working directory.
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

OPTIONS_MAP = {
    "segmentation": [
        "rerun_segmentation",
        "segmentation_column",
        "segmentation_name_suffix",
        "segmentation_method",
        "segmentation_channels",
        "pixelsize",
        "gaussian_filter_sigma",
        "model_path",
        "predict_on_tiles",
        "tiler_config",
        "enforce_n_channels",
        "scale_factor",
        "activation_layer",
        "batch_size",
    ],
    "straightening": [
        "rerun_straightening",
        "straightening_source",
        "straightening_masks",
        "keep_biggest_object",
    ],
    "morphology_computation": [
        "rerun_morphology_computation",
        "morphology_computation_masks",
        "pixelsize",
        "morphological_features",
    ],
    "quality_control": [
        "rerun_quality_control",
        "qc_images",
        "qc_masks",
        "qc_model_path",
        "qc_import_eggs_from",
    ],
    "molt_detection": [
        "rerun_molt_detection",
        "molt_detection_method",
        "molt_detection_columns",
        "molt_detection_model_path",
        "molt_detection_volume",  # deprecated, use "molt_detection_columns" instead
        "molt_detection_batch_size",
    ],
    "fluorescence_quantification": [
        "rerun_fluorescence_quantification",
        "fluorescence_quantification_source",
        "fluorescence_quantification_masks",
        "fluorescence_quantification_aggregations",
        "fluorescence_background_aggregation",
    ],
    "custom": [
        "rerun_custom",
        "custom_script_path",
        "custom_script_name",
        "custom_script_return_type",
        "custom_script_parameters",
        "custom_script_requires_gpu",
    ],
}

DEFAULT_OPTIONS = {
    "segmentation": {
        # default options for segmentation, allows user to make their config file shorter
        "rerun_segmentation": [False],
        "segmentation_column": ["raw"],
        "segmentation_name_suffix": [None],
        "gaussian_filter_sigma": [1.0],
        "predict_on_tiles": [False],
        "tiler_config": [None],
        "enforce_n_channels": [None],
        "activation_layer": [None],
        "model_path": [None],
        "batch_size": [1],
        "scale_factor": [1.0],
    },
    "straightening": {
        "rerun_straightening": [False],
        "keep_biggest_object": [False],
    },
    "morphology_computation": {
        "rerun_morphology_computation": [False],
        "morphological_features": [["volume", "length", "area"]],
    },
    "quality_control": {
        "rerun_quality_control": [False],
        "qc_import_eggs_from": [None],
    },
    "molt_detection": {
        "rerun_molt_detection": [False],
        "molt_detection_method": ["deep_learning"],
        "molt_detection_model_path": [
            os.path.join(
                _PIPELINE_DIR, "defaults", "models", "molt_detection_model.ckpt"
            )
        ],
        "molt_detection_batch_size": [1],
        "molt_detection_volume": [
            None
        ],  # deprecated, use "molt_detection_columns" instead, this is for backward compatibility
    },
    "fluorescence_quantification": {
        "rerun_fluorescence_quantification": [False],
        "fluorescence_background_aggregation": ["median"],
    },
    "custom": {
        "rerun_custom": [False],
        "custom_script_requires_gpu": [False],
    },
}

# User-settable top-level keys that are not per-block options. Keep this in sync
# when adding a new config key: validate_config rejects any key that is neither
# here, a per-block option (OPTIONS_MAP), nor an sbatch_* SLURM key.
GLOBAL_CONFIG_KEYS = frozenset(
    {
        "experiment_dir",
        "analysis_dir_name",
        "raw_dir_name",
        "temp_dir",
        "cleanup_on_success",
        "report_format",
        "get_experiment_time",
        "overwrite_annotated_filemap",
        "time_regex",
        "point_regex",
        "n_jobs",
        "backend",
        "slurm_config",
        "python_command",
        "building_blocks",
        "groups",
    }
)

# Every per-block option name, flattened across block types.
ALL_OPTION_KEYS = frozenset(
    option for options in OPTIONS_MAP.values() for option in options
)

# Option keys whose values are input files/dirs that must exist before the run.
# Only inputs the user supplies -- never a folder an earlier block produces.
PATH_OPTIONS = (
    "model_path",
    "qc_model_path",
    "molt_detection_model_path",
    "custom_script_path",
)


class BuildingBlock(ABC):
    def __init__(
        self,
        name,
        options,
        block_config,
        return_type,
        worker_module,
        requires_gpu=False,
        requires_filemap=False,
    ):
        self.name = name
        self.options = options
        self.block_config = block_config
        self.return_type = return_type
        self.worker_module = worker_module
        self.requires_gpu = requires_gpu
        self.requires_filemap = requires_filemap

    def __str__(self):
        return f"{self.name}: {self.block_config}"

    @abstractmethod
    def get_output_name(self, config, subdir):
        pass

    @abstractmethod
    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        pass

    def create_command(
        self,
        python_command,
        input_pickle_path,
        output_pickle_path,
        pickled_block_config,
        pickled_config,
        config,
        pickled_filemap_path=None,
    ):
        # Worker compute parallelism (joblib --n_jobs), needed by every backend.
        # Falls back to the SLURM cpu request so configs that only set
        # sbatch_cpus keep working.
        n_jobs = config.get("n_jobs", config.get("sbatch_cpus", 1))

        # Run the worker as a module, so it resolves by import rather than by a
        # working-directory-relative path.
        command = (
            f"{python_command} -m {self.worker_module} --input {input_pickle_path} "
            f"--output {output_pickle_path} --block_config {pickled_block_config} "
            f"--config {pickled_config} --n_jobs {n_jobs}"
        )

        if pickled_filemap_path is not None:
            command += f" -f {pickled_filemap_path}"
        return command

    def run(self, experiment_filemap, config, subdir=None):
        block_config = self.block_config
        python_command = get_python_command(config)
        temp_dir = config["temp_dir"]

        if self.requires_filemap:
            pickled_filemap_path = pickle_objects(
                temp_dir,
                {"path": "experiment_filemap", "obj": experiment_filemap},
            )[0]
        else:
            pickled_filemap_path = None

        if self.return_type == "subdir":
            subdir = self.get_output_name(config, subdir)
            input_files, output_files = self.get_input_and_output_files(
                config, experiment_filemap, subdir
            )

            if len(input_files) != 0:
                (
                    input_pickle_path,
                    output_pickle_path,
                    pickled_block_config,
                    pickled_config,
                ) = pickle_objects(
                    temp_dir,
                    {"path": f"{self.name}_input_files", "obj": input_files},
                    {"path": f"{self.name}_output_files", "obj": output_files},
                    {"path": f"{self.name}_block_config", "obj": block_config},
                    {"path": f"{self.name}_config", "obj": config},
                )

                command = self.create_command(
                    python_command,
                    input_pickle_path,
                    output_pickle_path,
                    pickled_block_config,
                    pickled_config,
                    config,
                    pickled_filemap_path=pickled_filemap_path,
                )

                linker_command = create_linker_command(python_command, temp_dir, subdir)

                run_command(
                    command,
                    self.name,
                    config,
                    requires_gpu=self.requires_gpu,
                    run_linker=True,
                    linker_command=linker_command,
                )

            else:
                linker_command = create_linker_command(python_command, temp_dir, subdir)
                run_command(
                    "# No input files found, skipping this building block.",
                    self.name,
                    config,
                    requires_gpu=False,
                    run_linker=True,
                    linker_command=linker_command,
                )

            return subdir

        elif self.return_type == "csv":
            output_file = self.get_output_name(config, subdir)
            input_files, _ = self.get_input_and_output_files(
                config, experiment_filemap, config["analysis_subdir"]
            )

            rerun = (self.block_config[f"rerun_{self.name}"]) or (
                os.path.exists(output_file) is False
            )

            if len(input_files) != 0 and rerun:
                input_pickle_path, pickled_block_config, pickled_config = (
                    pickle_objects(
                        temp_dir,
                        {"path": "input_files", "obj": input_files},
                        {"path": "block_config", "obj": block_config},
                        {"path": "config", "obj": config},
                    )
                )

                command = self.create_command(
                    python_command,
                    input_pickle_path,
                    output_file,
                    pickled_block_config,
                    pickled_config,
                    config,
                    pickled_filemap_path=pickled_filemap_path,
                )

                linker_command = create_linker_command(
                    python_command, temp_dir, output_file
                )

                run_command(
                    command,
                    self.name,
                    config,
                    requires_gpu=self.requires_gpu,
                    run_linker=True,
                    linker_command=linker_command,
                )

            else:
                linker_command = create_linker_command(
                    python_command, temp_dir, output_file
                )
                run_command(
                    "# No input files found, skipping this building block.",
                    self.name,
                    config,
                    requires_gpu=False,
                    run_linker=True,
                    linker_command=linker_command,
                )

            return output_file

        else:
            # A block that produces nothing the filemap knows about: run it, then
            # chain to the next block with no result to record. The inputs are
            # pickled under the same name the csv branch uses, because
            # CustomBuildingBlock.create_command passes that pickle as --filemap.
            input_files, _ = self.get_input_and_output_files(
                config, experiment_filemap, config["analysis_subdir"]
            )

            input_pickle_path, pickled_block_config, pickled_config = pickle_objects(
                temp_dir,
                {"path": "input_files", "obj": input_files},
                {"path": "block_config", "obj": block_config},
                {"path": "config", "obj": config},
            )

            command = self.create_command(
                python_command,
                input_pickle_path,
                None,
                pickled_block_config,
                pickled_config,
                config,
                pickled_filemap_path=pickled_filemap_path,
            )

            run_command(
                command,
                self.name,
                config,
                requires_gpu=self.requires_gpu,
                run_linker=True,
                linker_command=create_linker_command(python_command, temp_dir),
            )

            return None


class SegmentationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        NON_LEARNING_METHODS = ["double_threshold", "edge_based", "threshold"]
        LEARNING_BASED_METHODS = ["deep_learning", "cellpose"]

        if block_config["segmentation_method"] in NON_LEARNING_METHODS:
            requires_gpu = False
            worker_module = "align_pipeline.workers.segmentation_non_learning"
        elif block_config["segmentation_method"] in LEARNING_BASED_METHODS:
            requires_gpu = True
            worker_module = "align_pipeline.workers.segmentation_learning_based"
        else:
            raise ValueError(
                f"Segmentation method {block_config['segmentation_method']} not supported."
            )

        super().__init__(
            "segmentation",
            OPTIONS_MAP["segmentation"],
            block_config,
            "subdir",
            worker_module,
            requires_gpu,
        )

    def get_output_name(self, config, subdir):
        return get_output_name(
            config,
            self.block_config["segmentation_column"],
            "seg",
            channels=self.block_config["segmentation_channels"],
            subdir=subdir,
            return_subdir=True,
            add_raw=False,
            custom_suffix=self.block_config["segmentation_name_suffix"],
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        input_files, output_files = get_input_and_output_files(
            experiment_filemap,
            [resolve_ref(self.block_config["segmentation_column"], config)],
            subdir,
            rerun=self.block_config["rerun_segmentation"],
        )

        return input_files, output_files


class StraighteningBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        worker_module = "align_pipeline.workers.straightening"
        super().__init__(
            "straightening",
            OPTIONS_MAP["straightening"],
            block_config,
            "subdir",
            worker_module,
        )

    def get_output_name(self, config, subdir):
        return get_output_name(
            config,
            self.block_config["straightening_source"][0],
            "str",
            subdir=subdir,
            channels=self.block_config["straightening_source"][1],
            return_subdir=True,
            add_raw=True,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        block_config = self.block_config
        columns = [
            resolve_ref(block_config["straightening_source"][0], config),
            resolve_ref(block_config["straightening_masks"], config),
        ]

        for column in columns:
            if column not in experiment_filemap.columns:
                try:
                    column_subdir = os.path.join(config["experiment_dir"], column)
                    experiment_filemap = add_dir_to_experiment_filemap(
                        experiment_filemap, column_subdir, column
                    )
                    if config["filemap_path"].endswith(".parquet"):
                        experiment_filemap.write_parquet(config["filemap_path"])
                    else:
                        experiment_filemap.write_csv(config["filemap_path"])
                except Exception as e:
                    raise ValueError(
                        f"Column {column} not found in experiment filemap and could not be added. Error: {e}"
                    )

        input_files, straightening_output_files = get_input_and_output_files(
            experiment_filemap,
            columns,
            subdir,
            rerun=block_config["rerun_straightening"],
        )

        if len(input_files) != 0:
            input_files = [
                {"source_image_path": input_source_image, "mask_path": input_mask}
                for input_source_image, input_mask in input_files
            ]

        return input_files, straightening_output_files


class QualityControlBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        worker_module = "align_pipeline.workers.quality_control"
        super().__init__(
            "quality_control",
            OPTIONS_MAP["quality_control"],
            block_config,
            "csv",
            worker_module,
            requires_filemap=True,
        )
        self.mask_only = (
            block_config["qc_images"] is None or block_config["qc_images"][0] is None
        )

    def get_output_name(self, config, subdir):
        return get_output_name(
            config,
            self.block_config["qc_masks"],
            "qc",
            subdir=subdir,
            return_subdir=False,
            add_raw=False,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        block_config = self.block_config
        if self.mask_only:
            columns = [resolve_ref(block_config["qc_masks"], config)]
        else:
            columns = [
                resolve_ref(block_config["qc_images"][0], config),
                resolve_ref(block_config["qc_masks"], config),
            ]

        input_files, _ = get_input_and_output_files(
            experiment_filemap,
            columns,
            subdir,
        )

        if len(input_files) != 0:
            if self.mask_only:
                input_files = [
                    {"image_path": None, "mask_path": mask_path[0]}
                    for mask_path in input_files
                ]
            else:
                input_files = [
                    {"image_path": image_path, "mask_path": mask_path}
                    for image_path, mask_path in input_files
                ]

        return input_files, None


class MorphologyComputationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        worker_module = "align_pipeline.workers.morphology_computation"
        super().__init__(
            "morphology_computation",
            OPTIONS_MAP["morphology_computation"],
            block_config,
            "csv",
            worker_module,
        )

    def get_output_name(self, config, subdir):
        return get_output_name(
            config,
            self.block_config["morphology_computation_masks"],
            "morphology",
            subdir=subdir,
            return_subdir=False,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        morphology_computation_masks = [
            resolve_ref(self.block_config["morphology_computation_masks"], config)
        ]
        analysis_subdir = config["analysis_subdir"]

        input_files, _ = get_input_and_output_files(
            experiment_filemap,
            morphology_computation_masks,
            analysis_subdir,
        )

        input_files = [input_file[0] for input_file in input_files]

        return input_files, None


class MoltDetectionBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        worker_module = "align_pipeline.workers.molt_detection"
        super().__init__(
            "molt_detection",
            OPTIONS_MAP["molt_detection"],
            block_config,
            "csv",
            worker_module,
        )

    def get_output_name(self, config, subdir):
        return os.path.join(
            config["report_subdir"], f"ecdysis.{config['report_format']}"
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        return config["filemap_path"], None


class FluorescenceQuantificationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        worker_module = "align_pipeline.workers.fluorescence_quantification"
        super().__init__(
            "fluorescence_quantification",
            OPTIONS_MAP["fluorescence_quantification"],
            block_config,
            "csv",
            worker_module,
        )

    def get_output_name(self, config, subdir):
        fluorescence_quantification_source = self.block_config[
            "fluorescence_quantification_source"
        ][0]

        fluorescence_quantification_channel = self.block_config[
            "fluorescence_quantification_source"
        ][1]

        fluorescence_quantification_masks_name = self.block_config[
            "fluorescence_quantification_masks"
        ].split("/")[-1]

        output_name_suffix = f"on_{fluorescence_quantification_masks_name}"

        return get_output_name(
            config,
            fluorescence_quantification_source,
            "fluo_quant",
            channels=fluorescence_quantification_channel,
            return_subdir=False,
            add_raw=False,
            suffix=output_name_suffix,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        fluorescence_quantification_source = self.block_config[
            "fluorescence_quantification_source"
        ][0]

        columns = [
            resolve_ref(fluorescence_quantification_source, config),
            resolve_ref(self.block_config["fluorescence_quantification_masks"], config),
        ]

        input_files, _ = get_input_and_output_files(experiment_filemap, columns, subdir)

        if len(input_files) != 0:
            input_files = [
                {"source_image_path": input_source_image, "mask_path": input_mask}
                for input_source_image, input_mask in input_files
            ]

        return input_files, None


class CustomBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        # No worker module: custom blocks run a user-supplied script and override
        # create_command below.
        super().__init__(
            "custom",
            OPTIONS_MAP["custom"],
            block_config,
            block_config["custom_script_return_type"],
            None,
            requires_gpu=block_config.get("custom_script_requires_gpu", False),
            requires_filemap=False,
        )

    def get_output_name(self, config, subdir):
        custom_script_name = self.block_config["custom_script_name"]

        if self.return_type == "subdir":
            output = os.path.join(config["analysis_subdir"], custom_script_name)
            if subdir is not None:
                output = os.path.join(output, subdir)
            return output

        if self.return_type == "csv":
            name = (
                custom_script_name
                if subdir is None
                else f"{subdir}_{custom_script_name}"
            )
            return os.path.join(
                config["report_subdir"], f"{name}.{config['report_format']}"
            )

        # A block with no return type has no output path.
        return None

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        return experiment_filemap, None

    def create_command(
        self,
        python_command,
        input_pickle_path,
        output_pickle_path,
        pickled_block_config,
        pickled_config,
        config,
        pickled_filemap_path=None,
    ):
        custom_script_path = self.block_config["custom_script_path"]

        # concatenate the elements of the custom_script_parameters list
        custom_script_parameters = " ".join(
            self.block_config["custom_script_parameters"]
        )

        if custom_script_path.endswith(".sh"):
            command = f"bash {custom_script_path} --filemap {input_pickle_path} --output {output_pickle_path} --block_config {pickled_block_config} --config {pickled_config} {custom_script_parameters}"
        elif custom_script_path.endswith(".py"):
            command = f"{python_command} {custom_script_path} --filemap {input_pickle_path} --block_config {pickled_block_config} --output {output_pickle_path} --config {pickled_config} {custom_script_parameters}"
        else:
            print(
                f"Script type of {custom_script_path} is not supported. The pipeline only supports bash or python scripts."
            )
        return command


def validate_config(config):
    # Pre-flight check run before any run dir or job is created. Collects every
    # problem and raises them together, rather than failing on the first.
    errors = []

    for key in ("experiment_dir", "building_blocks"):
        if key not in config:
            errors.append(f"missing required key '{key}'")

    # Flag typo'd / unrecognised keys. sbatch_* keys (merged from the slurm
    # config) are cluster-specific and free-form, so they pass by prefix.
    for key in config:
        if (
            key in GLOBAL_CONFIG_KEYS
            or key in ALL_OPTION_KEYS
            or key.startswith("sbatch_")
        ):
            continue
        errors.append(f"unknown config key '{key}'")

    blocks = config.get("building_blocks")
    if "building_blocks" in config and (not isinstance(blocks, list) or not blocks):
        errors.append("'building_blocks' must be a non-empty list")
    elif isinstance(blocks, list):
        for name in blocks:
            if name == "classification":
                errors.append(
                    "building block 'classification' was replaced by "
                    "'quality_control'; update your config"
                )
            elif name not in OPTIONS_MAP:
                errors.append(f"unknown building block '{name}'")

        # Each per-block option list must hold one value per block of that type
        # (or a single value broadcast to all of them).
        counts = count_building_blocks_types(blocks)
        for name, indices in counts.items():
            if name not in OPTIONS_MAP:
                continue
            n = len(indices)
            for option in OPTIONS_MAP[name]:
                if option not in config:
                    if option not in DEFAULT_OPTIONS.get(name, {}):
                        errors.append(
                            f"'{option}' is required for the '{name}' building block"
                        )
                    continue
                value = config[option]
                if not isinstance(value, list):
                    errors.append(f"'{option}' must be a list")
                elif len(value) not in (1, n):
                    errors.append(
                        f"'{option}' has {len(value)} value(s) but there are "
                        f"{n} '{name}' block(s) (expected 1 or {n})"
                    )

    # Input paths given in the config must already exist. experiment_dir is a
    # directory; the model/script options are lists of files (skip None and any
    # non-string entry). Folders produced by earlier blocks are NOT checked --
    # they do not exist yet at validation time.
    experiment_dir = config.get("experiment_dir")
    if isinstance(experiment_dir, str) and not os.path.isdir(experiment_dir):
        errors.append(f"'experiment_dir' does not exist: {experiment_dir}")
    for option in PATH_OPTIONS:
        values = config.get(option)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and not os.path.isfile(value):
                errors.append(f"'{option}' file does not exist: {value}")

    backend = config.get("backend", "slurm")
    if backend not in ("slurm", "local"):
        errors.append(f"'backend' must be 'slurm' or 'local', got '{backend}'")

    report_format = config.get("report_format", "csv")
    if report_format not in ("csv", "parquet"):
        errors.append(
            f"'report_format' must be 'csv' or 'parquet', got '{report_format}'"
        )

    if errors:
        raise ValueError(
            "Invalid pipeline config:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def count_building_blocks_types(building_block_names):
    building_block_counts = {}
    for i, building_block in enumerate(building_block_names):
        if building_block not in building_block_counts:
            building_block_counts[building_block] = []
        building_block_counts[building_block] += [i]
    return building_block_counts


def parse_building_blocks_config(config):
    building_block_names = config["building_blocks"]
    building_block_counts = count_building_blocks_types(building_block_names)

    blocks_config = {}

    for i, building_block_name in enumerate(building_block_names):
        config_copy = config.copy()
        if building_block_name in OPTIONS_MAP:
            options = OPTIONS_MAP[building_block_name]
            # assert options
            for option in options:
                try:
                    assert (
                        len(config[option])
                        == len(building_block_counts[building_block_name])
                        or len(config[option]) == 1
                    ), f"{config[option]} The number of {option} options ({len(config[option])}) does not match the number of {building_block_name} building blocks ({len(building_block_counts[building_block_name])})"

                except KeyError:
                    if option in DEFAULT_OPTIONS[building_block_name]:
                        config_copy[option] = DEFAULT_OPTIONS[building_block_name][
                            option
                        ]
                        print(
                            f"{option} not found in config file, using default value: {config_copy[option]}"
                        )
                    else:
                        raise KeyError(
                            f"{option} is not in the config file, but is required for the {building_block_name} building block."
                        )

            # expand single options to match the number of blocks
            for option in options:
                if len(config_copy[option]) == 1:
                    config_copy[option] = config_copy[option] * len(
                        building_block_counts[building_block_name]
                    )

            # find the index of the building block
            idx = np.argwhere(
                np.array(building_block_counts[building_block_name]) == i
            ).squeeze()

            # set the options for the building block
            block_options = {}
            for option in options:
                block_options[option] = config_copy[option][idx]

            # add the building block name to the block options
            block_options["name"] = building_block_name

            blocks_config[i] = block_options
        else:
            raise ValueError(f"Unknown building block name : {building_block_name}")

    return blocks_config


def create_building_blocks(blocks_config):
    building_blocks = []
    for i, block_config in blocks_config.items():
        if block_config["name"] == "segmentation":
            building_block = SegmentationBuildingBlock(block_config)
        elif block_config["name"] == "straightening":
            building_block = StraighteningBuildingBlock(block_config)
        elif (
            block_config["name"] == "morphology_computation"
            or block_config["name"] == "volume_computation"
        ):  # volume_computation is there for backward compatibility
            building_block = MorphologyComputationBuildingBlock(block_config)
        elif block_config["name"] == "quality_control":
            building_block = QualityControlBuildingBlock(block_config)
        elif block_config["name"] == "classification":
            raise ValueError(
                "'classification' building block was replaced by 'quality_control'. Please check the wiki and update your config file."
            )
        elif block_config["name"] == "molt_detection":
            building_block = MoltDetectionBuildingBlock(block_config)
        elif block_config["name"] == "fluorescence_quantification":
            building_block = FluorescenceQuantificationBuildingBlock(block_config)
        elif block_config["name"] == "custom":
            building_block = CustomBuildingBlock(block_config)
        else:
            raise ValueError(f"Building block {block_config['name']} not supported.")
        building_blocks.append(building_block)

    return building_blocks


def parse_and_create_building_blocks(config):
    blocks_config = parse_building_blocks_config(config)
    building_blocks = create_building_blocks(blocks_config)
    return building_blocks
