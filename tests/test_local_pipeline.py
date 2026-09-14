"""Test suite for the pipeline.

Unit tests for the pure helpers (config validation and parsing, output naming,
folder refs, input selection, slurm resolution, logging) plus an end-to-end
smoke test for the local backend: generates tiny synthetic images, runs the
pipeline with `backend: local` (threshold segmentation -> area morphology), and
checks the outputs. No slurm, no micromamba, no bundled data.
"""

import csv
import os
import subprocess
import sys

import pytest

pytest.importorskip("towbintools")
np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
yaml = pytest.importorskip("yaml")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # import the package directly for unit tests


def _write_image(path, blob_width):
    # 3-channel image with a bright rectangle in channel 0 (a worm proxy).
    image = np.zeros((3, 128, 128), dtype=np.uint16)
    image[0, 30:100, 20 : 20 + blob_width] = 4000
    # Store channels as separate planes (explicit, so the reader sees 3 channels).
    tifffile.imwrite(str(path), image, photometric="minisblack")


def _build_experiment(
    tmp_path,
    config_experiment_dir=None,
    analysis_dir_name="analysis",
    extra_config=None,
):
    # Tiny 2-image raw experiment + a local-backend config; returns config_path.
    # config_experiment_dir overrides what the config records as experiment_dir.
    # extra_config merges extra keys into the config before writing.
    raw_dir = tmp_path / "exp" / "raw"
    raw_dir.mkdir(parents=True)
    _write_image(raw_dir / "Time000000_Point000000_synthetic.tiff", 50)
    _write_image(raw_dir / "Time000001_Point000000_synthetic.tiff", 55)

    config = {
        "experiment_dir": config_experiment_dir or str(tmp_path / "exp"),
        "analysis_dir_name": analysis_dir_name,
        "raw_dir_name": "raw",
        "report_format": "csv",
        "pixelsize": [0.65],
        "backend": "local",
        "get_experiment_time": False,
        "n_jobs": 1,
        "building_blocks": ["segmentation", "morphology_computation"],
        "segmentation_column": ["raw"],
        "segmentation_method": ["threshold"],
        "segmentation_channels": [[0]],
        "segmentation_name_suffix": [None],
        "morphology_computation_masks": [f"{analysis_dir_name}/ch1_seg"],
        "morphological_features": [["area"]],
    }
    if extra_config:
        config.update(extra_config)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


def _run_pipeline(config_path, extra_args=(), cwd=REPO_ROOT):
    # PYTHONPATH lets `-m towbintools_pipeline...` resolve from any cwd; the
    # local backend launches the workers with this same interpreter and resolves
    # their scripts by absolute path, so cwd only affects a relative temp dir.
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "towbintools_pipeline.init_pipeline",
            "-c",
            str(config_path),
            *extra_args,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


# ---- Config validation ----


def _valid_config(experiment_dir=REPO_ROOT):
    # A minimal config that validate_config accepts. experiment_dir defaults to
    # an existing directory so the path-existence check passes.
    return {
        "experiment_dir": experiment_dir,
        "pixelsize": [0.65],
        "building_blocks": ["segmentation", "morphology_computation"],
        "segmentation_method": ["threshold"],
        "segmentation_channels": [[0]],
        "morphology_computation_masks": ["ch1_seg"],
    }


def test_validate_config_accepts_valid():
    from towbintools_pipeline.building_blocks import validate_config

    validate_config(_valid_config())  # must not raise


def test_validate_config_missing_required_keys():
    from towbintools_pipeline.building_blocks import validate_config

    with pytest.raises(ValueError) as excinfo:
        validate_config({})
    message = str(excinfo.value)
    assert "experiment_dir" in message
    assert "building_blocks" in message


def test_validate_config_unknown_block():
    from towbintools_pipeline.building_blocks import validate_config

    config = _valid_config()
    config["building_blocks"] = ["segmentation", "not_a_block"]
    with pytest.raises(ValueError, match="unknown building block 'not_a_block'"):
        validate_config(config)


def test_validate_config_classification_hint():
    from towbintools_pipeline.building_blocks import validate_config

    config = _valid_config()
    config["building_blocks"] = ["classification"]
    with pytest.raises(ValueError, match="'classification' was replaced by"):
        validate_config(config)


def test_validate_config_list_length_mismatch():
    from towbintools_pipeline.building_blocks import validate_config

    # Two segmentation blocks but only enough channels for one and not the two
    # allowed lengths (1 or 2).
    config = _valid_config()
    config["building_blocks"] = ["segmentation", "segmentation"]
    config["segmentation_channels"] = [[0], [1], [2]]
    with pytest.raises(ValueError, match="segmentation_channels"):
        validate_config(config)


def test_validate_config_reports_all_errors_at_once():
    from towbintools_pipeline.building_blocks import validate_config

    config = _valid_config()
    config["backend"] = "cluster"
    config["report_format"] = "xlsx"
    with pytest.raises(ValueError) as excinfo:
        validate_config(config)
    message = str(excinfo.value)
    assert "backend" in message
    assert "report_format" in message


def test_validate_config_unknown_key():
    from towbintools_pipeline.building_blocks import validate_config

    config = _valid_config()
    config["pixlesize"] = [0.65]  # typo of pixelsize
    with pytest.raises(ValueError, match="unknown config key 'pixlesize'"):
        validate_config(config)


def test_validate_config_allows_sbatch_and_groups_keys():
    from towbintools_pipeline.building_blocks import validate_config

    # sbatch_* keys (merged from the slurm config) and groups are recognised.
    config = _valid_config()
    config["sbatch_memory"] = "8G"
    config["groups"] = [["a", "b"]]
    validate_config(config)  # must not raise


def test_validate_config_experiment_dir_must_exist():
    from towbintools_pipeline.building_blocks import validate_config

    config = _valid_config(experiment_dir="/no/such/dir")
    with pytest.raises(ValueError, match="'experiment_dir' does not exist"):
        validate_config(config)


def test_validate_config_model_path_must_exist(tmp_path):
    from towbintools_pipeline.building_blocks import validate_config

    # A model file that does not exist is rejected...
    config = _valid_config()
    config["model_path"] = [str(tmp_path / "missing.ckpt")]
    with pytest.raises(ValueError, match="'model_path' file does not exist"):
        validate_config(config)
    # ...but an existing file passes.
    model = tmp_path / "model.ckpt"
    model.write_text("")
    config["model_path"] = [str(model)]
    validate_config(config)  # must not raise


def test_pipeline_rejects_invalid_config(tmp_path):
    # An invalid config fails at startup, before any run dir is created.
    config_path = _build_experiment(
        tmp_path, extra_config={"building_blocks": ["not_a_block"]}
    )

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert result.returncode != 0
    assert "unknown building block 'not_a_block'" in result.stderr
    assert not (tmp_path / "pipeline_temp").exists()


# ---- Building-block config parsing ----


def test_parse_building_blocks_broadcast_and_select():
    # A single option value is broadcast to every block of its type; a per-block
    # list is selected positionally; a missing option falls back to its default.
    from towbintools_pipeline.building_blocks import parse_building_blocks_config

    config = {
        "building_blocks": ["segmentation", "segmentation", "morphology_computation"],
        "pixelsize": [0.65],
        "segmentation_method": ["threshold"],  # single value, broadcast to both
        "segmentation_channels": [[0], [1]],  # one per segmentation block
        "morphology_computation_masks": ["ch1_seg"],
    }

    blocks = parse_building_blocks_config(config)

    assert blocks[0]["name"] == "segmentation"
    assert blocks[0]["segmentation_channels"] == [0]
    assert blocks[1]["segmentation_channels"] == [1]
    assert (
        blocks[0]["segmentation_method"]
        == blocks[1]["segmentation_method"]
        == "threshold"
    )
    assert blocks[0]["rerun_segmentation"] is False  # from the block defaults
    assert blocks[2]["name"] == "morphology_computation"
    assert blocks[2]["morphology_computation_masks"] == "ch1_seg"
    assert blocks[2]["morphological_features"] == ["volume", "length", "area"]


# ---- Output naming ----


def _naming_config(tmp_path):
    # The subset of config keys get_output_name reads.
    analysis = tmp_path / "analysis"
    report = analysis / "report"
    report.mkdir(parents=True)
    return {
        "analysis_subdir": str(analysis),
        "report_subdir": str(report),
        "raw_dir_name": "raw",
        "analysis_dir_name": "analysis",
        "report_format": "csv",
    }


def test_get_output_name_channels_and_raw(tmp_path):
    # Channels prefix as ch{n+1}; a raw input contributes no basename unless
    # add_raw is set. A subdir output lives under the analysis dir and is created.
    from towbintools_pipeline.utils import get_output_name

    config = _naming_config(tmp_path)
    seg = get_output_name(config, "raw", "seg", channels=[1, 0], add_raw=False)
    assert os.path.basename(seg) == "ch2_ch1_seg"
    assert seg == os.path.join(config["analysis_subdir"], "ch2_ch1_seg")
    assert os.path.isdir(seg)

    straight = get_output_name(config, "raw", "str", channels=[1], add_raw=True)
    assert os.path.basename(straight) == "ch2_raw_str"


def test_get_output_name_raw_ref_with_renamed_raw_dir(tmp_path):
    # A "raw" segmentation ref must still name its output ch2_seg (not ch2_raw_seg)
    # when raw_dir_name is renamed; add_raw still forces the basename in.
    from towbintools_pipeline.utils import get_output_name

    config = _naming_config(tmp_path)
    config["raw_dir_name"] = "raw_subset"
    seg = get_output_name(config, "raw", "seg", channels=[1], add_raw=False)
    assert os.path.basename(seg) == "ch2_seg"
    straight = get_output_name(config, "raw", "str", channels=[1], add_raw=True)
    assert os.path.basename(straight) == "ch2_raw_str"


def test_get_output_name_prefix_strip_and_report(tmp_path):
    # An analysis-dir-prefixed input is reduced to its basename; return_subdir
    # False yields a report file named with the report_format extension, with
    # suffix appended.
    from towbintools_pipeline.utils import get_output_name

    config = _naming_config(tmp_path)
    straight = get_output_name(config, "analysis/ch2_seg", "str", add_raw=True)
    assert os.path.basename(straight) == "ch2_seg_str"

    report = get_output_name(
        config, "ch2_seg_str", "morphology", return_subdir=False, suffix="v2"
    )
    assert report == os.path.join(
        config["report_subdir"], "ch2_seg_str_morphology_v2.csv"
    )


# ---- Folder references ----


def test_resolve_ref():
    # Directory refs normalize to `{analysis_dir_name}/{name}` whether or not they
    # carry a prefix; raw and absolute paths pass through.
    from towbintools_pipeline.utils import resolve_ref

    assert resolve_ref("ch2_seg", {}) == "analysis/ch2_seg"
    assert resolve_ref("analysis/ch2_seg", {}) == "analysis/ch2_seg"
    assert resolve_ref("raw", {}) == "raw"
    # literal "raw" always resolves to the raw column, even when it is renamed
    assert resolve_ref("raw", {"raw_dir_name": "raw_subset"}) == "raw_subset"
    assert resolve_ref("raw_subset", {"raw_dir_name": "raw_subset"}) == "raw_subset"
    # a renamed analysis_dir_name re-homes both forms to the new prefix
    assert resolve_ref("ch2_seg", {"analysis_dir_name": "output"}) == "output/ch2_seg"
    assert (
        resolve_ref("analysis/ch2_seg", {"analysis_dir_name": "output"})
        == "output/ch2_seg"
    )
    # absolute paths are left alone
    assert resolve_ref("/mnt/data/ch2_seg", {}) == "/mnt/data/ch2_seg"


def test_warnings_filter():
    # The curated rules suppress their targets; unrelated warnings pass through.
    import warnings

    from towbintools_pipeline.warnings_filter import configure_warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.resetwarnings()
        configure_warnings()
        warnings.warn(
            "pin_memory argument is set as true but no accelerator is found",
            UserWarning,
        )
        warnings.warn("an unrelated warning that should pass through", UserWarning)

    messages = [str(w.message) for w in rec]
    assert not any("pin_memory" in m for m in messages)
    assert any("unrelated" in m for m in messages)


# ---- Per-file input selection ----


def test_process_input_output_files_rejects_bad_rows(tmp_path):
    # An empty row, or one holding a None / blank / non-string entry, is dropped.
    from towbintools_pipeline.utils import process_input_output_files

    out = str(tmp_path)
    assert process_input_output_files([], out, False) == (None, None)
    assert process_input_output_files([None], out, False) == (None, None)
    assert process_input_output_files([""], out, False) == (None, None)
    assert process_input_output_files([123], out, False) == (None, None)


def test_process_input_output_files_rerun_and_existing(tmp_path):
    # A valid row maps to <output_dir>/<input basename>; an already-produced
    # output is skipped unless rerun is set.
    from towbintools_pipeline.utils import process_input_output_files

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    src = "/data/a.tiff"

    files, output = process_input_output_files([src], str(out_dir), False)
    assert files == [src]
    assert output == os.path.join(str(out_dir), "a.tiff")

    (out_dir / "a.tiff").write_text("x")  # the output now exists
    assert process_input_output_files([src], str(out_dir), False) == (None, None)
    files_again, _ = process_input_output_files([src], str(out_dir), True)
    assert files_again == [src]


# ---- SLURM config resolution ----


def test_merge_slurm_config_resolves_relative_sibling(tmp_path):
    # A relative slurm_config resolves next to the main config file; its keys
    # merge in, but inline sbatch_* keys still win.
    from towbintools_pipeline.utils import merge_slurm_config

    (tmp_path / "slurm_config.yaml").write_text(
        "sbatch_time: 0-02:00:00\nsbatch_cpus: 8\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "backend: slurm\nslurm_config: slurm_config.yaml\nsbatch_cpus: 4\n"
    )

    config = yaml.safe_load(config_path.read_text())
    merged = merge_slurm_config(config, str(config_path))
    assert merged["sbatch_time"] == "0-02:00:00"  # from the file
    assert merged["sbatch_cpus"] == 4  # inline wins


def test_merge_slurm_config_missing_is_skipped(tmp_path):
    # A missing slurm_config is skipped, leaving the config untouched.
    from towbintools_pipeline.utils import merge_slurm_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text("backend: slurm\nslurm_config: does_not_exist.yaml\n")

    config = yaml.safe_load(config_path.read_text())
    merged = merge_slurm_config(config, str(config_path))
    assert "sbatch_time" not in merged


def test_resolve_block_slurm_default_and_override():
    # A block with no override gets the shared defaults; an overridden type gets
    # the defaults merged with its entry (the override winning per key).
    from towbintools_pipeline.utils import resolve_block_slurm

    config = {
        "sbatch_cpus": 8,
        "sbatch_memory": "16G",
        "sbatch_time": "0-02:00:00",
        "sbatch_overrides": {
            "segmentation": {"sbatch_gpus": "rtx6000:1", "sbatch_memory": "32G"}
        },
    }

    morph = resolve_block_slurm(config, "morphology_computation")
    assert morph == {
        "sbatch_cpus": 8,
        "sbatch_memory": "16G",
        "sbatch_time": "0-02:00:00",
    }

    seg = resolve_block_slurm(config, "segmentation")
    assert seg["sbatch_gpus"] == "rtx6000:1"
    assert seg["sbatch_memory"] == "32G"  # override wins
    assert seg["sbatch_cpus"] == 8  # default kept


def test_resolve_init_slurm():
    # The outer job gets the defaults overlaid with sbatch_init.
    from towbintools_pipeline.utils import resolve_init_slurm

    config = {
        "sbatch_cpus": 32,
        "sbatch_memory": "64G",
        "sbatch_init": {"sbatch_cpus": 4, "sbatch_memory": "8G"},
    }

    init = resolve_init_slurm(config)
    assert init == {"sbatch_cpus": 4, "sbatch_memory": "8G"}


def test_resolve_init_slurm_drops_gpu():
    # The orchestrator job never needs the GPU, so it isn't inherited from the
    # shared defaults (but shared extras like --account still flow through).
    from towbintools_pipeline.utils import resolve_init_slurm

    config = {
        "sbatch_cpus": 32,
        "sbatch_gpus": "rtx6000:1",
        "sbatch_extra_options": ["--account=gratis"],
        "sbatch_init": {"sbatch_cpus": 4},
    }

    init = resolve_init_slurm(config)
    assert "sbatch_gpus" not in init
    assert init["sbatch_cpus"] == 4
    assert init["sbatch_extra_options"] == ["--account=gratis"]


def test_slurm_extra_options_accumulate():
    # Scalar keys are replaced by a section, but sbatch_extra_options entries are
    # appended, so a cluster-wide option is never silently dropped.
    from towbintools_pipeline.utils import resolve_block_slurm
    from towbintools_pipeline.utils import resolve_init_slurm

    config = {
        "sbatch_cpus": 32,
        "sbatch_extra_options": ["--account=gratis"],
        "sbatch_overrides": {
            "segmentation": {
                "sbatch_cpus": 8,
                "sbatch_extra_options": ["--partition=gpu"],
            }
        },
        "sbatch_init": {"sbatch_extra_options": ["--gres=pipelinecapacity:1"]},
    }

    seg = resolve_block_slurm(config, "segmentation")
    assert seg["sbatch_cpus"] == 8  # scalar replaced
    assert seg["sbatch_extra_options"] == ["--account=gratis", "--partition=gpu"]

    # A block without an override keeps only the shared extras.
    other = resolve_block_slurm(config, "straightening")
    assert other["sbatch_extra_options"] == ["--account=gratis"]

    init = resolve_init_slurm(config)
    assert init["sbatch_extra_options"] == [
        "--account=gratis",
        "--gres=pipelinecapacity:1",
    ]


def test_build_resource_directives():
    # Standard directives emitted only when set; extras appended verbatim.
    from towbintools_pipeline.utils import build_resource_directives

    assert build_resource_directives(8, "0-02:00:00", "16G", "rtx6000:1", None) == [
        "-c 8",
        "-t 0-02:00:00",
        "--mem=16G",
        "--gres=gpu:rtx6000:1",
    ]
    # memory omitted, replaced via an extra option
    assert build_resource_directives(4, None, None, None, ["--mem-per-cpu=4G"]) == [
        "-c 4",
        "--mem-per-cpu=4G",
    ]


def test_run_params_sbatch_init_flags():
    # The helper turns sbatch_init (over the defaults, minus GPU) into flags.
    from towbintools_pipeline.run_params import sbatch_init_flags

    config = {
        "sbatch_cpus": 32,
        "sbatch_memory": "64G",
        "sbatch_gpus": "rtx6000:1",
        "sbatch_extra_options": ["--account=gratis"],
        "sbatch_init": {
            "sbatch_cpus": 4,
            "sbatch_memory": "8G",
            "sbatch_time": "0-12:00:00",
        },
    }

    flags = sbatch_init_flags(config)
    assert flags == ["-c 4", "-t 0-12:00:00", "--mem=8G", "--account=gratis"]


# ---- Python launcher ----


def test_get_python_command():
    # python_command overrides both backends; otherwise local uses the active
    # interpreter and slurm the default micromamba runner.
    from towbintools_pipeline.utils import get_python_command

    assert get_python_command({"backend": "local"}) == sys.executable
    assert get_python_command({"backend": "slurm"}) == (
        "~/.local/bin/micromamba run -n towbintools python3"
    )
    assert (
        get_python_command(
            {"backend": "slurm", "python_command": "conda run -n x python"}
        )
        == "conda run -n x python"
    )
    assert (
        get_python_command(
            {"backend": "local", "python_command": "conda run -n x python"}
        )
        == "conda run -n x python"
    )


# ---- Packaging / entry point ----


def test_init_pipeline_importable_without_side_effects():
    # The entry point requires the module to import without parsing args or
    # running anything; main() is the callable it points at.
    import towbintools_pipeline.init_pipeline as ip

    assert callable(ip.main)
    assert callable(ip.build_blocks_for_subdir)


# ---- CLI dispatch and scaffolding ----


def test_get_args_positional_and_flag_config():
    from towbintools_pipeline.init_pipeline import get_args

    assert get_args(["exp.yaml"]).config == "exp.yaml"
    assert get_args(["-c", "exp.yaml"]).config == "exp.yaml"
    # -c wins if both are given.
    assert get_args(["pos.yaml", "-c", "flag.yaml"]).config == "flag.yaml"


def test_get_args_requires_config():
    from towbintools_pipeline.init_pipeline import get_args

    with pytest.raises(SystemExit):
        get_args([])


def test_cli_dispatch_routes_to_run(monkeypatch):
    import towbintools_pipeline.cli as cli
    import towbintools_pipeline.init_pipeline as ip

    calls = []
    monkeypatch.setattr(ip, "main", lambda argv=None: calls.append(argv))

    cli.main(["run", "-c", "x.yaml"])
    cli.main(["config.yaml"])  # no subcommand -> run (back-compat)
    assert calls == [["-c", "x.yaml"], ["config.yaml"]]


def test_cli_init_config_writes_configs(tmp_path):
    import towbintools_pipeline.cli as cli

    cli.main(["init-configs", str(tmp_path)])
    for name in ("config.yaml", "slurm_config.yaml"):
        assert (tmp_path / name).is_file()


def test_cli_init_config_non_destructive(tmp_path):
    import towbintools_pipeline.cli as cli

    sentinel = tmp_path / "config.yaml"
    sentinel.write_text("keep me")
    cli.main(["init-configs", str(tmp_path)])
    assert sentinel.read_text() == "keep me"  # existing file preserved
    cli.main(["init-configs", str(tmp_path), "--force"])
    assert sentinel.read_text() != "keep me"  # --force overwrites


# ---- Run directory, backup, cleanup ----


def test_run_config_and_version_info_backed_up(tmp_path):
    # The config and a git_info.txt land in the backup, which sits beside the
    # report (under the analysis dir), not inside it.
    config_path = _build_experiment(tmp_path)

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    backups = list(
        (tmp_path / "exp" / "analysis" / "pipeline_backup").glob("pipeline_*")
    )
    assert len(backups) == 1
    assert (backups[0] / "config.yaml").exists()
    assert (backups[0] / "git_info.txt").exists()


def test_repeated_runs_get_separate_backups(tmp_path):
    # Two runs of the same experiment must not overwrite each other's backup;
    # each gets its own run directory (slurm keys it on the job id instead).
    config_path = _build_experiment(tmp_path)

    for _ in range(2):
        result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "t")])
        assert (
            result.returncode == 0
        ), f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    backups = list(
        (tmp_path / "exp" / "analysis" / "pipeline_backup").glob("pipeline_*")
    )
    assert len(backups) == 2


def test_cleanup_on_success_removes_temp_dir(tmp_path):
    # With cleanup_on_success, a finished run deletes its own temp dir; the
    # backup and the outputs survive.
    config_path = _build_experiment(tmp_path, extra_config={"cleanup_on_success": True})

    result = _run_pipeline(config_path, cwd=str(tmp_path))
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    # the run's temp subdir is gone
    assert list((tmp_path / "temp_files").glob("pipeline_*")) == []
    # the durable backup remains
    backups = list(
        (tmp_path / "exp" / "analysis" / "pipeline_backup").glob("pipeline_*")
    )
    assert len(backups) == 1
    # the outputs remain
    morph_csv = tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()


# ---- Log concatenation ----


def test_concatenate_sbatch_logs(tmp_path):
    # Per-block logs are joined per stream in write order, headed by their file
    # name, without touching the originals.
    import os as _os

    from towbintools_pipeline.utils import concatenate_sbatch_logs

    temp_dir = tmp_path / "pipeline_4242"
    log_dir = temp_dir / "sbatch_output"
    log_dir.mkdir(parents=True)

    # Written out of order on purpose; mtimes decide the order, not the names.
    (log_dir / "straightening-3.out").write_text("third\n")
    (log_dir / "init-1.out").write_text("first\n")
    (log_dir / "segmentation-2.out").write_text("second\n")
    (log_dir / "segmentation-2.err").write_text("a warning\n")
    for name, stamp in (
        ("init-1.out", 1000),
        ("segmentation-2.out", 2000),
        ("straightening-3.out", 3000),
    ):
        _os.utime(log_dir / name, (stamp, stamp))

    concatenate_sbatch_logs(
        str(temp_dir), "PIPELINE FINISHED -- all 3 blocks completed"
    )

    combined = (temp_dir / "pipeline-4242.out").read_text()
    assert combined.index("first") < combined.index("second") < combined.index("third")
    assert "===== init-1.out =====" in combined
    # the last line says whether the run got to the end
    assert combined.rstrip().endswith(
        "PIPELINE FINISHED -- all 3 blocks completed ====="
    )
    # each stream gets its own file, and the originals stay put
    assert (temp_dir / "pipeline-4242.err").read_text().count("a warning") == 1
    assert (log_dir / "init-1.out").read_text() == "first\n"

    # Rebuilding is idempotent: the combined file lives outside the log dir, so
    # it is never folded into itself.
    concatenate_sbatch_logs(
        str(temp_dir), "PIPELINE FINISHED -- all 3 blocks completed"
    )
    assert (temp_dir / "pipeline-4242.out").read_text() == combined


def test_concatenate_sbatch_logs_without_logs(tmp_path):
    # A local run has no sbatch logs at all; that must be a quiet no-op.
    from towbintools_pipeline.utils import concatenate_sbatch_logs

    concatenate_sbatch_logs(str(tmp_path))  # no sbatch_output dir
    (tmp_path / "sbatch_output").mkdir()
    concatenate_sbatch_logs(str(tmp_path))  # empty sbatch_output dir
    assert list(tmp_path.glob("pipeline-*")) == []


# ---- End-to-end local pipeline ----


def test_local_pipeline_segmentation_and_morphology(tmp_path):
    config_path = _build_experiment(tmp_path)
    temp_dir = tmp_path / "pipeline_temp"

    result = _run_pipeline(config_path, ["--temp_dir", str(temp_dir)])
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    # segmentation produced one mask per input image
    mask_dir = tmp_path / "exp" / "analysis" / "ch1_seg"
    masks = sorted(mask_dir.glob("*.tiff"))
    assert len(masks) == 2

    # morphology produced one record per image with a positive area
    morph_csv = tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()
    with open(morph_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(float(r["ch1_seg_area"]) > 0 for r in rows)


def test_find_existing_filemap_any_extension_newest_wins(tmp_path):
    from towbintools_pipeline.init_pipeline import find_existing_filemap

    assert find_existing_filemap(str(tmp_path), "analysis_filemap") is None

    parquet_path = tmp_path / "analysis_filemap.parquet"
    parquet_path.write_text("")
    assert find_existing_filemap(str(tmp_path), "analysis_filemap") == str(parquet_path)

    # With both extensions present, the most recent file is picked.
    csv_path = tmp_path / "analysis_filemap.csv"
    csv_path.write_text("")
    os.utime(parquet_path, (1_000, 1_000))
    os.utime(csv_path, (2_000, 2_000))
    assert find_existing_filemap(str(tmp_path), "analysis_filemap") == str(csv_path)
    os.utime(parquet_path, (3_000, 3_000))
    assert find_existing_filemap(str(tmp_path), "analysis_filemap") == str(parquet_path)


def _run_segmentation_as_parquet_then_full_as_csv(tmp_path, before_second_run=None):
    # First run: segmentation only, parquet filemap. Second run: segmentation +
    # morphology with report_format csv. Returns the report dir.
    config_path = _build_experiment(
        tmp_path,
        extra_config={"report_format": "parquet", "building_blocks": ["segmentation"]},
    )
    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "t1")])
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    report_dir = tmp_path / "exp" / "analysis" / "report"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["report_format"] = "csv"
    config["building_blocks"] = ["segmentation", "morphology_computation"]
    if before_second_run is not None:
        before_second_run(report_dir, config)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "t2")])
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"
    return report_dir


def test_existing_filemap_in_other_format_is_converted(tmp_path):
    # A parquet filemap from an earlier run is picked up by a csv run, which
    # converts it and writes its results to the csv; the parquet is untouched.
    import polars as pl

    parquet_bytes = {}

    def snapshot(report_dir, config):
        parquet_bytes["base"] = (report_dir / "analysis_filemap.parquet").read_bytes()

    report_dir = _run_segmentation_as_parquet_then_full_as_csv(tmp_path, snapshot)

    assert (report_dir / "analysis_filemap.parquet").read_bytes() == parquet_bytes[
        "base"
    ]
    csv_filemap = pl.read_csv(report_dir / "analysis_filemap.csv")
    assert "analysis/ch1_seg" in csv_filemap.columns  # carried over from the parquet
    assert "ch1_seg_area" in csv_filemap.columns  # added by this run


def test_existing_annotated_filemap_in_other_format_is_converted(tmp_path):
    # With overwrite_annotated_filemap, a parquet annotated filemap is converted
    # to csv and the run's results land there; the parquet files are untouched.
    import polars as pl

    parquet_bytes = {}

    def annotate(report_dir, config):
        base = pl.read_parquet(report_dir / "analysis_filemap.parquet")
        base.with_columns(pl.lit(7).alias("M1")).write_parquet(
            report_dir / "analysis_filemap_annotated.parquet"
        )
        for name in ("analysis_filemap", "analysis_filemap_annotated"):
            parquet_bytes[name] = (report_dir / f"{name}.parquet").read_bytes()
        config["overwrite_annotated_filemap"] = True

    report_dir = _run_segmentation_as_parquet_then_full_as_csv(tmp_path, annotate)

    for name, content in parquet_bytes.items():
        assert (report_dir / f"{name}.parquet").read_bytes() == content
    annotated = pl.read_csv(report_dir / "analysis_filemap_annotated.csv")
    assert "M1" in annotated.columns  # the annotation survived the conversion
    assert "ch1_seg_area" in annotated.columns  # added by this run
    assert not (report_dir / "analysis_filemap.csv").exists()


def test_local_pipeline_default_temp_dir_in_cwd(tmp_path):
    # Without --temp_dir (and no temp_dir config key), temp files default to
    # ./temp_files in the working directory (matching the sbatch launcher).
    config_path = _build_experiment(tmp_path)

    result = _run_pipeline(config_path, cwd=str(tmp_path))
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    # each run gets its own subdirectory, named after its start time locally
    runs = list((tmp_path / "temp_files").glob("pipeline_*"))
    assert len(runs) == 1
    assert (runs[0] / "pickles").is_dir()
    # the slurm-only folders are not created for a local run
    assert not (runs[0] / "batch").exists()
    assert not (runs[0] / "sbatch_output").exists()
    # the run still produced its outputs under the experiment dir
    morph_csv = tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()


def test_block_progress_is_logged(tmp_path):
    # Every block announces its start and its completion, numbered within the
    # planned sequence, and the run ends with the closing line.
    config_path = _build_experiment(tmp_path)

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    markers = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("###") or line.startswith("  1/") or line.startswith("  2/")
    ]
    assert markers == [
        "### Initializing the pipeline for subdir None ###",
        "### Pipeline plan: 2 blocks ###",
        "1/2 segmentation",
        "2/2 morphology_computation",
        "### Starting block 1/2 segmentation ###",
        "### Finished block 1/2 segmentation ###",
        "### Starting block 2/2 morphology_computation ###",
        "### Finished block 2/2 morphology_computation ###",
        "### End of the pipeline! (2 blocks) ###",
    ]


def test_custom_analysis_dir_name(tmp_path):
    # A non-default analysis_dir_name flows through: outputs land under it and
    # the downstream morphology block resolves its mask input correctly.
    config_path = _build_experiment(tmp_path, analysis_dir_name="results")

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    masks = sorted((tmp_path / "exp" / "results" / "ch1_seg").glob("*.tiff"))
    assert len(masks) == 2
    morph_csv = tmp_path / "exp" / "results" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()
    with open(morph_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(float(r["ch1_seg_area"]) > 0 for r in rows)


def test_prefix_free_folder_refs(tmp_path):
    # A mask ref written WITHOUT the analysis-dir prefix resolves to the right
    # column and the run completes.
    config_path = _build_experiment(
        tmp_path, extra_config={"morphology_computation_masks": ["ch1_seg"]}
    )

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"
    assert (
        tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    ).exists()


def test_prefix_free_refs_survive_analysis_dir_rename(tmp_path):
    # The same prefix-free ref works under a renamed analysis_dir_name, with no
    # ref to rewrite -- the coupling this PR removes.
    config_path = _build_experiment(
        tmp_path,
        analysis_dir_name="output",
        extra_config={"morphology_computation_masks": ["ch1_seg"]},
    )

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"
    assert (tmp_path / "exp" / "output" / "report" / "ch1_seg_morphology.csv").exists()


def test_experiment_dir_cli_overrides_config(tmp_path):
    # The config records a wrong experiment_dir; --experiment_dir points at the
    # real data and must win, so the run finds the images and produces outputs.
    config_path = _build_experiment(
        tmp_path, config_experiment_dir=str(tmp_path / "wrong")
    )

    result = _run_pipeline(
        config_path,
        [
            "--experiment_dir",
            str(tmp_path / "exp"),
            "--temp_dir",
            str(tmp_path / "pipeline_temp"),
        ],
    )
    assert result.returncode == 0, f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    morph_csv = tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()


def test_custom_block_with_no_return_type_runs_and_chains(tmp_path):
    # A custom block that produces nothing the filemap knows about still has to run
    # and still has to hand over to the next block.
    script = tmp_path / "sentinel.py"
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('-f', '--filemap')\n"
        "parser.add_argument('-c', '--config')\n"
        "parser.add_argument('-b', '--block_config')\n"
        "parser.add_argument('-o', '--output')\n"
        "parser.add_argument('--sentinel')\n"
        "args = parser.parse_args()\n"
        "open(args.sentinel, 'w').write('ran')\n"
    )
    sentinel = tmp_path / "sentinel.txt"

    config_path = _build_experiment(
        tmp_path,
        extra_config={
            "building_blocks": ["custom", "segmentation"],
            "custom_script_path": [str(script)],
            "custom_script_name": ["sentinel"],
            "custom_script_return_type": [None],
            "custom_script_parameters": [[f"--sentinel {sentinel}"]],
        },
    )

    result = _run_pipeline(config_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text() == "ran"
    # The block after it still ran, so the chain was not broken.
    masks = tmp_path / "exp" / "analysis" / "ch1_seg"
    assert masks.is_dir() and any(masks.iterdir())
    # And nothing was added to the filemap for it.
    with open(tmp_path / "exp" / "analysis" / "report" / "analysis_filemap.csv") as f:
        header = f.readline()
    assert "sentinel" not in header
