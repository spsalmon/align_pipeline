"""Console entry point: dispatch `align <subcommand>` to the
right handler. Handlers lazy-import their module so a plain `run` does not pay
for another subcommand's dependencies. With no known subcommand the arguments
are treated as a `run`, so the previous `align -c config.yaml`
form keeps working.
"""

import argparse
import os
import shutil
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_DIR = os.path.join(_PIPELINE_DIR, "defaults", "configs")
_CONFIG_FILES = ("config.yaml", "slurm_config.yaml")


def _run(argv):
    # Run the pipeline. Delegates to the runner, which accepts a positional
    # config or -c plus -e/-t.
    from align_pipeline.init_pipeline import main as run_main

    run_main(argv)


def _init_config(argv):
    # Copy the bundled default config files into a directory, so a user can start
    # from them without digging into the installed package. Configs only (the
    # bundled models are large); non-destructive unless --force.
    parser = argparse.ArgumentParser(
        prog="align init-configs",
        description="Copy the bundled default config files into a directory.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite files that already exist"
    )
    args = parser.parse_args(argv)

    os.makedirs(args.directory, exist_ok=True)
    skipped = []
    for name in _CONFIG_FILES:
        destination = os.path.join(args.directory, name)
        if os.path.exists(destination) and not args.force:
            skipped.append(name)
            continue
        shutil.copyfile(os.path.join(_DEFAULT_CONFIG_DIR, name), destination)
        print(f"Wrote {destination}")
    if skipped:
        print(
            "Skipped (already exist, pass --force to overwrite): " + ", ".join(skipped)
        )
    print("Bundled models are at: " + os.path.join(_PIPELINE_DIR, "defaults", "models"))


SUBCOMMANDS = {
    "run": _run,
    "init-configs": _init_config,
}


def _print_help():
    print("Usage: align <command> [options]\n")
    print("Commands:")
    print("  run           Run the pipeline (positional config or -c/--config)")
    print("  init-configs  Copy the bundled default config files into a directory")
    print("\nWith no command, the arguments are passed to `run`.")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        return SUBCOMMANDS[argv[0]](argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return
    # No known subcommand: treat the whole argument list as a `run` (back-compat).
    return _run(argv)


if __name__ == "__main__":
    main()
