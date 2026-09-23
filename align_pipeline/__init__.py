from align_pipeline.warnings_filter import configure_warnings

# Applied on import so every process that loads the package (workers,
# init_pipeline, tests) filters the known-benign noisy warnings.
configure_warnings()
