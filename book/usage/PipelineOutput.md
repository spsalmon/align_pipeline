# Pipeline Output

Everything the pipeline produces goes into the analysis directory of your
experiment (`analysis/` by default, set by `analysis_dir_name`):

```
my_experiment/
├── raw/                          <- your images
└── analysis/
    ├── ch2_seg/                  <- images produced by a block
    ├── ch2_seg_str/
    ├── report/                   <- the results
    │   ├── analysis_filemap.parquet
    │   └── ch2_seg_str_morphology.parquet
    └── pipeline_backup/          <- what was run, and how
        └── pipeline_1234567/
```

## Images

Any block that produces images (segmentation masks, straightened images, etc.)
saves them as OME-TIFF files in a subdirectory of the analysis directory. The name
of that subdirectory is predictable, which is what lets you feed the output of one
block into the next one — see each block's page.

## Reports

Any block that produces measurements (`morphology_computation`,
`fluorescence_quantification`, etc.) writes them into a single report file in
`analysis/report/`. If a block produces many measurements per image (a value at
every plane of a Z-stack, for instance), it writes one report file per image into
a subdirectory instead.

Report files are either CSV or Parquet, depending on `report_format`. Parquet
files are much smaller than CSVs (useful for big experiments) but less convenient
to edit (in Microsoft Excel, for example).

## The analysis filemap

`analysis/report/analysis_filemap.<csv|parquet>` is the table that ties everything
together: one row per image (a position at a time point), and one column per thing
the pipeline knows about it — the path to the raw image, the path to each mask,
each measured feature, the quality control result, the detected molts. Every block
adds its output to it as new columns.

This is the file you open for downstream analysis, and the file the
[GUI](https://spsalmon.github.io/towbintools_pipeline/usage/usinggui/) annotates.

When a run starts, an existing filemap is reused whatever its format (a new
filemap is only built when none exists). If both a `.csv` and a `.parquet` are
present, the most recent one is used. The run always writes to the filemap in the
configured `report_format`: if the one found is in the other format, it is
converted (e.g. `analysis_filemap.parquet` is copied to `analysis_filemap.csv`
when `report_format` is `csv`) and the original is left untouched. The same
applies to `analysis_filemap_annotated` when `overwrite_annotated_filemap` is set.

## Provenance: what was actually run

Each run copies into `analysis/pipeline_backup/pipeline_<id>/`:

- the configuration file(s) used;
- `git_info.txt` — the exact version of the pipeline and of the packages it ran
  with;
- the complete logs of the run.

You can therefore always go back to an old analysis and know exactly how it was
produced. This folder sits **beside** `report/`, not inside it: `report/` holds
results, the backup holds the record.
