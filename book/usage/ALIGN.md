# The ALIGN package

Almost everything the pipeline does is actually implemented in
[ALIGN](https://pypi.org/project/align_toolbox/), the lab's Python library.
The pipeline mostly handles configuration, job scheduling and bookkeeping; the
image analysis itself (segmentation, straightening, feature computation, quality
control, molt detection, plotting…) lives in ALIGN.

This means you can use the same functions in your own scripts and notebooks,
without running the pipeline. It is already installed in the `align`
environment, and can be installed anywhere else with:

```bash
pip install align_toolbox
```

## What's inside

| Module | Content |
|---|---|
| `align_toolbox.foundation` | Image and filemap I/O (`image_handling`, `file_handling`), binary image utilities, worm features (volume, length, width…), molt detection, z-stack utilities |
| `align_toolbox.segmentation` | Classical and deep learning segmentation |
| `align_toolbox.straightening` | Worm straightening |
| `align_toolbox.quantification` | Fluorescence quantification |
| `align_toolbox.classification` | Quality control features and classifiers |
| `align_toolbox.deep_learning` | Model architectures, datasets, losses and augmentations used for training |
| `align_toolbox.data_analysis` | Time series processing (smoothing, growth rates, values at molts…) |
| `align_toolbox.plotting` | Plotting utilities (see [plotting](https://spsalmon.github.io/align_pipeline/usage/plotting/)) |

## Example

```python
from align_toolbox.foundation.file_handling import read_filemap
from align_toolbox.foundation.image_handling import read_tiff_file
from align_toolbox.foundation.worm_features import compute_mask_volume

filemap = read_filemap("/path/to/experiment/analysis/report/analysis_filemap.parquet")

mask = read_tiff_file(filemap["analysis/ch2_seg_str"][0])
volume = compute_mask_volume(mask, pixelsize=0.65)
```

The full API reference is available on
[Read the Docs](https://align-toolbox.readthedocs.io/en/latest/).
