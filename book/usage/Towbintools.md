# The towbintools package

Almost everything the pipeline does is actually implemented in
[towbintools](https://pypi.org/project/towbintools/), the lab's Python library.
The pipeline mostly handles configuration, job scheduling and bookkeeping; the
image analysis itself (segmentation, straightening, feature computation, quality
control, molt detection, plotting…) lives in towbintools.

This means you can use the same functions in your own scripts and notebooks,
without running the pipeline. It is already installed in the `towbintools`
environment, and can be installed anywhere else with:

```bash
pip install towbintools
```

## What's inside

| Module | Content |
|---|---|
| `towbintools.foundation` | Image and filemap I/O (`image_handling`, `file_handling`), binary image utilities, worm features (volume, length, width…), molt detection, z-stack utilities |
| `towbintools.segmentation` | Classical and deep learning segmentation |
| `towbintools.straightening` | Worm straightening |
| `towbintools.quantification` | Fluorescence quantification |
| `towbintools.classification` | Quality control features and classifiers |
| `towbintools.deep_learning` | Model architectures, datasets, losses and augmentations used for training |
| `towbintools.data_analysis` | Time series processing (smoothing, growth rates, values at molts…) |
| `towbintools.plotting` | Plotting utilities (see [plotting](https://spsalmon.github.io/towbintools_pipeline/usage/plotting/)) |

## Example

```python
from towbintools.foundation.file_handling import read_filemap
from towbintools.foundation.image_handling import read_tiff_file
from towbintools.foundation.worm_features import compute_mask_volume

filemap = read_filemap("/path/to/experiment/analysis/report/analysis_filemap.parquet")

mask = read_tiff_file(filemap["analysis/ch2_seg_str"][0])
volume = compute_mask_volume(mask, pixelsize=0.65)
```

The full API reference is available on
[Read the Docs](https://towbintools.readthedocs.io/en/latest/).
