# gui/app_components/image_cache.py
import base64
import io
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait

import matplotlib
import numpy as np
from align_toolbox.foundation import image_handling
from PIL import Image as PILImage


class PointImageCache:
    """Thread-safe store of downsampled float32 channel arrays, keyed by (time_index, channel_idx)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[tuple[int, int], np.ndarray] = {}
        self._point: int | None = None

    def reset(self, point: int) -> None:
        with self._lock:
            self._data.clear()
            self._point = point

    def put(
        self, point: int, time_index: int, channel_idx: int, array: np.ndarray
    ) -> None:
        with self._lock:
            if self._point is None or point != self._point:
                return
            self._data[(time_index, channel_idx)] = array

    def get(self, time_index: int, channel_idx: int) -> np.ndarray | None:
        with self._lock:
            return self._data.get((time_index, channel_idx))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ProgressTracker:
    """Thread-safe counter for background loader progress."""

    def __init__(self):
        self._lock = threading.Lock()
        self._completed = 0
        self._total = 0

    def reset(self, total: int) -> None:
        with self._lock:
            self._completed = 0
            self._total = total

    def increment(self) -> None:
        with self._lock:
            self._completed += 1

    def get(self) -> tuple[int, int]:
        with self._lock:
            return self._completed, self._total


MAX_DISPLAY_PX = 768

COLORMAPS: dict[str, np.ndarray] = {
    name: (
        matplotlib.colormaps.get_cmap(name)(np.linspace(0, 1, 256))[:, :3] * 255
    ).astype(np.uint8)
    for name in ("viridis", "magma", "autumn")
}


def downsample(img: np.ndarray) -> np.ndarray:
    """Stride-based downsample so the spatial dimensions fit within MAX_DISPLAY_PX."""
    step = max(1, math.ceil(max(img.shape[-2:]) / MAX_DISPLAY_PX))
    return img[..., ::step, ::step]


def apply_lut(img: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Convert a float32 [0, 1] 2-D array to an RGB uint8 array using a pre-built LUT."""
    lut = COLORMAPS[cmap_name]
    u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return lut[u8]


def alpha_composite(
    base: np.ndarray, overlay: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend two RGB uint8 arrays: result = base*(1-alpha) + overlay*alpha."""
    return np.clip(
        base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def composite_mask(
    base: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "autumn",
) -> np.ndarray:
    """Overlay a segmentation mask (non-zero pixels only) onto an RGB base image."""
    mask_f = mask.astype(np.float32)
    if mask_f.max() > 0:
        mask_f = mask_f / mask_f.max()
    overlay = apply_lut(mask_f, cmap_name)
    visible = mask > 0
    result = base.copy()
    result[visible] = np.clip(
        base[visible].astype(np.float32) * (1.0 - alpha)
        + overlay[visible].astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    return result


def extract_channel(img: np.ndarray, channel: int) -> np.ndarray:
    """Extract a single spatial channel from a 2-D, 3-D, or 4-D array."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img[channel]
    if img.ndim == 4:
        return img[img.shape[0] // 2, channel]
    raise ValueError(f"Unexpected image dimensions: {img.ndim}")


def prepare_channel(channel_img: np.ndarray) -> np.ndarray:
    """Normalize a single channel to float32 [0, 1] and downsample for display."""
    normalized = image_handling.normalize_image(channel_img, dest_dtype=np.float32)
    return downsample(normalized)


def compose_display_image(
    main_arr: np.ndarray,
    main_cmap: str = "viridis",
    overlay_arr: np.ndarray | None = None,
    overlay_cmap: str = "magma",
    overlay_alpha: float = 0.5,
    seg_arr: np.ndarray | None = None,
    seg_alpha: float = 0.5,
) -> np.ndarray:
    """Compose main channel + optional overlay channel + optional segmentation mask into RGB uint8."""
    rgb = apply_lut(main_arr, main_cmap)
    if overlay_arr is not None:
        rgb = alpha_composite(rgb, apply_lut(overlay_arr, overlay_cmap), overlay_alpha)
    if seg_arr is not None:
        rgb = composite_mask(rgb, seg_arr, seg_alpha, "autumn")
    return rgb


def array_to_data_url(rgb: np.ndarray, quality: int = 85) -> str:
    """Encode an RGB uint8 array as a base64 JPEG data URL."""
    buf = io.BytesIO()
    PILImage.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class BackgroundLoader:
    """Reads all TIFFs for a point in background threads and populates a PointImageCache."""

    def __init__(
        self,
        point: int,
        image_paths: list[str],
        n_channels: int,
        cache: PointImageCache,
        progress_tracker: ProgressTracker,
        n_workers: int = 4,
    ):
        self._cancel = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._futures = [
            self._executor.submit(
                self._process,
                point,
                time_index,
                path,
                n_channels,
                cache,
                progress_tracker,
            )
            for time_index, path in enumerate(image_paths)
        ]
        self._executor.shutdown(wait=False)

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self) -> None:
        """Block until all submitted futures are done. Used in tests."""
        futures_wait(self._futures)

    def _process(
        self,
        point: int,
        time_index: int,
        path: str,
        n_channels: int,
        cache: PointImageCache,
        progress_tracker: ProgressTracker,
    ) -> None:
        if self._cancel.is_set():
            return
        try:
            img = image_handling.read_tiff_file(path)
        except Exception:
            return
        if self._cancel.is_set():
            return
        for ch in range(n_channels):
            if self._cancel.is_set():
                return
            try:
                channel_img = extract_channel(img, ch)
                cache.put(point, time_index, ch, prepare_channel(channel_img))
            except Exception as e:
                print(f"BackgroundLoader: failed ch={ch} t={time_index}: {e!r}")
                continue
        progress_tracker.increment()
