import atexit
import os

from app_components.backend import open_filemap
from app_components.image_cache import default_min_available_bytes
from app_components.server import main_server
from app_components.ui import initialize_ui
from shiny import App

recompute_features_at_molt = os.environ.get("RECOMPUTE_FEATURES", "0") == "1"
open_annotated = os.environ.get("OPEN_ANNOTATED", "1") == "1"
filemap_path = os.environ.get("FILEMAP_PATH", "")
preload_images = os.environ.get("PRELOAD_IMAGES", "1") == "1"
preload_min_free_mb = os.environ.get("PRELOAD_MIN_FREE_MB", "")
preload_min_available_bytes = (
    int(float(preload_min_free_mb) * 1024**2)
    if preload_min_free_mb
    else default_min_available_bytes()
)

print("Opening the filemap ...")
filemap, filemap_save_path = open_filemap(
    filemap_path, open_annotated=open_annotated, lazy_loading=False
)
print("Creating the app ...")
(
    app_ui,
    filemap,
    raw_column,
    feature_columns,
    custom_columns_choices,
    points,
    times,
    default_plotted_column,
    n_channels,
) = initialize_ui(filemap, recompute_features_at_molt=recompute_features_at_molt)

_server_save_hook = {"fn": None}


def _save_on_exit():
    if _server_save_hook["fn"] is not None:
        try:
            _server_save_hook["fn"]()
        except Exception as e:
            print(f"Save on exit failed: {e}")


atexit.register(_save_on_exit)


def server(input, output, session):
    save_fn = main_server(
        input,
        output,
        session,
        filemap=filemap,
        filemap_save_path=filemap_save_path,
        raw_column=raw_column,
        feature_columns=feature_columns,
        custom_columns_choices=custom_columns_choices,
        points=points,
        times=times,
        default_plotted_column=default_plotted_column,
        n_channels=n_channels,
        preload_images=preload_images,
        preload_min_available_bytes=preload_min_available_bytes,
    )
    _server_save_hook["fn"] = save_fn


app = App(app_ui, server)
