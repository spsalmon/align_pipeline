"""Silence known-benign, high-volume third-party warnings so they don't clutter
the logs. Repo-maintained; NOT part of the user config surface. Add an entry
here when a benign warning is found to fire noisily (e.g. once per image).

Only Python `warnings` are reachable here; library-native logging (e.g.
xgboost's C++ logger, plain print()s) needs its own knob, not this list.
"""

import warnings

# (action, message-regex, category). action is "ignore" (drop) or "once" (show a
# single copy). message is matched from the start (re.match), so lead with ".*".
# Keep each entry commented with its source.
_RULES = [
    # torch DataLoader on CPU-only machines (no accelerator) -- benign
    (
        "ignore",
        r".*pin_memory.*argument is set as true but no accelerator.*",
        UserWarning,
    ),
    # huggingface_hub on filesystems without symlink support (e.g. Windows) -- benign
    ("ignore", r".*cache-system uses symlinks by default.*", UserWarning),
]


def configure_warnings():
    for action, message, category in _RULES:
        warnings.filterwarnings(action, message=message, category=category)
