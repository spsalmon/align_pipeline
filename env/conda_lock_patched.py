"""Thin wrapper around the conda-lock CLI that widens its manylinux tag list.

conda-lock does not query the local interpreter for wheel compatibility: it
builds a synthetic tag set for the target platform from the hard-coded
``conda_lock.pypi_solver.MANYLINUX_TAGS`` list, which only contains the tags
used by the official manylinux Docker images.  Wheels published with any other
glibc tag therefore resolve to zero candidates and the solve dies with::

    RuntimeError: Unable to find installation candidates for nvidia-nccl-cu13 (2.32.3)

nvidia ships ``nvidia-nccl-cu13`` (pulled in by xgboost >= 3.4) as
``manylinux_2_27`` wheels, which are perfectly installable here, so we add that
tag before handing over to the real CLI.  Add further tags to EXTRA_TAGS if new
dependencies hit the same wall; the list must stay sorted by glibc version.

Usage is identical to the ``conda-lock`` executable:

    python conda_lock_patched.py lock -f environment.yml -p linux-64 --micromamba
"""

import sys

from conda_lock import pypi_solver

EXTRA_TAGS = ["_2_27"]


def _widen_manylinux_tags() -> None:
    tags = list(pypi_solver.MANYLINUX_TAGS)
    tags += [tag for tag in EXTRA_TAGS if tag not in tags]
    tags.sort(key=pypi_solver._glibc_version_from_manylinux_tag)
    pypi_solver.MANYLINUX_TAGS = tags


if __name__ == "__main__":
    _widen_manylinux_tags()

    from conda_lock.conda_lock import main

    sys.argv[0] = "conda-lock"
    main()
