# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

try:
    # Preferred: IsaacLab helper (handles nested packages and common ignore patterns)
    from isaaclab_tasks.utils import import_packages  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Fallback: keep this extension importable even if `isaaclab_tasks` isn't on PYTHONPATH.
    # This is useful for simple unit tests / static tooling / non-Isaac entrypoints.
    import importlib
    import pkgutil

    def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None) -> None:
        """Recursively import submodules under `package_name`, skipping blacklisted package fragments."""
        blacklist_pkgs = blacklist_pkgs or []
        pkg = importlib.import_module(package_name)
        if not hasattr(pkg, "__path__"):
            return
        prefix = pkg.__name__ + "."
        for modinfo in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
            name = modinfo.name
            if any(b in name for b in blacklist_pkgs):
                continue
            importlib.import_module(name)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
