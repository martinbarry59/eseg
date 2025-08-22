"""Global configuration paths for datasets, results, and checkpoints.

These paths are resolved relative to the package directory so that the
project can be installed as a package while still using a conventional
folder layout in a local repository.

If you package this for PyPI and want to avoid writing outside the
installed package directory, consider overriding these via environment
variables or a user configuration directory (e.g. using `platformdirs`).
"""

import os
from typing import Final

package_dir: Final[str] = os.path.dirname(os.path.abspath(__file__))
parent_dir: Final[str] = os.path.dirname(package_dir)

# Root folders (created lazily if missing)
data_path: Final[str] = os.path.join(parent_dir, "dataset/")

results_path: Final[str] = os.path.join(parent_dir, "results/")
if not os.path.exists(results_path):  # side-effect okay for research workflow
    os.makedirs(results_path, exist_ok=True)

checkpoint_path: Final[str] = os.path.join(parent_dir, "checkpoints/")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
