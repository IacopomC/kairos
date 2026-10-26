"""Command-line interface."""

import pathlib
from typing import Optional, Union

OptPath = Optional[Union[str, pathlib.Path]]


def resolve_output_path(output_path: OptPath, force: bool = False):
    """
    Resolve an output path.

    An existing output is kept, whatever `force` says: runs add per-scene
    subdirectories to it.

    Args:
        output_path: Optional path for the run to output to
        force: Accepted for the CLI's --force flag; nothing is removed
    """
    if output_path is None:
        return None

    return pathlib.Path(output_path).expanduser().absolute()
