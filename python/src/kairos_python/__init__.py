"""The kairos_python package: KAIROS flow pipeline + `kairos run` CLI.

The compiled bindings (_kairos_bindings) are self-contained: they bundle
Hydra's generic pybind helpers + base pipeline, so this package does not import
_hydra_bindings. Generic data-loading utilities are reused from hydra_python.
"""
from kairos_python._kairos_bindings import *  # noqa: F401,F403
