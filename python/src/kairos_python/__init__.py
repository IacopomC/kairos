"""The kairos_python package: KAIROS flow pipeline + `kairos run` CLI.

The pipeline is Hydra's (hydra_python); the compiled bindings (_kairos_bindings)
add the flow-module handle and the bag-of-words producer. Importing them also
registers the KAIROS modules, sinks and functors with Hydra's factories.
"""
import hydra_python  # noqa: F401 - Hydra's bindings first, so its types are known
from kairos_python._kairos_bindings import *  # noqa: F401,F403
from kairos_python import bow_producer  # noqa: F401 - registers the "bow_producer" callback
from kairos_python.data_loader import DataLoader, DataLoaderIter, get_dataloader  # noqa: F401
