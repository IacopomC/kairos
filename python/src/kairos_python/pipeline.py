# Copyright 2022, Massachusetts Institute of Technology.
# All Rights Reserved
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Research was sponsored by the United States Air Force Research Laboratory and
# the United States Air Force Artificial Intelligence Accelerator and was
# accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
# and conclusions contained in this document are those of the authors and should
# not be interpreted as representing the official policies, either expressed or
# implied, of the United States Air Force or the U.S. Government. The U.S.
# Government is authorized to reproduce and distribute reprints for Government
# purposes notwithstanding any copyright notation herein.
#
#
"""Module containing python utilites for loading a valid config."""
import logging
import pathlib
import pprint

import yaml
from hydra_python import HydraPipeline, set_glog_dir

from kairos_python._kairos_bindings import FlowModule, dsg_stats, mesh_snapshot, nav_snapshot


class KairosPipeline:
    """Hydra's pipeline with the Kairos flow module running inside it.

    Hydra runs the flow module (configured under `modules`), so this pairs the
    Hydra pipeline (stepping, saving, the scene graph) with a handle on the flow
    module (detections, flow state and queries). Hydra methods and flow methods
    are both available on this object.
    """

    def __init__(self, pipeline, output_path=None, flow_module="flow_temporal"):
        self._pipeline = pipeline
        self._output_path = output_path
        self._flow = FlowModule(flow_module)
        # In step mode this starts only the background modules (the flow
        # worker, loop closure); step() drives the rest.
        self._pipeline.start()

    @property
    def flow(self):
        """Handle on the flow module."""
        return self._flow

    @property
    def graph(self):
        """Copy of the current backend scene graph."""
        return self._pipeline.graph

    def save(self, output_path=None):
        """Save the pipeline state (graph, flow state, timing) to the output path."""
        path = output_path if output_path is not None else self._output_path
        if path is None:
            raise ValueError("no output path to save to")

        self._pipeline.save(pathlib.Path(path))

    def nav_snapshot(self):
        """Places layer as primitives (positions, edges, flow summaries)."""
        return nav_snapshot(self._pipeline.graph.to_binary(include_mesh=False))

    def mesh_snapshot(self):
        """Reconstruction mesh as flat vertex/face arrays."""
        return mesh_snapshot(self._pipeline.graph.to_binary(include_mesh=True))

    def dsg_stats(self):
        """Per-layer active/archived node counts and mesh size."""
        return dsg_stats(self._pipeline.graph.to_binary(include_mesh=True))

    def dump_dsg(self, path, include_mesh=True):
        """Save the current backend scene graph to `path`."""
        self._pipeline.save_graph(str(path), include_mesh)

    def __getattr__(self, name):
        # flow methods first, then the Hydra pipeline's
        flow = self.__dict__.get("_flow")
        if flow is not None and hasattr(flow, name):
            return getattr(flow, name)

        return getattr(self.__dict__["_pipeline"], name)


def get_config_path(dataset=None):
    """Return config directory for this package."""
    package_path = pathlib.Path(__file__).absolute().parent
    install_path = package_path / "config"
    if install_path.exists():
        return install_path

    devel_path = package_path.parent.parent.parent / "config"
    if devel_path.exists():
        return devel_path

    logging.error("failed to find config directory of package!")
    raise RuntimeError("invalid config path")


def flow_config(contents, name="flow_temporal"):
    """Config of the flow module in a resolved config tree (empty if absent)."""
    return ((contents or {}).get("modules", {}) or {}).get(name, {}) or {}


def update_nested(contents, other):
    """Merge two config trees recursively."""
    for key, value in other.items():
        if key not in contents:
            contents[key] = {}

        if isinstance(value, dict):
            update_nested(contents[key], value)
        else:
            contents[key] = value


def load_configs(
    dataset: str,
    labelspace: str = "ade20k_outdoor",
    bounding_box_type: str = "AABB",
    lcd_config: str = None,
    enable_lcd: bool = True,
):
    """
    Load various configs to construct the Hydra pipeline.

    dataset_name: Dataset name to load config from
    labelspace_name: Labelspace name to use
    bounding_box_type: Type of bounding box to use
    lcd_config: Name of the LCD config under config/lcd/<name>.yaml. When
        None (default), auto-selects: if config/lcd/<dataset>.yaml exists
        we use it (so `-c tbd` picks `lcd/tbd.yaml`, matching the
        uhumans2 / uhumans2_gnn pattern); otherwise falls back to "default".
        Loaded only when enable_lcd is True and the resolved file exists.
    enable_lcd: Whether the loop-closure module should be constructed and
        run. When True, lcd_use_bow_vectors is left at its C++ default (true)
        so PipelineQueues::bow_queue is allocated and the offline BoW producer
        (kairos_python.bow_producer) can feed agent-layer LCD;
        enable_agent_registration is forced True so the agent-level solver
        configured under lcd.agent_registration (BowAgentSolver, substituting
        for the ROS DsgAgentSolver) is used. When False, no LCD config is merged and the
        baseline `enable_lcd: false` default applies, leaving the dataset
        YAML in control of any non-LCD lcd_* fields.

    Returns:
        (Optional[PythonConfig]) Pipline config or none if invalid
    """
    config_path = get_config_path()
    dataset_path = config_path / "datasets" / f"{dataset}.yaml"
    labelspace_path = config_path / "label_spaces" / f"{labelspace}_label_space.yaml"
    if not dataset_path.exists():
        logging.error(f"invalid dataset path: {dataset_path}")
        return None

    if not labelspace_path.exists():
        logging.error(f"invalid labelspace path: {labelspace_path}")
        return None

    contents = {}
    with dataset_path.open("r") as fin:
        update_nested(contents, yaml.safe_load(fin.read()))

    with labelspace_path.open("r") as fin:
        update_nested(contents, yaml.safe_load(fin.read()))

    enable_lcd = bool(enable_lcd)

    # Merge the LCD config so LoopClosureConfig has the detector / search /
    # registration sub-blocks available. Must happen BEFORE KairosPipeline
    # construction: PipelineQueues only allocates `lcd_queue` when
    # GlobalInfo.config.enable_lcd was true at first dereference, and the
    # frontend's LcdInput producer is registered based on that queue's
    # existence. Setting enable_lcd later is a silent no-op.
    if enable_lcd:
        # Auto-select dataset-specific LCD config when caller didn't pin one.
        if lcd_config is None:
            candidate = config_path / "lcd" / f"{dataset}.yaml"
            lcd_config = dataset if candidate.exists() else "default"
            logging.info(
                "[hydra_python] auto-selected lcd config '%s' for dataset '%s'",
                lcd_config,
                dataset,
            )
        lcd_path = config_path / "lcd" / f"{lcd_config}.yaml"
        if lcd_path.exists():
            with lcd_path.open("r") as fin:
                update_nested(contents, yaml.safe_load(fin.read()))

    overrides = {
        "frontend": {
            "type": "GraphBuilder",
            "objects": {"bounding_box_type": bounding_box_type},
        },
        "backend": {"type": "BackendModule"},
        "enable_lcd": bool(enable_lcd),
    }
    if enable_lcd:
        overrides["lcd"] = {
            # The agent-level solver (lcd.agent_registration) runs only with
            # this flag set, so agent-LCD short-circuits the cascade before
            # places-LCD runs. lcd_use_bow_vectors is left at the C++ default
            # (true) so bow_queue is allocated for the offline DBoW producer in
            # kairos_python.bow_producer.
            "enable_agent_registration": True,
        }
    update_nested(contents, overrides)
    return contents


def viz_params_from_config(contents):
    """Pull the viz-relevant spatial params out of a resolved config tree so the
    recorder inherits them from the run rather than restating them.

    Both are genuine pipeline parameters (single source of truth = the dataset
    yaml): the flow-voxel edge (`active_window.volumetric_map.voxel_size`) sizes
    the ground cells, and the robot's sensing window
    (`map_window.max_radius_m`) is the sphere the viz draws. Falls back to the
    historical defaults if a config omits either.
    """
    aw = (contents or {}).get("active_window", {}).get("volumetric_map", {})
    mw = (contents or {}).get("map_window", {})
    return {
        "voxel_size": float(aw.get("voxel_size", 0.4)),
        "active_window_radius": float(mw.get("max_radius_m", 5.0)),
    }


def archetype_params_from_config(contents):
    """Pull the archetype classifier thresholds out of a resolved config tree.

    Single source of truth is the dataset yaml's
    `backend.update_functors.archetypes` block (the same one the C++
    UpdateArchetypesFunctor reads). The offline / ATC path feeds these to
    `classify_archetype` so it applies the exact rule and parameters the backend
    does. Falls back to the C++ struct defaults if a config omits the block.
    """
    arch = _archetype_block(contents)
    return {
        "dominant_weight_threshold": float(arch.get("dominant_weight_threshold", 0.20)),
        "min_flow_support": int(arch.get("min_flow_support", 1)),
        "min_probability": float(arch.get("min_probability", 0.05)),
    }


def archetype_horizon_from_config(contents):
    """Pull the archetype classification horizon out of a resolved config tree.

    A place stores an occupancy and no probability, so the horizon is what fixes
    the meaning of the `min_probability` gate. It is read from the same
    `backend.update_functors.archetypes` block as the thresholds, so a reader
    computing the probability itself (`place_summaries`) gates at the horizon the
    backend functor gates at, rather than at whichever horizon its caller
    defaults to.
    """
    return float(_archetype_block(contents).get("presence_horizon_s", 0.0))


def _archetype_block(contents):
    return (((contents or {}).get("backend", {})
             .get("update_functors", {})).get("archetypes", {}))


def load_pipeline(
    data,
    config_name,
    labelspace,
    output_path=None,
    config_verbosity=0,
    place_feature_strategy=None,
    zmq_url=None,
    enable_lcd=True,
    config_overrides=None,
):
    """Load Hydra pipeline from configs.

    `config_overrides` is an optional nested dict merged into the resolved config
    (e.g. {"flow_temporal": {"place_summary_max_nodes": 0}}), for callers that
    need to tweak a field without editing the dataset yaml. A `flow_temporal`
    entry applies to the flow module (`modules.flow_temporal`).
    """
    contents = load_configs(
        config_name, labelspace=labelspace, enable_lcd=enable_lcd
    )
    if not contents:
        return None

    if config_overrides:
        overrides = dict(config_overrides)
        flow_overrides = overrides.pop("flow_temporal", None)
        if flow_overrides:
            update_nested(overrides, {"modules": {"flow_temporal": flow_overrides}})

        update_nested(contents, overrides)

    if output_path:
        output_path = pathlib.Path(output_path)
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir(parents=True)

        set_glog_dir(glog_dir)

    if place_feature_strategy is not None:
        update_nested(
            contents,
            {
                "frontend": {
                    "view_database": {"view_selection_method": place_feature_strategy}
                }
            },
        )

    logging.debug(pprint.pformat(contents, sort_dicts=False))
    # keep key order: update functors are applied in the order they are listed
    pipeline = HydraPipeline.from_config(
        yaml.safe_dump(contents, sort_keys=False),
        data.sensor,
        robot_id=0,
        config_verbosity=config_verbosity,
        zmq_url="" if zmq_url is None else zmq_url,
    )

    return KairosPipeline(pipeline, output_path=output_path)
