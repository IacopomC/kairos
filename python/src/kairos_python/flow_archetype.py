"""Python-side mirror of the C++ flow-archetype definition. NOT a classifier.

The archetype is a genuine Kairos concept, defined and computed in C++:
  * the taxonomy is `spark_dsg::ArchetypeType`
    (STATIC/UNIMODAL/BIMODAL/MULTIMODAL/DIFFUSE), on the `ARCHETYPES` scene-graph
    layer (`DsgLayers::ARCHETYPES`, LayerId 4);
  * the classification runs in `hydra::UpdateArchetypesFunctor`, config-driven
    (`layer_updates.archetypes`: dominant_weight_threshold / min_flow_support /
    min_probability), so it is available to every consumer of the DSG including
    the ROS library.

This module only restates the enum codes, names, and node colours so Python
tools (the viz recorder / renderer / analysis scripts) can label an archetype
they READ back from the graph. It deliberately holds no thresholds and does no
classification: doing so is what let four copies drift. If the taxonomy or
colours change, change them in C++ (`node_attributes.h` /
`update_archetypes_functor.cpp`) and update this mirror to match.
"""

# Order == spark_dsg::ArchetypeType uint8 values, so the code doubles as the
# index into COLOR_LIST / NAMES.
NAMES = ["static", "unimodal", "bimodal", "multimodal", "diffuse"]
CODE = {name: i for i, name in enumerate(NAMES)}

# RGB hex mirroring UpdateArchetypesFunctor::rewriteArchetypes node colours.
COLORS = {
    "static": "#808080",      # (128,128,128)
    "unimodal": "#00c800",    # (0,200,0)
    "bimodal": "#0064ff",     # (0,100,255)
    "multimodal": "#ff3200",  # (255,50,0)
    "diffuse": "#e6c800",     # (230,200,0)
}


def color_list():
    """Colours indexed by archetype code (for a recording's arch_colors)."""
    return [COLORS[name] for name in NAMES]
