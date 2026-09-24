"""Command line: python -m kairos_visualizer RECORDING.npz -o OUT.rrd"""
import argparse

from .render import render_recording
from .schema import RenderConfig


def main():
    ap = argparse.ArgumentParser(
        prog="kairos-visualize",
        description="Render a Kairos flow scene-graph recording to a Rerun .rrd.")
    ap.add_argument("recording", help="input recording .npz")
    ap.add_argument("-o", "--out", required=True, help="output .rrd")
    ap.add_argument("--arrow-color", choices=("archetype", "heading"),
                    default=RenderConfig.arrow_color,
                    help="colour the per-place flow arrows by motion archetype "
                         "or by heading (hue wheel over direction)")
    ap.add_argument("--static-archetype-edges", action="store_true",
                    help="draw the archetype web for STATIC components")
    ap.add_argument("--arrow-scale", type=float, default=RenderConfig.arrow_scale,
                    help="metres of arrow per m/s of speed. An arrow stays inside "
                         "its own place when this times the top speed is below the "
                         "spacing between places, so nothing overlaps a neighbour.")
    ap.add_argument("--arrow-radius", type=float, default=RenderConfig.arrow_radius,
                    help="shaft radius (m) of the dominant component")
    ap.add_argument("--arrow-radius-minor", type=float,
                    default=RenderConfig.arrow_radius_minor,
                    help="shaft radius (m) of the weaker components")
    ap.add_argument("--max-components", type=int, default=RenderConfig.max_components,
                    help="components drawn per place; 1 keeps only the dominant "
                         "heading, so each place carries a single arrow")
    ap.add_argument("--arrow-source", choices=("components", "dominant"),
                    default=RenderConfig.arrow_source,
                    help="components: one arrow per mixture component, each at its "
                         "pinned cardinal heading, so multimodality is visible. "
                         "dominant: one arrow per place at the weighted-mean "
                         "heading, free of the cardinal layout, so direction "
                         "varies smoothly across the map.")
    ap.add_argument("--fade-by-weight", action="store_true",
                    help="carry component weight in opacity as well as in size, so "
                         "a place's weaker directions read as faint overlays")
    ap.add_argument("--alpha-min", type=float, default=RenderConfig.arrow_alpha_min,
                    help="opacity of the weakest component with --fade-by-weight")
    args = ap.parse_args()
    render_recording(args.recording, args.out,
                     RenderConfig(arrow_color=args.arrow_color,
                                  static_archetype_edges=args.static_archetype_edges,
                                  arrow_scale=args.arrow_scale,
                                  arrow_radius=args.arrow_radius,
                                  arrow_radius_minor=args.arrow_radius_minor,
                                  max_components=args.max_components,
                                  arrow_source=args.arrow_source,
                                  arrow_alpha_by_weight=args.fade_by_weight,
                                  arrow_alpha_min=args.alpha_min))


if __name__ == "__main__":
    main()
