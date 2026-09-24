"""Kairos visualizer: render a flow scene-graph *recording* to a Rerun .rrd.

Optional and standalone (numpy + rerun). It never imports the Kairos C++ core,
and the core has no dependency on it, so installing Kairos does not pull this in.
Install it only if you want the Rerun visualization; the ROS/rviz viewer remains
the primary viewer either way. A run emits a recording (see schema.py) and this
package turns it into a scrubbable Rerun recording.
"""
from .schema import Recording, RenderConfig
from .render import render_recording

__all__ = ["render_recording", "Recording", "RenderConfig"]
__version__ = "0.1.0"
