# Rerun visualizer image: renders a portable recording (.npz) to an .rrd.
# Standalone and small; shares nothing with the kairos pipeline image (which is
# Python 3.8, too old for rerun). Build from this directory:
#   docker build -t kairos-viz .
FROM python:3.11-slim

# Install the dependencies before copying the sources, so that editing a source
# file reuses the cached dependency layer instead of reinstalling rerun-sdk and
# numpy. pyproject.toml stays the only place the dependencies are declared: a
# stub package makes it installable on its own, and the real sources land in the
# following layer.
COPY pyproject.toml /opt/kairos_visualizer/
RUN mkdir -p /opt/kairos_visualizer/kairos_visualizer \
    && touch /opt/kairos_visualizer/kairos_visualizer/__init__.py \
    && pip install --no-cache-dir /opt/kairos_visualizer

COPY . /opt/kairos_visualizer
RUN pip install --no-cache-dir --no-deps --force-reinstall /opt/kairos_visualizer

WORKDIR /out
ENTRYPOINT ["python", "-m", "kairos_visualizer"]
