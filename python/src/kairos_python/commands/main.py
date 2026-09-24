"""Entry point for the `kairos` CLI."""
import click

import kairos_python.commands.run as run


@click.group()
def cli():
    """KAIROS: 4D Scene Graphs for Periodic Flow Dynamics."""
    pass


cli.add_command(run.cli)
