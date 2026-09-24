"""Allow `python -m kairos_python ...` to invoke the kairos CLI."""
from kairos_python.commands.main import cli

if __name__ == "__main__":
    cli()
