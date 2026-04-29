"""CLI entry point for the rag-tester application."""

import logging
from typing import Annotated

import typer

from rag_tester.commands import load_command
from rag_tester.config import Settings
from rag_tester.logging_config import setup_logging
from rag_tester.tracing import setup_tracing
from rag_tester.version import __version__

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Only show warnings and errors"),
    ] = False,
) -> None:
    """RAG Tester - Testing and evaluating Retrieval-Augmented Generation systems."""
    settings = Settings()

    # Adjust log level based on flags
    if verbose:
        settings.log_level = "DEBUG"
    elif quiet:
        settings.log_level = "WARNING"

    setup_logging(settings)
    setup_tracing(settings)


@app.command()
def version() -> None:
    """Display the version of rag-tester."""
    typer.echo(f"rag-tester version {__version__}")


# Register load command
app.command(name="load")(load_command)


if __name__ == "__main__":
    app()
