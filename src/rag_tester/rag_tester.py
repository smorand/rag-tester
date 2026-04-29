"""CLI entry point for the rag-tester application."""

import logging
from typing import Annotated

import typer

from rag_tester.config import Settings
from rag_tester.logging_config import setup_logging
from rag_tester.tracing import configure_tracing
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
    setup_logging(app_name=settings.app_name, verbose=verbose, quiet=quiet)
    configure_tracing(app_name=settings.app_name)


@app.command()
def version() -> None:
    """Display the version of rag-tester."""
    typer.echo(f"rag-tester version {__version__}")


if __name__ == "__main__":
    app()
