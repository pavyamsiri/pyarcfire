"""This module contains functions to help set up logging."""

import logging
from collections.abc import Iterable

from rich.console import Console
from rich.logging import RichHandler


def setup_logging() -> Iterable[logging.Handler]:
    FORMAT = "%(message)s"
    console = Console()
    console_handler = RichHandler(console=console, show_time=False, markup=True)
    handlers: list[logging.Handler] = [console_handler]
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=handlers,
        encoding="utf-8",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return handlers
