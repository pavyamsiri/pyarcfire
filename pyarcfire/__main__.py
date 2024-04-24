# Standard libraries
import argparse
import logging
from typing import Sequence

# Internal libraries
from .log_utils import setup_logging

log = logging.getLogger(__name__)


def main(raw_args: Sequence[str]) -> None:
    args = _parse_args(raw_args)
    log.debug(args)


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyarcfire",
        description="Python port of SpArcFiRe, a program that finds and reports spiral features in images.",
    )
    parser.add_argument(
        "-i",
        "--i",
        type=str,
        dest="input_path",
        help="Path to the input image.",
        required=True,
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    import sys

    setup_logging()
    main(sys.argv[1:])
