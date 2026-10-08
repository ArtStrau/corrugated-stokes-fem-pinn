"""Private shared CLI support for selecting a top-level scientific result root."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_RESULT_ROOT = Path("reproduction")
CANONICAL_RESULT_ROOT = Path("results")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_result_root_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the mutually exclusive canonical/custom root selectors."""

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--canonical",
        action="store_true",
        help="use the canonical top-level results/ root",
    )
    group.add_argument(
        "--results-root",
        type=Path,
        metavar="PATH",
        help="use PATH as the top-level result root",
    )


def _absolute_from_project(path: Path) -> Path:
    """Resolve a root as commands issued from the project directory would."""

    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether *path* is *parent* or one of its descendants."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_result_root(args: argparse.Namespace) -> Path:
    """Resolve the selected top-level root and protect canonical ``results/``.

    A custom ``--results-root`` may not name the project's canonical results
    directory or a directory beneath it.  Canonical writes therefore always
    require the explicit ``--canonical`` selector.
    """

    if bool(getattr(args, "canonical", False)):
        return CANONICAL_RESULT_ROOT
    custom = getattr(args, "results_root", None)
    if custom is None:
        return DEFAULT_RESULT_ROOT
    selected = Path(custom)
    if _is_within(
        _absolute_from_project(selected),
        _absolute_from_project(CANONICAL_RESULT_ROOT),
    ):
        raise ValueError(
            "--results-root may not target the canonical results/ tree; "
            "use --canonical explicitly"
        )
    return selected


def parse_result_root(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Path:
    """Resolve a parsed root, reporting safety violations through argparse."""

    try:
        return resolve_result_root(args)
    except ValueError as error:
        parser.error(str(error))
        raise AssertionError("argparse.error must terminate") from error
