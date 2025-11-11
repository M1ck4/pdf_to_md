"""pdfmd package initializer.

Exposes the core high‑level API:
  from pdfmd import pdf_to_markdown, Options

Also provides __version__ and console entry hint.
"""
from __future__ import annotations

from .models import Options
from .pipeline import pdf_to_markdown

__all__ = ["Options", "pdf_to_markdown"]

__version__ = "1.0.0"


def main():
    """Entry point alias for `python -m pdfmd.cli`.

    This allows `python -m pdfmd` to behave like running the CLI directly.
    """
    from .cli import main as _main
    raise SystemExit(_main())


if __name__ == "__main__":
    main()
