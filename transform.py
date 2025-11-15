"""Text shaping & heuristics for pdfmd.

This module transforms `PageText` structures prior to Markdown rendering.
It is *format-agnostic*: it never emits Markdown. The goal is to clean and
annotate the intermediate model so the renderer can stay simple and
predictable.

Included heuristics:
- Detect and remove repeating headers/footers across pages.
- Strip obvious drop caps (oversized first letter at paragraph start).
- Compute body-size baselines used for heading promotion (by size).
- Provide ALL-CAPS helpers used by the renderer for heading promotion.

Transform functions return new `PageText` instances (immutability by copy), so
upstream stages can compare before/after if needed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import List, Optional, Tuple
import re

from .models import PageText, Block, Line, Span, Options


# --------------------------- CAPS heuristics ---------------------------

def is_all_caps_line(s: str) -> bool:
    """Return True if a line is *entirely* alphabetic and uppercase.

    Non‑letters are ignored when making the decision.
    """
    core = re.sub(r"[^A-Za-z]+", "", s)
    return bool(core) and core.isupper()


def is_mostly_caps(s: str, threshold: float = 0.75) -> bool:
    """Return True if at least `threshold` of alphabetic chars are uppercase."""
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    return sum(1 for ch in letters if ch.isupper()) / len(letters) >= threshold


# --------------------------- Header/Footer utils ---------------------------

def _line_text(line: Line) -> str:
    return "".join(sp.text for sp in line.spans).strip()


def _first_nonblank_line_text(page: PageText) -> str:
    for blk in page.blocks:
        for ln in blk.lines:
            t = _line_text(ln)
            if t:
                return t
    return ""


def _last_nonblank_line_text(page: PageText) -> str:
    for blk in reversed(page.blocks):
        for ln in reversed(blk.lines):
            t = _line_text(ln)
            if t:
                return t
    return ""


def detect_repeating_edges(pages: List[PageText], min_pages: int = 3) -> Tuple[Optional[str], Optional[str]]:
    """Detect repeating header/footer lines across pages.

    We look at the first and last non‑blank line of each page and count exact
    matches. If a candidate appears on at least `min_pages` pages, we treat it
    as a header/footer.

    This is intentionally conservative; pattern‑based footer cleanup is
    handled separately in `remove_header_footer`.
    """
    heads: Counter[str] = Counter()
    tails: Counter[str] = Counter()

    for p in pages:
        h = _first_nonblank_line_text(p)
        t = _last_nonblank_line_text(p)
        if h:
            heads[h] += 1
        if t:
            tails[t] += 1

    header = next((h for h, c in heads.most_common() if c >= min_pages and h), None)
    footer = next((t for t, c in tails.most_common() if c >= min_pages and t), None)
    return header, footer


def remove_header_footer(
    pages: List[PageText], header: Optional[str], footer: Optional[str]
) -> List[PageText]:
    """Return copies of pages with matching header/footer lines removed.

    We compare the joined text of each line to the detected strings and also
    apply some light pattern‑based cleanup for common footer artifacts like
    "- - 1" or bare page numbers at the very bottom.
    """
    footer_pattern = re.compile(r"^[-–—]\s*[-–—]?\s*\d*\s*$")
    page_num_pattern = re.compile(r"^\d+$")

    out_pages: List[PageText] = []

    for p in pages:
        new_blocks: List[Block] = []

        for blk_idx, blk in enumerate(p.blocks):
            new_lines: List[Line] = []
            num_lines = len(blk.lines)

            for ln_idx, ln in enumerate(blk.lines):
                text = _line_text(ln)

                # Exact header/footer match
                if header and text == header:
                    continue
                if footer and text == footer:
                    continue

                # Pattern‑based footer removal: only consider *last* line of
                # the *last* block on the page.
                is_last_block = blk_idx == len(p.blocks) - 1
                is_last_line = ln_idx == num_lines - 1
                if is_last_block and is_last_line:
                    if footer_pattern.match(text) or page_num_pattern.match(text):
                        continue

                new_lines.append(ln)

            if new_lines:
                new_blocks.append(replace(blk, lines=new_lines))

        out_pages.append(replace(p, blocks=new_blocks))

    return out_pages


# ------------------------------- Drop caps -------------------------------

def strip_drop_caps_in_page(page: PageText) -> PageText:
    """Strip obvious decorative drop caps from the start of blocks.

    Heuristic: if the first span in the first *non‑blank* line of a block is a
    single alphabetic character, and its font size is much larger than the
    median size of the rest of the line, we remove it and optionally merge it
    into the following span.
    """
    new_blocks: List[Block] = []

    for blk in page.blocks:
        lines = blk.lines
        if not lines:
            new_blocks.append(blk)
            continue

        modified = False
        new_lines: List[Line] = []

        for ln in lines:
            spans = ln.spans
            if not spans or modified:
                new_lines.append(ln)
                continue

            first = spans[0]
            rest = spans[1:]
            core = first.text

            # Only consider obvious single‑letter decorative initials
            if (
                core
                and len(core.strip()) == 1
                and core.strip().isalpha()
                and rest
            ):
                # Compare sizes
                sizes = [sp.size for sp in rest if getattr(sp, "size", 0) > 0]
                if sizes:
                    median = sorted(sizes)[len(sizes) // 2]
                    if first.size >= median * 1.6:
                        # Drop the decorative span; keep the rest of the line.
                        new_ln = replace(ln, spans=rest)
                        new_lines.append(new_ln)
                        modified = True
                        continue

            new_lines.append(ln)

        new_blocks.append(replace(blk, lines=new_lines))

    return replace(page, blocks=new_blocks)


def strip_drop_caps(pages: List[PageText]) -> List[PageText]:
    """Apply `strip_drop_caps_in_page` to all pages."""
    return [strip_drop_caps_in_page(p) for p in pages]


# ------------------------------ Body sizes ------------------------------

def estimate_body_size(pages: List[PageText]) -> List[float]:
    """Estimate a "body text" font size per page.

    We collect all non‑empty span sizes on each page and take the median.
    If a page has no spans with a positive size, we fall back to 11.0.
    """
    body_sizes: List[float] = []

    for p in pages:
        sizes: List[float] = []
        for blk in p.blocks:
            for ln in blk.lines:
                for sp in ln.spans:
                    if getattr(sp, "size", 0) > 0 and sp.text.strip():
                        sizes.append(float(sp.size))

        if sizes:
            sizes_sorted = sorted(sizes)
            m = len(sizes_sorted) // 2
            if len(sizes_sorted) % 2:
                body_sizes.append(sizes_sorted[m])
            else:
                body_sizes.append((sizes_sorted[m - 1] + sizes_sorted[m]) / 2.0)
        else:
            body_sizes.append(11.0)

    return body_sizes


# ----------------------------- High-level pass -----------------------------

def transform_pages(
    pages: List[PageText], options: Options
) -> Tuple[List[PageText], Optional[str], Optional[str], List[float]]:
    """Run the standard transform pipeline.

    Returns:
        pages_t        : transformed pages
        header, footer : detected repeating header/footer strings (if any)
        body_sizes     : per-page body-size baselines
    """
    # First, strip decorative drop caps
    pages_t = strip_drop_caps(pages)

    header: Optional[str] = None
    footer: Optional[str] = None

    if options.remove_headers_footers:
        header, footer = detect_repeating_edges(pages_t)
        pages_t = remove_header_footer(pages_t, header, footer)

    # Compute per-page body font size baselines for heading promotion.
    body_sizes = estimate_body_size(pages_t)

    return pages_t, header, footer, body_sizes


__all__ = [
    "is_all_caps_line",
    "is_mostly_caps",
    "detect_repeating_edges",
    "remove_header_footer",
    "strip_drop_caps_in_page",
    "strip_drop_caps",
    "estimate_body_size",
    "transform_pages",
]
