"""Markdown rendering for pdfmd.

This module converts transformed `PageText` structures into Markdown.
It assumes header/footer removal and drop-cap stripping have already been run
(see `transform.py`).

Main entry: `render_document(pages, options, body_sizes=None, progress_cb=None)`

Key behaviours:
- Applies heading promotion via font size and optional CAPS heuristics.
- Normalizes bullets and numbered lists to proper Markdown formats.
- Repairs hyphenation and unwraps hard line breaks into paragraphs.
- Optionally inserts `---` page break markers between pages.
- Defragments short orphan lines into their preceding paragraphs.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Callable, List, Optional

from .models import Block, Line, PageText, Options
from .utils import normalize_punctuation, linkify_urls, escape_markdown
from .transform import is_all_caps_line, is_mostly_caps


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _wrap_inline(text: str, bold: bool, italic: bool) -> str:
    """Wrap text in Markdown bold/italic markers, if any.

    Assumes `text` has already been escaped with `escape_markdown`.
    """
    if not text.strip():
        return text
    if bold and italic:
        return f"***{text}***"
    if bold:
        return f"**{text}**"
    if italic:
        return f"*{text}*"
    return text


# ---------------------------------------------------------------------------
# Line / paragraph utilities
# ---------------------------------------------------------------------------


def _fix_hyphenation(text: str) -> str:
    """Repair line-wrap hyphenation.

    Typical case in PDFs:
        'hy-\nphen' → 'hyphen'

    We only remove hyphen + newline when it is clearly a wrap.
    """
    return re.sub(r"-\n(\s*)", r"\1", text)


def _unwrap_hard_breaks(lines: List[str]) -> str:
    """Merge wrapped lines into paragraphs. Blank lines remain paragraph breaks.

    Rules:
    - Consecutive non-blank lines are joined with spaces.
    - Blank lines are preserved as paragraph separators.
    - Lines ending with two spaces `"  "` are treated as explicit hard breaks
      (Markdown convention) and terminate the paragraph.
    """
    out: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            out.append(" ".join(buf).strip())
            buf.clear()

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            flush()
            out.append("")
            continue

        if line.endswith("  "):
            buf.append(line.rstrip())
            flush()
        else:
            buf.append(line.strip())

    flush()
    return "\n".join(out)


def _defragment_orphans(md: str, max_len: int = 45) -> str:
    """Merge short orphan lines into the previous paragraph.

    An orphan is:
    - A non-empty line,
    - Surrounded by blank lines,
    - Shorter than or equal to `max_len`,
    - Not a heading.

    These often come from center titles, section labels, etc.
    """
    lines = md.splitlines()
    res: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if (
            i > 0
            and i < len(lines) - 1
            and not lines[i - 1].strip()
            and not lines[i + 1].strip()
            and 0 < len(line.strip()) <= max_len
            and not line.strip().startswith("#")
        ):
            # Attach orphan to the previous non-blank line
            j = len(res) - 1
            while j >= 0 and not res[j].strip():
                j -= 1
            if j >= 0:
                res[j] = (res[j].rstrip() + " " + line.strip()).strip()
                i += 2
                continue

        res.append(line)
        i += 1

    return "\n".join(res)


# ---------------------------------------------------------------------------
# Span joining (word-safe)
# ---------------------------------------------------------------------------


def _safe_join_texts(parts: List[str]) -> str:
    """Join span texts while preserving word boundaries.

    We inject a space only when concatenating would otherwise merge two
    alphanumeric characters from adjacent spans (e.g. "Hello" + "world"
    → "Hello world"). Existing whitespace at boundaries is respected.
    """
    if not parts:
        return ""
    out: List[str] = [parts[0]]
    for t in parts[1:]:
        if not t:
            continue
        prev = out[-1]
        prev_last = prev[-1] if prev else ""
        cur_first = t[0]

        if (
            prev_last
            and cur_first
            and not prev_last.isspace()
            and not cur_first.isspace()
            and prev_last.isalnum()
            and cur_first.isalnum()
        ):
            out.append(" " + t)
        else:
            out.append(t)
    return "".join(out)


# ---------------------------------------------------------------------------
# Block → Markdown lines
# ---------------------------------------------------------------------------


def _is_footer_noise(line: str) -> bool:
    """Heuristic to detect noisy footer/header artifacts.

    Examples:
        "- - 1"
        "- - - - - - 1. 2. 3. 4."
        "---- 2 ----"
    """
    s = line.strip()
    if not s:
        return False

    # Strong signal: mostly dashes, dots, and digits
    if re.fullmatch(r"[-\s\.0-9]+", s):
        # Ensure there's more punctuation than digits/words
        dash_count = s.count("-")
        if dash_count >= 2:
            return True
    return False


def _normalize_list_line(ln: str) -> str:
    """Normalize various bullet/numbered prefixes into Markdown list syntax."""
    s = ln.lstrip()
    # Bullet-like prefixes
    if re.match(r"^[•○◦·\-–—]\s+", s):
        s = re.sub(r"^[•○◦·\-–—]\s+", "- ", s)
        return s

    # Numbered: "1. text" or "1) text"
    m_num = re.match(r"^(\d+)[\.\)]\s+", s)
    if m_num:
        num = m_num.group(1)
        s = re.sub(r"^\d+[\.\)]\s+", f"{num}. ", s)
        return s

    # Lettered outlines: "A. text" or "a) text" → bullet
    if re.match(r"^[A-Za-z][\.\)]\s+", s):
        s = re.sub(r"^[A-Za-z][\.\)]\s+", "- ", s)
        return s

    return ln.strip()


def _block_to_lines(
    block: Block,
    body_size: float,
    caps_to_headings: bool,
    heading_size_ratio: float,
) -> List[str]:
    """Convert a Block into a list of Markdown lines.

    We build two parallel views:
      - raw_lines: plain text (no Markdown), for heading detection
      - rendered_lines: text with inline styling (bold/italic), for body output

    Heading detection uses:
      - average span font size vs body_size
      - optional ALL-CAPS / MOSTLY-CAPS heuristic across the block
    """
    rendered_lines: List[str] = []
    raw_lines: List[str] = []
    line_sizes: List[float] = []

    for line in block.lines:
        spans = line.spans

        texts_fmt: List[str] = []
        texts_raw: List[str] = []
        sizes: List[float] = []

        for sp in spans:
            raw_text = sp.text or ""
            texts_raw.append(raw_text)

            esc = escape_markdown(raw_text)
            esc = _wrap_inline(esc, sp.bold, sp.italic)
            texts_fmt.append(esc)

            if getattr(sp, "size", 0.0):
                sizes.append(float(sp.size))

        joined_fmt = _safe_join_texts(texts_fmt)
        joined_raw = _safe_join_texts(texts_raw)

        if joined_fmt.strip():
            rendered_lines.append(joined_fmt)
            raw_lines.append(joined_raw)
            if sizes:
                line_sizes.append(median(sizes))

    if not rendered_lines:
        return []

    avg_line_size = median(line_sizes) if line_sizes else body_size

    # Use RAW text (no ** or *) for heading heuristics
    block_text_flat = " ".join(raw_lines).strip()

    heading_by_size = avg_line_size >= body_size * heading_size_ratio
    heading_by_caps = caps_to_headings and (
        is_all_caps_line(block_text_flat) or is_mostly_caps(block_text_flat)
    )

    if heading_by_size or heading_by_caps:
        # H1 if much larger than body or if CAPS; otherwise H2
        level = 1 if (avg_line_size >= body_size * 1.6) or heading_by_caps else 2

        # Heading text: use ONLY the first RAW line, not the formatted one
        heading_raw = raw_lines[0]
        heading_text = escape_markdown(heading_raw)
        heading_text = re.sub(r"\s+", " ", heading_text).strip(" -:–—")
        heading_text = normalize_punctuation(heading_text)
        heading_line = f"{'#' * level} {heading_text}"

        # If there's no additional text, just output heading + blank line
        if len(rendered_lines) == 1:
            return [heading_line, ""]

        # Otherwise, render remaining lines as normal paragraph/list text
        tail_text = _fix_hyphenation("\n".join(rendered_lines[1:]))

        lines: List[str] = []
        for ln in tail_text.splitlines():
            if not ln.strip():
                lines.append("")
                continue

            if _is_footer_noise(ln):
                continue

            norm = _normalize_list_line(ln)
            lines.append(norm)

        para = _unwrap_hard_breaks(lines)
        para = normalize_punctuation(para)
        para = linkify_urls(para)

        out: List[str] = [heading_line, ""]
        if para.strip():
            out.append(para)
            out.append("")
        return out

    # ----------------- Normal paragraph path -----------------

    para_text = _fix_hyphenation("\n".join(rendered_lines))

    lines: List[str] = []
    for ln in para_text.splitlines():
        if not ln.strip():
            lines.append("")
            continue

        if _is_footer_noise(ln):
            continue

        norm = _normalize_list_line(ln)
        lines.append(norm)

    para = _unwrap_hard_breaks(lines)
    para = normalize_punctuation(para)
    para = linkify_urls(para)
    return [para, ""]


# ---------------------------------------------------------------------------
# Document render
# ---------------------------------------------------------------------------

DefProgress = Optional[Callable[[int, int], None]]


def render_document(
    pages: List[PageText],
    options: Options,
    body_sizes: Optional[List[float]] = None,
    progress_cb: DefProgress = None,
) -> str:
    """Render transformed pages to a Markdown string.

    Args:
        pages: transformed PageText pages
        options: rendering options (see models.Options)
        body_sizes: optional per-page body-size baselines.
                    If not provided, the renderer falls back to 11.0.
        progress_cb: optional progress callback (done, total)
    """
    md_lines: List[str] = []
    total = len(pages)

    for i, page in enumerate(pages):
        body = body_sizes[i] if body_sizes and i < len(body_sizes) else 11.0

        for blk in page.blocks:
            if blk.is_empty():
                continue
            md_lines.extend(
                _block_to_lines(
                    blk,
                    body_size=body,
                    caps_to_headings=options.caps_to_headings,
                    heading_size_ratio=options.heading_size_ratio,
                )
            )

        if options.insert_page_breaks and i < total - 1:
            md_lines.extend(["---", ""])  # page rule

        if progress_cb:
            progress_cb(i + 1, total)

    md = "\n".join(md_lines)
    # Collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"

    if options.defragment_short:
        md = _defragment_orphans(md, max_len=options.orphan_max_len)

    # Strip common footer artefacts like trailing "- - 1" or "- -" at end of lines
    md = re.sub(r"\s*-+\s*-+\s*\d*\s*$", "", md, flags=re.MULTILINE)

    # Tighten spaces before punctuation
    md = re.sub(r"\s+([,.;:?!])", r"\1", md)

    return md


__all__ = [
    "render_document",
]
