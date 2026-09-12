"""
extractors.py — Tier 1 Document Text Extractor

Handles PDF (single-column and 2-column layouts) and DOCX files.
Produces clean, reading-order-correct raw text for downstream processing.

Architecture:
    PDFExtractor   — pdfplumber-based, handles 2-column CV layouts via
                     bounding box crop strategy (page.crop) with
                     automatic column detection per page.
    DOCXExtractor  — python-docx-based, handles paragraphs and tables.
    extract_text_from_file() — Factory function auto-detects file type.

Column Detection Strategy (per page):
    Strategy A: Detect explicit vertical line/rect dividers (LaTeX/Word/Canva templates)
    Strategy B: Gutter Histogram Analysis — occupancy mask of word X-coordinates,
                find widest contiguous whitespace in 20%-80% of page width.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, Optional

import pdfplumber
from pdfplumber.page import Page
from loguru import logger

try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
except ImportError:  # pragma: no cover
    DocxDocument = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Internal Data Structures
# ---------------------------------------------------------------------------

class PageRegions(NamedTuple):
    """Bounding box regions identified for a single PDF page."""
    has_two_columns: bool
    y_header_bottom: float
    y_footer_top: float
    x_split: float  # Only meaningful when has_two_columns=True


# ---------------------------------------------------------------------------
# PDF Extractor
# ---------------------------------------------------------------------------

class PDFExtractor:
    """
    Extracts text from PDF files handling 1-column and 2-column CV layouts.

    Uses pdfplumber with bounding-box crop strategy to ensure correct reading
    order (Header → Left Column → Right Column → Footer) on each page.

    Args:
        min_gutter_width:      Min whitespace gap (pt) between 2 columns (default 12.0).
        min_column_width_ratio: Columns must occupy ≥ this fraction of page width (default 0.20).
        header_height_ratio_cap: Header is at most this fraction of page height (default 0.35).
        footer_height_ratio_cap: Footer is at most this fraction of page height (default 0.08).
    """

    def __init__(
        self,
        min_gutter_width: float = 12.0,
        min_column_width_ratio: float = 0.20,
        header_height_ratio_cap: float = 0.35,
        footer_height_ratio_cap: float = 0.08,
    ) -> None:
        self.min_gutter_width = min_gutter_width
        self.min_column_width_ratio = min_column_width_ratio
        self.header_height_ratio_cap = header_height_ratio_cap
        self.footer_height_ratio_cap = footer_height_ratio_cap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, file_path: str | Path) -> str:
        """
        Extract all text from a PDF file in correct reading order.

        Args:
            file_path: Path to the .pdf file.

        Returns:
            Full extracted text string (pages separated by form-feed '\\f').

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a readable PDF.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        page_texts: list[str] = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                logger.debug(f"Opened PDF '{path.name}' — {len(pdf.pages)} page(s).")
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = self._extract_page(page)
                        if page_text.strip():
                            page_texts.append(page_text)
                        logger.debug(f"  Page {i + 1}: {len(page_text)} chars extracted.")
                    except Exception as exc:  # pragma: no cover
                        logger.warning(f"  Page {i + 1} extraction failed: {exc}")
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{path}': {exc}") from exc

        return "\n\n".join(page_texts)

    # ------------------------------------------------------------------
    # Per-Page Extraction
    # ------------------------------------------------------------------

    def _extract_page(self, page: Page) -> str:
        """Extract text from a single page in correct reading order."""
        regions = self._detect_regions(page)

        if not regions.has_two_columns:
            return page.extract_text() or ""

        sections: list[str] = []

        # Header (full-width)
        if regions.y_header_bottom > 0:
            text = self._crop_extract(
                page, 0, 0, page.width, regions.y_header_bottom
            )
            if text:
                sections.append(text)

        # Left Column
        text = self._crop_extract(
            page,
            0,
            regions.y_header_bottom,
            regions.x_split,
            regions.y_footer_top,
        )
        if text:
            sections.append(text)

        # Right Column
        text = self._crop_extract(
            page,
            regions.x_split,
            regions.y_header_bottom,
            page.width,
            regions.y_footer_top,
        )
        if text:
            sections.append(text)

        # Footer (full-width)
        if regions.y_footer_top < page.height:
            text = self._crop_extract(
                page, 0, regions.y_footer_top, page.width, page.height
            )
            if text:
                sections.append(text)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Region Detection
    # ------------------------------------------------------------------

    def _detect_regions(self, page: Page) -> PageRegions:
        """Identify header/body/footer boundaries and check for 2-column layout."""
        y_header_bottom, y_footer_top = self._detect_vertical_zones(page)
        has_two_col, x_split = self._detect_two_columns(
            page, y_header_bottom, y_footer_top
        )
        return PageRegions(
            has_two_columns=has_two_col,
            y_header_bottom=y_header_bottom,
            y_footer_top=y_footer_top,
            x_split=x_split,
        )

    def _detect_vertical_zones(self, page: Page) -> tuple[float, float]:
        """
        Identify y-coordinates for header bottom and footer top.

        Full-width header detection: A line is 'full-width' if its x span
        crosses both the 40% and 60% width marks (i.e., spans the centre),
        OR if it is wider than 55% of the page, OR if it starts at x<5% of page.
        This captures candidate name, contacts, summary at top of CV.
        """
        width, height = page.width, page.height
        max_header_y = height * self.header_height_ratio_cap
        min_footer_y = height * (1.0 - self.footer_height_ratio_cap)

        words = page.extract_words()
        if not words:
            return 0.0, height

        # --- Header bottom ---
        header_words = [w for w in words if w["bottom"] <= max_header_y]
        y_header_bottom = 0.0

        if header_words:
            # Cluster words into lines by 'top' with tolerance 4pt
            header_words.sort(key=lambda w: w["top"])
            lines: list[list[dict]] = []
            current: list[dict] = [header_words[0]]
            for w in header_words[1:]:
                if abs(w["top"] - current[-1]["top"]) < 4.0:
                    current.append(w)
                else:
                    lines.append(current)
                    current = [w]
            lines.append(current)

            for line_words in lines:
                min_x = min(w["x0"] for w in line_words)
                max_x = max(w["x1"] for w in line_words)
                span = max_x - min_x
                is_full_width = (
                    span > 0.55 * width
                    or (min_x < 0.40 * width and max_x > 0.60 * width)
                    or min_x < 0.05 * width
                )
                if is_full_width:
                    line_bottom = max(w["bottom"] for w in line_words)
                    y_header_bottom = max(y_header_bottom, line_bottom)

            # Small buffer below last header line
            if y_header_bottom > 0:
                y_header_bottom = min(y_header_bottom + 4.0, max_header_y)

        # --- Footer top ---
        y_footer_top = height
        footer_words = [w for w in words if w["top"] >= min_footer_y]
        if footer_words:
            y_footer_top = min(w["top"] for w in footer_words) - 2.0

        return y_header_bottom, y_footer_top

    def _detect_two_columns(
        self, page: Page, y_top: float, y_bottom: float
    ) -> tuple[bool, float]:
        """
        Determine if the body zone has 2 columns. Returns (found, x_split).

        Strategy A: Explicit vertical line/rect divider.
        Strategy B: Gutter histogram on word X-coordinates.
        """
        width = page.width
        body_height = y_bottom - y_top
        if body_height <= 0:
            return False, 0.0

        # --- Strategy A: Vertical line or thin rect divider ---
        for line in getattr(page, "lines", []):
            if abs(line.get("x0", 0) - line.get("x1", 0)) < 2.0:  # vertical
                line_height = line.get("bottom", 0) - line.get("top", 0)
                if line_height >= 0.35 * body_height:
                    x_pos = (line["x0"] + line["x1"]) / 2.0
                    if (
                        self.min_column_width_ratio * width
                        <= x_pos
                        <= (1.0 - self.min_column_width_ratio) * width
                    ):
                        logger.debug(f"  Strategy A (line divider): x_split={x_pos:.1f}")
                        return True, x_pos

        for rect in getattr(page, "rects", []):
            rect_w = rect.get("x1", 0) - rect.get("x0", 0)
            rect_h = rect.get("bottom", 0) - rect.get("top", 0)
            if rect_w <= 3.0 and rect_h >= 0.35 * body_height:
                x_pos = (rect["x0"] + rect["x1"]) / 2.0
                if (
                    self.min_column_width_ratio * width
                    <= x_pos
                    <= (1.0 - self.min_column_width_ratio) * width
                ):
                    logger.debug(f"  Strategy A (rect divider): x_split={x_pos:.1f}")
                    return True, x_pos

        # --- Strategy B: Gutter histogram ---
        body_words = [
            w for w in page.extract_words()
            if w["top"] >= y_top and w["bottom"] <= y_bottom
        ]
        if len(body_words) < 15:
            return False, 0.0

        min_search_x = int(self.min_column_width_ratio * width)
        max_search_x = int((1.0 - self.min_column_width_ratio) * width)

        # Build 1D occupancy array (how many words cover each x-pixel)
        occupancy = [0] * (int(width) + 2)
        for w in body_words:
            for x in range(max(0, int(w["x0"])), min(int(width), int(w["x1"])) + 1):
                occupancy[x] += 1

        # Find contiguous empty gaps (gutters) in the search zone
        gutters: list[tuple[int, int, int]] = []  # (width, x_start, x_end)
        in_gap = False
        gap_start = 0
        for x in range(min_search_x, max_search_x + 1):
            if occupancy[x] == 0:
                if not in_gap:
                    in_gap = True
                    gap_start = x
            else:
                if in_gap:
                    in_gap = False
                    gap_len = x - gap_start
                    if gap_len >= self.min_gutter_width:
                        gutters.append((gap_len, gap_start, x))
        if in_gap:
            gap_len = (max_search_x + 1) - gap_start
            if gap_len >= self.min_gutter_width:
                gutters.append((gap_len, gap_start, max_search_x + 1))

        if not gutters:
            return False, 0.0

        # Pick the widest gutter
        gutters.sort(key=lambda g: g[0], reverse=True)
        best_gap = gutters[0]
        x_split = (best_gap[1] + best_gap[2]) / 2.0

        # Verify both sides have substantial text
        left_count = sum(1 for w in body_words if w["x1"] <= x_split)
        right_count = sum(1 for w in body_words if w["x0"] >= x_split)
        if left_count >= 5 and right_count >= 5:
            logger.debug(
                f"  Strategy B (gutter): x_split={x_split:.1f}, "
                f"left={left_count} words, right={right_count} words"
            )
            return True, x_split

        return False, 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _crop_extract(
        self,
        page: Page,
        x0: float,
        top: float,
        x1: float,
        bottom: float,
    ) -> str:
        """
        Safely crop a page region and extract text.

        Clamps all coordinates to page bounds and uses strict=False to
        avoid pdfplumber bounding-box validation errors from float imprecision.
        """
        x0 = max(0.0, min(x0, page.width))
        x1 = max(0.0, min(x1, page.width))
        top = max(0.0, min(top, page.height))
        bottom = max(0.0, min(bottom, page.height))

        if (x1 - x0) <= 2.0 or (bottom - top) <= 2.0:
            return ""

        try:
            cropped = page.crop((x0, top, x1, bottom), strict=False)
            return (cropped.extract_text() or "").strip()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"  Crop failed ({x0:.0f},{top:.0f},{x1:.0f},{bottom:.0f}): {exc}")
            return ""


# ---------------------------------------------------------------------------
# DOCX Extractor
# ---------------------------------------------------------------------------

class DOCXExtractor:
    """
    Extracts text from DOCX files using python-docx.

    Processes:
        - Paragraphs (body text, headings)
        - Tables (skill matrices, education tables)
        - Section-level multi-column layouts (via XML namespace)
    """

    def extract(self, file_path: str | Path) -> str:
        """
        Extract all text from a DOCX file.

        Args:
            file_path: Path to the .docx file.

        Returns:
            Extracted text string.

        Raises:
            ImportError: If python-docx is not installed.
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be opened as a DOCX.
        """
        if DocxDocument is None:  # pragma: no cover
            raise ImportError("python-docx is not installed. Run: pip install python-docx")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        try:
            doc = DocxDocument(str(path))
        except Exception as exc:
            raise ValueError(f"Cannot open DOCX '{path}': {exc}") from exc

        parts: list[str] = []

        # Iterate body elements in document order (paragraphs and tables interleaved)
        body = doc.element.body
        for child in body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag == "p":
                # Paragraph
                text = "".join(
                    run.text for run in child.iter(qn("w:t"))
                )
                if text.strip():
                    parts.append(text.strip())

            elif tag == "tbl":
                # Table: read row by row, cell by cell
                table_lines: list[str] = []
                for row in child.iter(qn("w:tr")):
                    cell_texts = []
                    for cell in row.iter(qn("w:tc")):
                        cell_text = " ".join(
                            "".join(t.text for t in cell.iter(qn("w:t"))).split()
                        )
                        if cell_text:
                            cell_texts.append(cell_text)
                    if cell_texts:
                        table_lines.append(" | ".join(cell_texts))
                if table_lines:
                    parts.append("\n".join(table_lines))

        logger.debug(f"DOCX '{path.name}': extracted {len(parts)} text blocks.")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

# Module-level singleton instances (reusable across calls)
_pdf_extractor = PDFExtractor()
_docx_extractor = DOCXExtractor()


def extract_text_from_file(
    file_path: str | Path,
    pdf_extractor: Optional[PDFExtractor] = None,
    docx_extractor: Optional[DOCXExtractor] = None,
) -> str:
    """
    Detect file type and extract text from a CV file (PDF or DOCX).

    Args:
        file_path: Path to a .pdf or .docx file.
        pdf_extractor: Optional custom PDFExtractor instance.
        docx_extractor: Optional custom DOCXExtractor instance.

    Returns:
        Extracted plain text string in reading order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported (.pdf or .docx).

    Examples:
        >>> text = extract_text_from_file("data/raw/cvs/resume.pdf")
        >>> text = extract_text_from_file("data/raw/cvs/cv.docx")
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        extractor = pdf_extractor or _pdf_extractor
        logger.info(f"Extracting PDF: {path.name}")
        return extractor.extract(path)

    elif suffix in (".docx", ".doc"):
        extractor = docx_extractor or _docx_extractor
        logger.info(f"Extracting DOCX: {path.name}")
        return extractor.extract(path)

    else:
        raise ValueError(
            f"Unsupported file type '{suffix}' for '{path.name}'. "
            "Supported: .pdf, .docx"
        )
