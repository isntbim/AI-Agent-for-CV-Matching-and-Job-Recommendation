"""
test_extractors.py — Unit tests for src/parser/extractors.py

Tests PDF extraction (single-column, 2-column layout detection),
DOCX extraction (paragraphs + tables), and the factory function.
Smoke tests run against real CV files from data/raw/cvs/.
"""

import pytest
from pathlib import Path

from src.parser.extractors import (
    PDFExtractor,
    DOCXExtractor,
    extract_text_from_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/raw/cvs")

def _get_pdf_files(subdir: str = "", max_count: int = 3) -> list[Path]:
    """Return up to max_count PDF files from the dataset."""
    base = DATA_DIR / subdir if subdir else DATA_DIR
    files = list(base.rglob("*.pdf"))[:max_count]
    return files


# ---------------------------------------------------------------------------
# PDFExtractor Unit Tests
# ---------------------------------------------------------------------------

class TestPDFExtractor:
    @pytest.fixture
    def extractor(self) -> PDFExtractor:
        return PDFExtractor()

    def test_raises_file_not_found(self, extractor):
        with pytest.raises(FileNotFoundError, match="not found"):
            extractor.extract("nonexistent_file_xyz.pdf")

    def test_raises_value_error_on_invalid_file(self, extractor, tmp_path):
        bad_file = tmp_path / "fake.pdf"
        bad_file.write_bytes(b"this is not a pdf")
        with pytest.raises(ValueError, match="Cannot open PDF"):
            extractor.extract(bad_file)

    @pytest.mark.skipif(
        not list(DATA_DIR.rglob("*.pdf")),
        reason="No PDF files found in data/raw/cvs/"
    )
    def test_smoke_extract_real_pdfs(self, extractor):
        """Smoke test: extract text from up to 5 real PDFs from dataset."""
        files = _get_pdf_files(max_count=5)
        assert len(files) > 0, "No PDFs found in dataset directory."

        for pdf_path in files:
            text = extractor.extract(pdf_path)
            assert isinstance(text, str), f"Expected str for {pdf_path.name}"
            # A real CV should have at least some content
            if pdf_path.stat().st_size > 5000:  # Skip very small files
                assert len(text) > 50, (
                    f"Suspiciously short output for {pdf_path.name} "
                    f"({pdf_path.stat().st_size} bytes): '{text[:100]}'"
                )

    @pytest.mark.skipif(
        not list(DATA_DIR.rglob("*.pdf")),
        reason="No PDF files found in data/raw/cvs/"
    )
    def test_smoke_text_is_not_mixed_columns(self, extractor):
        """
        Verify that 2-column CV text is extracted in reading order per column,
        NOT line-by-line across columns.

        A mixed-column output would concatenate section headers from both columns
        on the same line, e.g. 'EXPERIENCE  SKILLS' or 'EDUCATION  LANGUAGES'.
        We check that no line contains two different well-known CV section headers
        that would never naturally appear together.
        """
        SECTION_PAIRS = [
            ("EXPERIENCE", "SKILLS"),
            ("EDUCATION", "LANGUAGES"),
            ("EXPERIENCE", "LANGUAGES"),
            ("WORK", "CONTACT"),
            ("EXPERIENCE", "CERTIFICATIONS"),
        ]

        files = _get_pdf_files(max_count=10)
        for pdf_path in files:
            text = extractor.extract(pdf_path)
            lines = text.splitlines()
            for line in lines:
                line_upper = line.upper()
                for header_a, header_b in SECTION_PAIRS:
                    if header_a in line_upper and header_b in line_upper:
                        # Only flag if both are in the same short line (< 80 chars)
                        # Long lines may legitimately mention both concepts
                        if len(line) < 80:
                            pytest.fail(
                                f"Possible column mixing in '{pdf_path.name}':\n"
                                f"  Line ({len(line)} chars): {line!r}\n"
                                f"  Contains both '{header_a}' and '{header_b}'"
                            )

    @pytest.mark.skipif(
        not list((DATA_DIR / "Software Engineer").iterdir() if (DATA_DIR / "Software Engineer").is_dir() else []),
        reason="Software Engineer subdirectory not found"
    )
    def test_smoke_software_engineer_cvs(self, extractor):
        """Test extraction on Software Engineer category CVs."""
        files = _get_pdf_files(subdir="Software Engineer", max_count=3)
        for pdf_path in files:
            text = extractor.extract(pdf_path)
            assert isinstance(text, str)


class TestPDFExtractorColumnDetection:
    """Unit tests for internal column detection logic."""

    @pytest.fixture
    def extractor(self) -> PDFExtractor:
        return PDFExtractor(
            min_gutter_width=12.0,
            min_column_width_ratio=0.20,
        )

    def test_customizable_parameters(self):
        extractor = PDFExtractor(
            min_gutter_width=20.0,
            min_column_width_ratio=0.30,
            header_height_ratio_cap=0.25,
            footer_height_ratio_cap=0.05,
        )
        assert extractor.min_gutter_width == 20.0
        assert extractor.min_column_width_ratio == 0.30
        assert extractor.header_height_ratio_cap == 0.25
        assert extractor.footer_height_ratio_cap == 0.05

    def test_crop_extract_clamps_out_of_bounds(self, extractor):
        """_crop_extract should not raise even with slightly out-of-bounds coords."""
        import pdfplumber
        files = _get_pdf_files(max_count=1)
        if not files:
            pytest.skip("No PDF files available for this test.")
        with pdfplumber.open(str(files[0])) as pdf:
            page = pdf.pages[0]
            # Intentionally out-of-bounds crop — should not raise
            result = extractor._crop_extract(
                page,
                x0=-10.0,
                top=-10.0,
                x1=page.width + 50.0,
                bottom=page.height + 50.0,
            )
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# DOCXExtractor Unit Tests
# ---------------------------------------------------------------------------

class TestDOCXExtractor:
    @pytest.fixture
    def extractor(self) -> DOCXExtractor:
        return DOCXExtractor()

    def test_raises_file_not_found(self, extractor):
        with pytest.raises(FileNotFoundError, match="not found"):
            extractor.extract("nonexistent_file_xyz.docx")

    def test_raises_value_error_on_invalid_file(self, extractor, tmp_path):
        bad_file = tmp_path / "fake.docx"
        bad_file.write_bytes(b"this is not a docx")
        with pytest.raises((ValueError, Exception)):
            extractor.extract(bad_file)

    @pytest.mark.skipif(
        not list(DATA_DIR.rglob("*.docx")),
        reason="No DOCX files found in data/raw/cvs/"
    )
    def test_smoke_extract_real_docx(self, extractor):
        """Smoke test: extract text from real DOCX files."""
        files = list(DATA_DIR.rglob("*.docx"))[:3]
        for docx_path in files:
            text = extractor.extract(docx_path)
            assert isinstance(text, str)
            assert len(text) > 10


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------

class TestExtractTextFromFile:
    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            extract_text_from_file("does_not_exist.pdf")

    def test_raises_on_unsupported_extension(self, tmp_path):
        txt_file = tmp_path / "resume.txt"
        txt_file.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_from_file(txt_file)

    def test_raises_on_xlsx_extension(self, tmp_path):
        xlsx_file = tmp_path / "resume.xlsx"
        xlsx_file.write_bytes(b"fake xlsx")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_from_file(xlsx_file)

    @pytest.mark.skipif(
        not list(DATA_DIR.rglob("*.pdf")),
        reason="No PDF files in dataset"
    )
    def test_factory_auto_detects_pdf(self):
        files = _get_pdf_files(max_count=1)
        if not files:
            pytest.skip("No PDFs available")
        text = extract_text_from_file(files[0])
        assert isinstance(text, str)

    @pytest.mark.skipif(
        not list(DATA_DIR.rglob("*.docx")),
        reason="No DOCX files in dataset"
    )
    def test_factory_auto_detects_docx(self):
        files = list(DATA_DIR.rglob("*.docx"))[:1]
        if not files:
            pytest.skip("No DOCX files available")
        text = extract_text_from_file(files[0])
        assert isinstance(text, str)
