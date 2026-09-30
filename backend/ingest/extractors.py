"""
backend/ingest/extractors.py — Get text out of uploaded files
==============================================================================

SUPPORTED INPUTS
    .pdf                 text-layer PDFs (email exports, transcripts, reports)
    .pdf (scanned)       falls back to OCR automatically
    .png .jpg .jpeg      screenshots of WhatsApp / Slack / Teams conversations
    .webp .bmp .tiff     other common screenshot formats
    .txt .md .csv        plain text

WHY THE PDF PATH TRIES TWICE
    A PDF either carries a text layer or it doesn't. An email exported from
    Outlook does; a photographed or scanned document does not, and pdfplumber
    returns an empty string for it — silently, with no error.

    Rather than showing the user "0 messages found" for a file that visibly
    contains text, the extractor measures how much text it recovered per page
    and automatically re-runs those pages through OCR. The user is told which
    method was used, because OCR output is materially less reliable and that
    affects how much weight the resulting analysis deserves.

WHY IMAGES MATTER HERE
    Workplace conversation increasingly lives in chat, and the realistic way a
    person shares it is a screenshot. Supporting images means the system can
    analyse the communication people actually have, not only the formal email
    they can export.

OCR ACCURACY, HONESTLY
    Tesseract on a clean phone screenshot is good but not perfect. It confuses
    l/1/I and O/0, drops emoji, and can mangle names. Since downstream
    sentiment and NRC lexicon matching are word-based, a mis-OCR'd word is
    simply not counted — it degrades quietly rather than corrupting results.
    Every extraction reports a confidence signal so the UI can warn the user.
==============================================================================
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

#: Extensions accepted by the upload endpoint.
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS

#: Below this many characters, a PDF page is assumed to have no text layer.
#: A genuine text page of an email carries hundreds of characters; a scanned
#: page typically yields a handful of stray marks or nothing at all.
_MIN_CHARS_PER_PAGE = 40

#: Upload ceiling. Large enough for a long email export, small enough that a
#: mis-drag of a video file fails fast instead of consuming server memory.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class ExtractionError(RuntimeError):
    """Raised when a file cannot be read at all."""


@dataclass
class ExtractionResult:
    """Text recovered from one uploaded file, plus how it was obtained."""

    text: str
    source_name: str
    method: str                       # "pdf-text" | "pdf-ocr" | "image-ocr" | "plain-text"
    page_count: int = 0
    ocr_used: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def describe(self) -> dict:
        return {
            "source": self.source_name,
            "method": self.method,
            "pages": self.page_count,
            "characters": self.char_count,
            "ocr_used": self.ocr_used,
            "warnings": self.warnings,
        }


# ==============================================================================
# Dependency probes
# ==============================================================================
def _require_pdfplumber():
    try:
        import pdfplumber

        return pdfplumber
    except ImportError as exc:
        raise ExtractionError(
            "pdfplumber is not installed, so PDF files cannot be read.\n"
            "  Fix: pip install -r backend/requirements.txt"
        ) from exc


def ocr_available() -> Tuple[bool, Optional[str]]:
    """Check whether OCR can run. Returns ``(available, reason_if_not)``.

    Two separate things must be present: the Python wrapper (`pytesseract`)
    and the Tesseract *binary*, which pip cannot install. Reporting them
    separately is what lets the error message name the right fix instead of a
    generic "OCR failed".
    """
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        return False, (
            "pytesseract/Pillow not installed. Fix: "
            "pip install -r backend/requirements.txt"
        )

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False, (
            "The Tesseract engine is not installed (pytesseract is only a "
            "wrapper).\n"
            "  macOS:   brew install tesseract\n"
            "  Ubuntu:  sudo apt install tesseract-ocr\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    return True, None


# ==============================================================================
# Image OCR
# ==============================================================================
def _preprocess_for_ocr(image):
    """Improve OCR accuracy on chat screenshots.

    Three cheap transforms, each targeting a real failure mode:

      Greyscale   — chat bubbles are coloured; colour carries no textual
                    information and Tesseract handles single-channel better.
      Upscale     — phone screenshots are often rendered small. Tesseract is
                    trained around 300 DPI and degrades badly below roughly
                    20px character height, so anything narrow is doubled.
      Autocontrast— dark-mode chats are light text on dark backgrounds, which
                    Tesseract handles poorly. Normalising the range helps both
                    polarities.
    """
    from PIL import Image, ImageOps

    image = image.convert("L")

    if image.width < 1000:
        scale = max(2, 1000 // max(image.width, 1))
        image = image.resize(
            (image.width * scale, image.height * scale), Image.LANCZOS
        )

    return ImageOps.autocontrast(image)


def extract_from_image(data: bytes, source_name: str = "image") -> ExtractionResult:
    """OCR an image of a conversation."""
    available, reason = ocr_available()
    if not available:
        raise ExtractionError(f"Cannot read image files. {reason}")

    import pytesseract
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(
            f"'{source_name}' could not be opened as an image: {exc}"
        ) from exc

    warnings: List[str] = []
    if image.width < 400 or image.height < 400:
        warnings.append(
            f"Image is small ({image.width}x{image.height}px). OCR accuracy "
            f"improves considerably above 800px wide."
        )

    try:
        # --psm 6: "assume a single uniform block of text". Chat screenshots are
        # columns of short lines, which the default page-segmentation mode
        # frequently misreads as multi-column layout and interleaves.
        text = pytesseract.image_to_string(
            _preprocess_for_ocr(image), config="--psm 6"
        )
    except Exception as exc:
        raise ExtractionError(f"OCR failed on '{source_name}': {exc}") from exc

    if not text.strip():
        warnings.append(
            "No text was recognised. If this is a screenshot, check it is not "
            "blurred or heavily cropped."
        )

    return ExtractionResult(
        text=_tidy(text),
        source_name=source_name,
        method="image-ocr",
        page_count=1,
        ocr_used=True,
        warnings=warnings,
    )


# ==============================================================================
# PDF
# ==============================================================================
def _ocr_pdf_page(page, page_number: int) -> str:
    """Render one PDF page to an image and OCR it."""
    available, _ = ocr_available()
    if not available:
        return ""

    import pytesseract

    try:
        # 200 DPI: below this Tesseract loses small type, above it the render
        # cost rises sharply for no accuracy gain on document scans.
        image = page.to_image(resolution=200).original
        return pytesseract.image_to_string(_preprocess_for_ocr(image), config="--psm 6")
    except Exception as exc:
        logger.warning("OCR failed on page %d: %s", page_number, exc)
        return ""


def extract_from_pdf(data: bytes, source_name: str = "document.pdf") -> ExtractionResult:
    """Extract text from a PDF, falling back to OCR per page where needed."""
    pdfplumber = _require_pdfplumber()

    warnings: List[str] = []
    pages_text: List[str] = []
    ocr_pages: List[int] = []

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            page_count = len(pdf.pages)

            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                # A page with almost no characters has no text layer. Re-run it
                # through OCR rather than silently dropping its content.
                if len(text.strip()) < _MIN_CHARS_PER_PAGE:
                    ocr_text = _ocr_pdf_page(page, index)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        ocr_pages.append(index)

                pages_text.append(text)

    except Exception as exc:
        raise ExtractionError(
            f"'{source_name}' could not be read as a PDF: {exc}\n"
            f"  If the file is password-protected, remove the password and "
            f"re-upload."
        ) from exc

    if ocr_pages:
        available, reason = ocr_available()
        if available:
            warnings.append(
                f"{len(ocr_pages)} of {page_count} page(s) had no text layer "
                f"and were read with OCR (pages {', '.join(map(str, ocr_pages[:8]))}"
                f"{'...' if len(ocr_pages) > 8 else ''}). OCR text is less "
                f"reliable than extracted text."
            )
        else:
            warnings.append(
                f"{len(ocr_pages)} page(s) appear to be scanned images with no "
                f"text layer, and OCR is unavailable. {reason}"
            )

    combined = _tidy("\n\n".join(pages_text))
    if not combined.strip():
        warnings.append(
            "No text could be extracted. The PDF may be a scan without OCR "
            "available, or may contain only images."
        )

    return ExtractionResult(
        text=combined,
        source_name=source_name,
        method="pdf-ocr" if ocr_pages else "pdf-text",
        page_count=page_count,
        ocr_used=bool(ocr_pages),
        warnings=warnings,
    )


# ==============================================================================
# Plain text
# ==============================================================================
def extract_from_text(data: bytes, source_name: str = "file.txt") -> ExtractionResult:
    """Decode a plain-text upload.

    Tries UTF-8 first, then Windows-1252, which is what Outlook exports and
    Windows-authored transcripts commonly use. Falls back to lossy UTF-8 rather
    than failing, so one bad byte never costs the user their whole document.
    """
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return ExtractionResult(
                text=_tidy(data.decode(encoding)),
                source_name=source_name,
                method="plain-text",
                page_count=1,
            )
        except UnicodeDecodeError:
            continue

    return ExtractionResult(
        text=_tidy(data.decode("utf-8", errors="replace")),
        source_name=source_name,
        method="plain-text",
        page_count=1,
        warnings=["Some characters could not be decoded and were replaced."],
    )


# ==============================================================================
# Dispatch
# ==============================================================================
def extract(data: bytes, filename: str) -> ExtractionResult:
    """Extract text from an uploaded file, choosing the method by extension.

    Args:
        data: Raw file bytes.
        filename: Original filename, used to pick the extractor.

    Raises:
        ExtractionError: unsupported type, oversized, empty, or unreadable.
    """
    if not data:
        raise ExtractionError(f"'{filename}' is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise ExtractionError(
            f"'{filename}' is {len(data) / 1e6:.1f} MB, above the "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit."
        )

    suffix = Path(filename).suffix.lower()

    if suffix in PDF_EXTENSIONS:
        return extract_from_pdf(data, filename)
    if suffix in IMAGE_EXTENSIONS:
        return extract_from_image(data, filename)
    if suffix in TEXT_EXTENSIONS:
        return extract_from_text(data, filename)

    raise ExtractionError(
        f"'{filename}' has an unsupported type ('{suffix or 'none'}').\n"
        f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


# ==============================================================================
# Cleaning
# ==============================================================================
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl",
}


def _tidy(text: str) -> str:
    """Normalise extracted text without changing its meaning.

    PDF extraction introduces artefacts that would otherwise corrupt every
    downstream word-based step:

      - Typographic ligatures ("ﬁ") that break lexicon lookups for words like
        "difficult" and "efficiency" — both directly relevant to burnout.
      - Smart quotes and en/em dashes, which split words during tokenisation.
      - Soft hyphens and zero-width spaces, invisible but word-breaking.
      - Runs of blank lines from page boundaries.

    Deliberately does NOT lowercase or strip punctuation: `uppercase_ratio`
    and `exclamation_count` are real behavioural features, and destroying them
    here would silently zero out part of the burnout score.
    """
    if not text:
        return ""

    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)

    text = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
        .replace(" ", " ").replace("​", "").replace("­", "")
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


__all__ = [
    "ExtractionResult",
    "ExtractionError",
    "extract",
    "extract_from_pdf",
    "extract_from_image",
    "extract_from_text",
    "ocr_available",
    "SUPPORTED_EXTENSIONS",
    "MAX_UPLOAD_BYTES",
]
