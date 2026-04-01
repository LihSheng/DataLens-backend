"""
PDF exporter — converts Markdown conversation content to a PDF bytes payload.

Uses WeasyPrint (preferred) or a fallback HTML → pdfkit approach.
Both require external binary dependencies; this module gracefully degrades
to raising an informative ImportError when neither is available.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── WeasyPrint path ────────────────────────────────────────────────────────────

def _weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401
        return True
    except ImportError:
        return False


def _pdfkit_available() -> bool:
    try:
        import pdfkit  # noqa: F401
        return True
    except ImportError:
        return False


# ── HTML template ─────────────────────────────────────────────────────────────

_MARKDOWN_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 40px auto;
    padding: 0 20px;
}
h1 { font-size: 1.8em; border-bottom: 2px solid #eee; padding-bottom: 8px; }
h2 { font-size: 1.4em; }
h3 { font-size: 1.1em; color: #555; }
small { color: #888; }
hr { border: none; border-top: 1px solid #eee; margin: 20px 0; }
details { background: #f9f9f9; padding: 8px; border-radius: 4px; }
pre { background: #f5f5f5; padding: 12px; overflow-x: auto; border-radius: 4px; }
code { background: #f5f5f5; padding: 1px 4px; border-radius: 2px; }
"""


def _markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown_text to HTML using the `markdown` library if available.
    """
    try:
        import markdown
        md = markdown.Markdown(extensions=["extra", "tables"])
        body_html = md.convert(markdown_text)
    except ImportError:
        # Degrade: escape and wrap as preformatted text
        body_html = f"<pre>{markdown_text}</pre>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>{_MARKDOWN_CSS}</style>
</head>
<body>{body_html}</body>
</html>"""


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    """
    Convert a Markdown string to a PDF as bytes.

    Attempts WeasyPrint first, then pdfkit, then raises a clear error.

    Args:
        markdown_text: The conversation rendered as Markdown.

    Returns:
        Raw PDF bytes.

    Raises:
        ImportError: If neither WeasyPrint nor pdfkit is installed.
    """
    html = _markdown_to_html(markdown_text)

    if _weasyprint_available():
        try:
            import weasyprint
            wp = weasyprint.HTML(string=html)
            return bytes(wp.write_pdf())
        except Exception as exc:
            logger.warning(f"[export/pdf] WeasyPrint failed: {exc}, trying pdfkit")
            # Fall through to pdfkit

    if _pdfkit_available():
        try:
            import pdfkit
            options = {"quiet": "", "no-outline": None}
            return pdfkit.from_string(markdown_text, False, options=options)
        except Exception as exc:
            logger.error(f"[export/pdf] pdfkit failed: {exc}")
            raise ImportError(
                "PDF generation failed. Install weasyprint "
                "(`pip install weasyprint`) and its system dependencies, "
                "or ensure wkhtmltopdf is on your PATH."
            ) from exc

    raise ImportError(
        "PDF export requires weasyprint or pdfkit. "
        "Install weasyprint with: pip install weasyprint "
        "(also requires: apt-get install python3-cffi libcairo2 libpango1.0-0 libgdk-pixbuf2.0 libffi-dev shared-mime-info) "
        "Or: pip install pdfkit && apt-get install wkhtmltopdf"
    )


async def markdown_to_pdf_bytes_async(markdown_text: str) -> bytes:
    """Async wrapper — runs the blocking PDF conversion in a thread pool."""
    import asyncio
    return await asyncio.to_thread(markdown_to_pdf_bytes, markdown_text)
