from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict

from docxtpl import DocxTemplate


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _safe_filename(stem: str, *, default: str = "sred_report") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (stem or "").strip()).strip("._-")
    if not cleaned:
        cleaned = default
    return cleaned[:120]


def render_report_docx(*, template_path: Path, context: Dict[str, Any]) -> bytes:
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    if template_path.suffix.lower() != ".docx":
        raise ValueError(f"Template must be a .docx file: {template_path}")

    doc = DocxTemplate(str(template_path))
    doc.render(context)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def suggested_report_filename(project_title: str) -> str:
    return _safe_filename(project_title) + ".docx"

