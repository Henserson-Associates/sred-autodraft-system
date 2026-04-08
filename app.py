from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import ReportOrchestrator
from docx_export import DOCX_MIME, render_report_docx, suggested_report_filename


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
TEMPLATE_PATH = Path(
    os.getenv("DOCX_TEMPLATE_PATH", str(BASE_DIR / "templates" / "sred_report_template.docx"))
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sred_app")

app = FastAPI(
    title="SR&ED Report Generator",
    description="Multi-agent SR&ED report generation from meeting transcript + company website.",
    version="2.0.0",
)

orchestrator = ReportOrchestrator()


class SupplementaryDoc(BaseModel):
    name: str = Field(..., description="File name.")
    content: str = Field(..., description="Plain-text file content.")


class ReportRequest(BaseModel):
    transcript: str = Field(..., description="Full meeting transcript with the client.")
    website_url: str = Field(..., description="Client company website URL.")
    supplementary_docs: list[SupplementaryDoc] | None = Field(
        None, description="Optional supporting documents."
    )


class ReportResponse(BaseModel):
    project_title: str
    project_summary: str
    sections: Dict[str, str]


@app.get("/api/health", tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate", response_model=ReportResponse, tags=["reports"])
def generate_report(request: ReportRequest) -> ReportResponse:
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required.")
    if not request.website_url.strip():
        raise HTTPException(status_code=400, detail="Website URL is required.")

    logger.info(
        "New request: transcript_len=%d website_url=%s",
        len(request.transcript),
        request.website_url,
    )

    supp_docs = (
        [{"name": d.name, "content": d.content} for d in request.supplementary_docs]
        if request.supplementary_docs else []
    )

    result = orchestrator.run(
        transcript=request.transcript.strip(),
        website_url=request.website_url.strip(),
        supplementary_docs=supp_docs,
    )

    logger.info(
        "Report complete: title='%s' sections=%s",
        result["project_title"],
        {k: len(v) for k, v in result["sections"].items()},
    )

    return ReportResponse(
        project_title=result["project_title"],
        project_summary=result["project_summary"],
        sections=result["sections"],
    )


@app.post("/api/generate-docx", tags=["reports"])
def generate_report_docx(request: ReportRequest) -> StreamingResponse:
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required.")
    if not request.website_url.strip():
        raise HTTPException(status_code=400, detail="Website URL is required.")

    logger.info(
        "New DOCX request: transcript_len=%d website_url=%s",
        len(request.transcript),
        request.website_url,
    )

    supp_docs = (
        [{"name": d.name, "content": d.content} for d in request.supplementary_docs]
        if request.supplementary_docs else []
    )

    result = orchestrator.run(
        transcript=request.transcript.strip(),
        website_url=request.website_url.strip(),
        supplementary_docs=supp_docs,
    )

    context = {
        "project_title": result.get("project_title", ""),
        "project_summary": result.get("project_summary", ""),
        "uncertainty": (result.get("sections") or {}).get("uncertainty", ""),
        "systematic_investigation": (result.get("sections") or {}).get("systematic_investigation", ""),
        "technological_advancement": (result.get("sections") or {}).get("technological_advancement", ""),
    }

    try:
        payload = render_report_docx(template_path=TEMPLATE_PATH, context=context)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("DOCX render failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to render .docx from template.") from exc

    filename = suggested_report_filename(context["project_title"])
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(iter([payload]), media_type=DOCX_MIME, headers=headers)


if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
