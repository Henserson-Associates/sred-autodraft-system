from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import ReportOrchestrator


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

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


class ReportRequest(BaseModel):
    transcript: str = Field(..., description="Full meeting transcript with the client.")
    website_url: str = Field(..., description="Client company website URL.")


class ReportResponse(BaseModel):
    project_title: str
    project_summary: str
    sections: Dict[str, str]


@app.get("/api/health", response_class=JSONResponse, tags=["system"])
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

    result = orchestrator.run(
        transcript=request.transcript.strip(),
        website_url=request.website_url.strip(),
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


if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
