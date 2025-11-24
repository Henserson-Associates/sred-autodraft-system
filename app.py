from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import ReportGenerator


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sred_app")

app = FastAPI(
    title="SR&ED Report Generator API",
    description="Generate SR&ED draft sections based on project inputs.",
    version="0.1.0",
)

generator = ReportGenerator()


class ReportRequest(BaseModel):
    industry: str = Field(..., description="Industry or domain, e.g., pharmacy.")
    tech_code: str | None = Field(None, description="Technology code, e.g., 01.01.")
    project_description: str | None = Field(None, description="Brief project summary.")
    company_summary: str | None = Field(None, description="Short summary about the company.")


class ReportResponse(BaseModel):
    sections: Dict[str, str]


@app.get("/api/health", response_class=JSONResponse, tags=["system"])
def health() -> Dict[str, str]:
    logger.info("Health check called")
    return {"status": "ok"}


@app.post("/api/generate", response_model=ReportResponse, tags=["reports"])
def generate_report(request: ReportRequest) -> ReportResponse:
    if not (request.industry and request.industry.strip()):
        logger.warning("Rejected request: missing industry")
        raise HTTPException(status_code=400, detail="Industry is required.")

    tech_code = request.tech_code.strip() if request.tech_code else None
    project_description = request.project_description.strip() if request.project_description else None
    company_summary = request.company_summary.strip() if request.company_summary else None

    logger.info(
        "Generating report industry=%s tech_code=%s description_len=%d summary_len=%d",
        request.industry.strip(),
        tech_code or "N/A",
        len(project_description or ""),
        len(company_summary or ""),
    )

    sections = generator.generate_report(
        industry=request.industry.strip(),
        tech_code=tech_code,
        project_description=project_description,
        company_summary=company_summary,
    )

    logger.info(
        "Generated sections lengths=%s",
        {name: len(text or "") for name, text in sections.items()},
    )

    return ReportResponse(sections=sections)


if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
