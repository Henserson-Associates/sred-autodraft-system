from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import ReportGenerator


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="SR&ED Report Generator API",
    description="Generate SR&ED draft sections based on project inputs.",
    version="0.1.0",
)

generator = ReportGenerator()


class ReportRequest(BaseModel):
    industry: str = Field("", description="Industry or domain, e.g., pharmacy.")
    tech_code: str = Field("", description="Technology code, e.g., 01.01.")
    project_description: str = Field(..., min_length=10, description="Brief project summary.")


class ReportResponse(BaseModel):
    sections: Dict[str, str]


@app.get("/api/health", response_class=JSONResponse, tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate", response_model=ReportResponse, tags=["reports"])
def generate_report(request: ReportRequest) -> ReportResponse:
    if not request.project_description.strip():
        raise HTTPException(status_code=400, detail="Project description is required.")

    sections = generator.generate_report(
        industry=request.industry.strip(),
        tech_code=request.tech_code.strip(),
        project_description=request.project_description.strip(),
    )

    return ReportResponse(sections=sections)


if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
