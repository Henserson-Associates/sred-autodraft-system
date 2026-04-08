# SR&ED Report Generator

Generate SR&ED draft sections using retrieval-augmented LLMs. The system exposes a FastAPI service for report generation and ships with a lightweight React page (served statically) to exercise the endpoint.

## Overview

- Uses Chroma for semantic retrieval of approved SR&ED sections.
- Generates three CRA-aligned sections: Technological Uncertainty, Systematic Investigation, Technological Advancement.
- Wraps generation behind a REST API (`/api/generate`) and serves a simple in-browser client at `/`.

## Tech Stack

| Layer              | Technology                              |
|--------------------|-----------------------------------------|
| API                | FastAPI + Uvicorn                       |
| Retrieval          | ChromaDB + Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM                | OpenAI Chat Completions (model via `OPENAI_MODEL`) |
| Frontend           | React (CDN), static files served by FastAPI |
| Utilities          | python-docx / ReportLab for formatting (future use) |

## Setup

1. Install Python 3.12+.  
2. Install deps: `pip install -r requirements.txt`.  
3. Place approved/rejected markdown reports under `data/approved/` and `data/rejected/`.  
4. Build the vector store: `python scripts/ingest.py` (writes `chroma_db/`).  
5. Start the API + frontend: `uvicorn app:app --reload` (API under `/api`, UI at `/`).  
6. Open `http://localhost:8000/`, fill the form, and click **Generate Draft**.

## API

- `GET /api/health` → `{ "status": "ok" }`
- `POST /api/generate`
  - Request body:
    ```json
    {
      "transcript": "Full meeting transcript",
      "website_url": "https://example.com",
      "supplementary_docs": [
        { "name": "notes.txt", "content": "..." }
      ],
      "review": false,
      "max_revisions": 3,
      "time_budget_seconds": 55
    }
    ```

- `POST /api/generate-docx` (same request body as `/api/generate`) â†’ returns a downloadable `.docx` rendered from `templates/sred_report_template.docx`.
- `POST /api/revise-section` â†’ revise a single section with user instructions while keeping CRA word limits.
- `POST /api/render-docx` â†’ render a `.docx` from the current `project_title`, `project_summary`, and `sections` (does not re-run the LLM).
  - Response:
    ```json
    {
      "project_title": "…",
      "project_summary": "…",
      "sections": {
        "uncertainty": "...",
        "systematic_investigation": "...",
        "technological_advancement": "..."
      },
      "meta": { "approved": true }
    }
    ```

## Data Prep Notes

- `scripts/parse_markdown_reports.py` parses markdown reports into `data/processed/approved_sections.jsonl`.
- `scripts/ingest.py` embeds those sections and stores them in `chroma_db/`.
- `agents.py` wires retrieval + generation; `llm_client.py` wraps the OpenAI client.

## Environment

- Required env vars: `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`).
- Default model: `gpt-5.4`.
- Optional: `SRED_TIME_BUDGET_SECONDS` (defaults to 55) to keep requests under common reverse-proxy timeouts.
- Optional: `SRED_MAX_COMPLETION_TOKENS_<CONTENT_TYPE>` to cap model output tokens per section (e.g. `SRED_MAX_COMPLETION_TOKENS_SYSTEMATIC_INVESTIGATION=1700`).

## Future Ideas

- Add rejection classifier and iterative reviewer agent.
- Expand UI for bulk generation and downloads.
- Swap in self-hosted or fine-tuned models to reduce cost.
