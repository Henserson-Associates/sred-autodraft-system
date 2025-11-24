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
      "industry": "pharmacy",
      "tech_code": "01.01",
      "project_description": "Brief project summary"
    }
    ```
  - Response:
    ```json
    {
      "sections": {
        "uncertainty": "...",
        "systematic_investigation": "...",
        "technological_advancement": "..."
      }
    }
    ```

## Data Prep Notes

- `scripts/parse_markdown_reports.py` parses markdown reports into `data/processed/approved_sections.jsonl`.
- `scripts/ingest.py` embeds those sections and stores them in `chroma_db/`.
- `agents.py` wires retrieval + generation; `llm_client.py` wraps the OpenAI client.

## Environment

- Required env vars: `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`).
- Default model: `gpt-4o-mini`.

## Future Ideas

- Add rejection classifier and iterative reviewer agent.
- Expand UI for bulk generation and downloads.
- Swap in self-hosted or fine-tuned models to reduce cost.
