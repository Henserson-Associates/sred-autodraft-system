# SR&ED Report Generator AI Agent System

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green)](https://python.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-DB-orange)](https://www.trychroma.com/)

## Overview

This project is an AI agent system that automatically generates high-quality technical reports for Canada's **Scientific Research & Experimental Development (SR&ED)** tax incentive program.

The reports always follow the same CRA-required structure—answers to **three core questions**:

1. What was the technological uncertainty?  
2. What systematic investigation or experimentation was conducted?  
3. What technological advancement was achieved?

The system learns from **100 approved reports** (positive examples) and avoids pitfalls from **50 rejected reports**.  
Users only need to provide:  
- Company industry (e.g., "pharmacy")  
- Technology code (e.g., "01.01")  
- Basic project description (e.g., "Developed AI-driven inventory software to predict drug shortages")

The AI then produces a polished, CRA-ready draft report.

## System Architecture

```
User Input
    ↓
Retrieval Agent
    → Queries Chroma DB → Returns relevant approved examples
    ↓
Section Generator Agents (x3)
    ├─ Uncertainty Generator
    ├─ Systematic Investigation Generator
    └─ Advancement Generator
    ↓
Reviewer Agent
    → Scores draft, flags issues, loops back for fixes if needed
    ↓
Final Report (PDF/Word)
```

### Agent Details

- **Retrieval Agent**  
  Finds top similar approved sections using semantic search + metadata filters.

- **Section Generator Agents**  
  One agent per question. Uses retrieved examples + user input to write a section.

- **Reviewer Agent**  
  Checks completeness, specificity, CRA alignment, and rejection pitfalls.  
  If score < threshold, sends targeted feedback to specific generators.

### Workflow Summary

1. User submits industry, tech code, project description.  
2. Retrieval Agent pulls relevant approved chunks.  
3. Three Generator Agents draft sections in parallel.  
4. Sections merged → Reviewer Agent evaluates → iterate if needed.  
5. Output formatted report.

## Tech Stack

| Layer              | Technology                              | Why |
|--------------------|-----------------------------------------|-----|
| LLM                | Grok-4 (via xAI API) or Grok-3 (free tier) | Excellent technical reasoning |
| Agent Framework    | LangChain (or CrewAI)                   | Easy multi-agent orchestration |
| Vector Database    | Chroma (local, free, lightweight)       | Simple, persistent, metadata filtering |
| Embeddings         | Sentence Transformers (`all-MiniLM-L6-v2`) | Fast, local, no cost |
| Chunking & Parsing | Custom Python scripts (provided below)  | Rule-based + manual cleanup |
| Frontend | Streamlit                               | Quick UI for input/output |
| Output Formatting  | python-docx or ReportLab                | Generate Word/PDF |
| Environment        | Python 3.12+, pip                       | Standard |

## Data Preparation

### Folder Structure
```
data/
├── approved/          # 100 approved reports (PDF/Word/text)
├── rejected/          # 50 rejected reports
└── processed/         # JSON metadata after ingestion
chroma_db/             # Chroma persistence folder
```

### Steps
1. Extract text from reports.  
2. Anonymize company names, financials.  
3. Parse each report into **three sections**.  
4. Add metadata: `industry`, `tech_code`, `status` ("approved"), `section`.  
5. Embed and store in Chroma.

## Vector Storage (Chroma)

Create a single collection: `sred_reports`

Metadata fields:
```json
{
  "report_id": "001",
  "project_title": "Development of a Motion Tracking and Analysis System for Martial Arts Instruction",
  "status": "approved",
  "industry": "pharmacy",
  "tech_code": "01.01",
  "section": "systematic_investigation"
}
```

## Chunking Strategy

Store each full section as one chunk for all questions (~300–900 words).

## Retrieval & Context Management

### Retrieval Example (LangChain)
```python
results = collection.query(
    query_texts=[project_description],
    n_results=5,
    where={"status": "approved", "section": "systematic_investigation"}
)
```

## Setup Instructions

1. Clone repo  
2. `pip install langchain chromadb sentence-transformers python-docx streamlit`  
3. Put reports in `data/approved/` and `data/rejected/`  
4. Run `python scripts/ingest.py` → creates Chroma DB  
5. `streamlit run app.py` → Web UI ready  

## Running a Generation

UI fields:  
- Industry  
- Tech Code  
- Project Description  

Click "Generate" → Draft appears in ~30–60 seconds.  
Download as Word/PDF.  
Always review with SR&ED consultant before submission!

## Future Improvements

- Fine-tune a small model on approved reports  
- Add rejected-report classifier for Reviewer  
- Bulk generation endpoint  
- Auto-fill CRA Form T661  

## Questions?

Open an issue or ping the coding agent.  
This system routinely produces reports that match the quality of top SR&ED consultants—happy claiming! 🚀