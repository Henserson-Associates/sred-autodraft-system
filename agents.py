from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from llm_client import LLMClient


logger = logging.getLogger("sred_app.agents")

MAX_REVISIONS = 3

WORD_LIMITS = {
    "uncertainty": (300, 350),
    "systematic_investigation": (650, 700),
    "technological_advancement": (300, 350),
}


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class ProjectBrief:
    title: str
    industry: str
    tech_domain: str
    company_background: str
    technical_work: str
    technological_uncertainty: str
    investigation_approach: str
    potential_advancement: str
    selection_rationale: str


# ─── Website fetcher ─────────────────────────────────────────────────────────

def _fetch_website_text(url: str, timeout: int = 10, max_chars: int = 5000) -> str:
    """Fetch and extract readable text content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SR-ED-Agent/2.0)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        # Prefer main content areas if present
        main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body
        text = (main or soup).get_text(separator=" ", strip=True)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]

    except Exception as exc:
        logger.warning("Failed to fetch website %s: %s", url, exc)
        return ""


# ─── Agents ──────────────────────────────────────────────────────────────────

class ResearchAgent:
    """
    Analyzes the meeting transcript and company website to identify
    and select the single strongest SR&ED-eligible project.
    """

    def __init__(self) -> None:
        self.llm = LLMClient(model="gpt-4o")

    def analyze(
        self,
        transcript: str,
        website_url: str,
        supplementary_docs: list[dict] | None = None,
    ) -> ProjectBrief:
        logger.info("ResearchAgent: fetching website %s", website_url)
        website_text = _fetch_website_text(website_url)
        if not website_text:
            logger.warning("ResearchAgent: no website content retrieved; using transcript only")

        docs = supplementary_docs or []
        logger.info(
            "ResearchAgent: calling LLM (transcript_len=%d website_len=%d docs=%d)",
            len(transcript), len(website_text), len(docs),
        )
        raw = self.llm.research(
            transcript=transcript,
            website_text=website_text,
            supplementary_docs=docs,
        )

        return ProjectBrief(
            title=raw.get("title", "SR&ED Project"),
            industry=raw.get("industry", "Technology"),
            tech_domain=raw.get("tech_domain", ""),
            company_background=raw.get("company_background", ""),
            technical_work=raw.get("technical_work", ""),
            technological_uncertainty=raw.get("technological_uncertainty", ""),
            investigation_approach=raw.get("investigation_approach", ""),
            potential_advancement=raw.get("potential_advancement", ""),
            selection_rationale=raw.get("selection_rationale", ""),
        )


class WriterAgent:
    """
    Generates the three SR&ED report sections (Lines 242, 244, 246)
    from a ProjectBrief.
    """

    def __init__(self) -> None:
        self.llm = LLMClient(model="gpt-4o")

    def write_all(self, brief: ProjectBrief) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        for key in ["uncertainty", "systematic_investigation", "technological_advancement"]:
            min_w, max_w = WORD_LIMITS[key]
            logger.info("WriterAgent: writing section '%s'", key)
            sections[key] = self.llm.write_section(
                section_key=key,
                brief=brief,
                min_words=min_w,
                max_words=max_w,
            )
        return sections

    def revise_section(
        self,
        key: str,
        brief: ProjectBrief,
        draft: str,
        feedback: str,
    ) -> str:
        min_w, max_w = WORD_LIMITS[key]
        logger.info("WriterAgent: revising section '%s'", key)
        return self.llm.revise_section(
            section_key=key,
            brief=brief,
            draft=draft,
            feedback=feedback,
            min_words=min_w,
            max_words=max_w,
        )


class ReviewerAgent:
    """
    Reviews all three SR&ED sections holistically against CRA criteria.
    Returns (approved, per_section_feedback).
    """

    def __init__(self) -> None:
        self.llm = LLMClient(model="gpt-4o")

    def review(
        self,
        brief: ProjectBrief,
        sections: Dict[str, str],
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        logger.info("ReviewerAgent: reviewing report for '%s'", brief.title)
        result = self.llm.review_report(brief=brief, sections=sections)
        approved: bool = result.get("approved", False)
        feedback: Dict[str, Optional[str]] = result.get("feedback", {})
        overall = result.get("overall_notes", "")
        logger.info("ReviewerAgent: approved=%s overall='%s'", approved, overall)
        return approved, feedback


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class ReportOrchestrator:
    """
    Orchestrates the full pipeline:
      ResearchAgent → WriterAgent → ReviewerAgent loop (max MAX_REVISIONS)
    """

    def __init__(self) -> None:
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

    def run(
        self,
        transcript: str,
        website_url: str,
        supplementary_docs: list[dict] | None = None,
    ) -> Dict:
        # Step 1 — Research
        logger.info("Orchestrator: Step 1 — Research")
        brief = self.researcher.analyze(transcript, website_url, supplementary_docs or [])
        logger.info("Orchestrator: selected project '%s'", brief.title)

        # Step 2 — Initial draft
        logger.info("Orchestrator: Step 2 — Initial write")
        sections = self.writer.write_all(brief)

        # Step 3 — Review / revise loop
        for attempt in range(1, MAX_REVISIONS + 1):
            logger.info("Orchestrator: Step 3 — Review attempt %d/%d", attempt, MAX_REVISIONS)
            approved, feedback = self.reviewer.review(brief, sections)

            if approved:
                logger.info("Orchestrator: report approved on attempt %d", attempt)
                break

            # Revise only the sections that have feedback
            needs_revision = {k: v for k, v in feedback.items() if v}
            logger.info("Orchestrator: revising sections %s", list(needs_revision.keys()))
            for key, note in needs_revision.items():
                if key in sections:
                    sections[key] = self.writer.revise_section(key, brief, sections[key], note)

            if attempt == MAX_REVISIONS:
                logger.warning("Orchestrator: max revisions reached; returning best available draft")

        return {
            "project_title": brief.title,
            "project_summary": brief.selection_rationale,
            "sections": sections,
        }
