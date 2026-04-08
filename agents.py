from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from llm_client import LLMClient


logger = logging.getLogger("sred_app.agents")

MAX_REVISIONS = 3
DEFAULT_TIME_BUDGET_SECONDS = int(os.getenv("SRED_TIME_BUDGET_SECONDS", "55"))

WORD_LIMITS = {
    "uncertainty": (300, 350),
    "systematic_investigation": (700, 750),
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
        self.llm = LLMClient(model="gpt-5.4")

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
        self.llm = LLMClient(model="gpt-5.4")

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


class TitleAgent:
    """
    Generates a validated project title (strictly fewer than 70 characters)
    from a ProjectBrief using the check_content tool.
    """

    def __init__(self) -> None:
        self.llm = LLMClient(model="gpt-5.4")

    def generate(self, brief: ProjectBrief) -> str:
        logger.info("TitleAgent: generating title (initial='%s')", brief.title)
        title = self.llm.generate_title(brief)
        logger.info("TitleAgent: final title='%s' (%d chars)", title, len(title))
        return title


class ReviewerAgent:
    """
    Reviews all three SR&ED sections holistically against CRA criteria.
    Returns (approved, per_section_feedback).
    """

    def __init__(self) -> None:
        self.llm = LLMClient(model="gpt-5.4")

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
        self.title_agent = TitleAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

    def run(
        self,
        transcript: str,
        website_url: str,
        supplementary_docs: list[dict] | None = None,
        review: bool = True,
        max_revisions: int | None = None,
        time_budget_seconds: int | None = None,
    ) -> Dict:
        started = time.monotonic()
        budget = DEFAULT_TIME_BUDGET_SECONDS if time_budget_seconds is None else int(time_budget_seconds)
        deadline = (started + budget) if budget > 0 else None

        def time_left_seconds() -> float | None:
            if deadline is None:
                return None
            return deadline - time.monotonic()

        def out_of_time() -> bool:
            remaining = time_left_seconds()
            return remaining is not None and remaining <= 0

        # Step 1 — Research
        logger.info("Orchestrator: Step 1 — Research")
        brief = self.researcher.analyze(transcript, website_url, supplementary_docs or [])
        logger.info("Orchestrator: selected project '%s'", brief.title)
        if out_of_time():
            logger.warning("Orchestrator: time budget exceeded after research; returning partial draft")
            return {
                "project_title": brief.title,
                "project_summary": brief.selection_rationale,
                "sections": {},
                "meta": {
                    "timed_out": True,
                    "time_budget_seconds": budget,
                    "review_enabled": False,
                    "review_attempts": 0,
                    "revisions_applied": 0,
                },
            }

        # Step 2 — Generate validated project title (< 70 characters)
        logger.info("Orchestrator: Step 2 — Generate title")
        project_title = self.title_agent.generate(brief)
        if out_of_time():
            logger.warning("Orchestrator: time budget exceeded after title; returning partial draft")
            return {
                "project_title": project_title,
                "project_summary": brief.selection_rationale,
                "sections": {},
                "meta": {
                    "timed_out": True,
                    "time_budget_seconds": budget,
                    "review_enabled": False,
                    "review_attempts": 0,
                    "revisions_applied": 0,
                },
            }

        # Step 3 — Initial draft
        logger.info("Orchestrator: Step 3 — Initial write")
        sections = self.writer.write_all(brief)
        if out_of_time():
            logger.warning("Orchestrator: time budget exceeded after initial write; skipping review")
            return {
                "project_title": project_title,
                "project_summary": brief.selection_rationale,
                "sections": sections,
                "meta": {
                    "timed_out": True,
                    "time_budget_seconds": budget,
                    "review_enabled": False,
                    "review_attempts": 0,
                    "revisions_applied": 0,
                },
            }

        effective_max_revisions = MAX_REVISIONS if max_revisions is None else max(0, int(max_revisions))
        if not review or effective_max_revisions == 0:
            logger.info("Orchestrator: review disabled; returning initial draft")
            return {
                "project_title": project_title,
                "project_summary": brief.selection_rationale,
                "sections": sections,
                "meta": {
                    "timed_out": False,
                    "time_budget_seconds": budget,
                    "review_enabled": False,
                    "review_attempts": 0,
                    "revisions_applied": 0,
                },
            }

        # Step 4 — Review / revise loop
        review_attempts = 0
        revisions_applied = 0
        approved = False
        for attempt in range(1, effective_max_revisions + 1):
            if out_of_time():
                logger.warning("Orchestrator: time budget exceeded before review; returning best draft")
                break

            remaining = time_left_seconds()
            if remaining is not None and remaining < 8:
                logger.warning(
                    "Orchestrator: only %.1fs left; skipping further review calls to avoid request timeout",
                    remaining,
                )
                break

            logger.info("Orchestrator: Step 4 — Review attempt %d/%d", attempt, effective_max_revisions)
            approved, feedback = self.reviewer.review(brief, sections)
            review_attempts += 1

            if approved:
                logger.info("Orchestrator: report approved on attempt %d", attempt)
                break

            # Revise only the sections that have feedback
            needs_revision = {k: v for k, v in feedback.items() if v}
            logger.info("Orchestrator: revising sections %s", list(needs_revision.keys()))
            for key, note in needs_revision.items():
                if out_of_time():
                    logger.warning("Orchestrator: time budget exceeded during revisions; returning best draft")
                    break
                if key in sections:
                    sections[key] = self.writer.revise_section(key, brief, sections[key], note)
                    revisions_applied += 1

            if attempt == effective_max_revisions:
                logger.warning("Orchestrator: max revisions reached; returning best available draft")

        return {
            "project_title": project_title,
            "project_summary": brief.selection_rationale,
            "sections": sections,
            "meta": {
                "timed_out": out_of_time(),
                "time_budget_seconds": budget,
                "review_enabled": True,
                "review_attempts": review_attempts,
                "revisions_applied": revisions_applied,
                "approved": approved,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        }
