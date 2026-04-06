from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:
    from agents import ProjectBrief

load_dotenv()

logger = logging.getLogger("sred_app.llm")


# ─── Content validation tool ─────────────────────────────────────────────────

CONTENT_LIMITS: Dict[str, dict] = {
    "title": {"type": "chars", "max": 69},
    "uncertainty": {"type": "words", "min": 300, "max": 350},
    "systematic_investigation": {"type": "words", "min": 700, "max": 750},
    "technological_advancement": {"type": "words", "min": 300, "max": 350},
}


def check_content(content: str, content_type: str) -> dict:
    """
    Tool: validate that generated content meets required word count (sections)
    or character limit (title).  Returns a dict with 'valid' (bool) and
    'feedback' (str | None) so the LLM agent can self-correct.
    """
    limits = CONTENT_LIMITS.get(content_type)
    if not limits:
        return {"valid": False, "feedback": f"Unknown content type: {content_type!r}"}

    text = content.strip()
    if limits["type"] == "chars":
        count = len(text)
        max_c = limits["max"]
        if count <= max_c:
            return {"valid": True, "character_count": count, "feedback": None}
        return {
            "valid": False,
            "character_count": count,
            "feedback": (
                f"Title is {count} characters but must be {max_c} or fewer. "
                f"Shorten by at least {count - max_c} character(s)."
            ),
        }
    else:
        words = len(text.split())
        min_w, max_w = limits["min"], limits["max"]
        if min_w <= words <= max_w:
            return {"valid": True, "word_count": words, "feedback": None}
        if words < min_w:
            return {
                "valid": False,
                "word_count": words,
                "feedback": (
                    f"Content has {words} words but requires {min_w}–{max_w}. "
                    f"Add approximately {min_w - words} more words."
                ),
            }
        return {
            "valid": False,
            "word_count": words,
            "feedback": (
                f"Content has {words} words but requires {min_w}–{max_w}. "
                f"Remove approximately {words - max_w} words."
            ),
        }


# ─── System Prompts ──────────────────────────────────────────────────────────

RESEARCH_SYSTEM = """You are a senior SR&ED (Scientific Research & Experimental Development) consultant \
in Canada with 15+ years of experience writing successful CRA claims.

Your task: analyze a client meeting transcript and their company website, then identify and select \
the single strongest SR&ED-eligible project to file a claim for.

SUPPLEMENTARY DOCUMENTS: The user may provide optional supporting files (technical specs, prior reports, \
emails, meeting notes). When present, treat these as primary source material alongside the transcript — \
they often contain the most technically detailed evidence of SR&ED-eligible work. Cross-reference them \
with the transcript to build the most complete picture of what was actually investigated.

IMPORTANT — HANDLING SPARSE OR VAGUE TRANSCRIPTS:
Meeting transcripts are often unhelpful: clients speak in business terms, skip technical details, \
or the recording is incomplete. When the transcript is sparse, vague, or missing key technical \
information, do the following:
- Rely primarily on the company website to understand their products, technology stack, and domain.
- Use the transcript only for signals: any mention of a technical challenge, a project name, \
a technology, a performance problem, or a deadline is a useful clue even if unexplained.
- Apply your SR&ED domain expertise to infer what technical uncertainties a company in their \
industry and at their stage of development would plausibly have encountered.
- It is better to construct a well-reasoned, plausible SR&ED project from limited information \
than to declare the information insufficient. A good consultant reads between the lines.
- If transcript content is entirely absent or uninformative, base the project entirely on the \
website and flag this in the selection_rationale.

SR&ED ELIGIBILITY — all three criteria must be present simultaneously:
1. TECHNOLOGICAL UNCERTAINTY: A technical problem whose solution cannot be determined using commonly \
available scientific or technological knowledge or experience. A qualified practitioner in the field \
could NOT resolve it without original investigation. This is NOT a business risk, NOT a timeline \
problem, NOT a feature gap.
2. SYSTEMATIC INVESTIGATION: Hypothesis-driven experimentation with documented results — not ad hoc \
troubleshooting or standard trial-and-error.
3. TECHNOLOGICAL ADVANCEMENT: New knowledge generated that advances the understanding of science or \
technology — even failed attempts that prove an approach doesn't work count.

WHAT QUALIFIES:
- Novel algorithm development where known approaches cannot meet performance requirements
- New protocols or architectures where interaction behavior under target conditions is unknown
- Material/process development where behavior at required conditions is undocumented in literature
- System integration where combining known components produces unpredictable emergent failure modes
- Any domain where the team had to conduct original experiments because existing knowledge was insufficient

WHAT DOES NOT QUALIFY:
- Standard feature development using documented APIs or frameworks
- Performance tuning through established profiling/optimization methods
- Business process improvements without a technical knowledge gap
- Routine bug fixes or maintenance
- Applying known solutions to known problems, even if the work was difficult

OUTPUT FORMAT: Return valid JSON only, with these exact keys:
{
  "title": "Concise descriptive project title (e.g., 'Development of Adaptive Cache Invalidation for Distributed Stateful Sessions')",
  "industry": "Industry sector (e.g., 'Software', 'Manufacturing', 'Biotechnology', 'Clean Technology')",
  "tech_domain": "Technical domain (e.g., 'Distributed Systems', 'Machine Learning', 'Materials Science')",
  "company_background": "2–3 sentences describing what the company does, drawn from website content",
  "technical_work": "Detailed description of the specific technical work performed in this project",
  "technological_uncertainty": "The specific technical unknowns — what the team did NOT know at project start and why standard knowledge or published approaches were insufficient to determine the answer",
  "investigation_approach": "How the team systematically investigated — hypotheses formed, experiments and tests run, results obtained including failures",
  "potential_advancement": "What new technical knowledge or principles were generated or attempted, including what was learned from failures",
  "selection_rationale": "1–2 sentences explaining why this project is the strongest SR&ED candidate, and note if the transcript was sparse so the claim is primarily inferred from the website"
}"""


TITLE_SYSTEM = """You are a senior SR&ED technical writer. Generate a concise, descriptive project \
title for a T661 SR&ED claim.

REQUIREMENTS:
- Strictly fewer than 70 characters (including spaces) — this is a hard limit.
- Describe the technical work, not the business outcome.
- Use technical terminology appropriate for CRA review.
- Common format: "Development of [Technical Approach] for [Problem Domain]"
- No company names.

Output the title as a single line of plain text with no quotes, labels, or punctuation \
beyond what is part of the title itself."""


WRITER_UNCERTAINTY = """You are a senior SR&ED technical writer. Write the Technological Uncertainty \
section (Line 242) of a T661 form.

SECTION PURPOSE: Establish that at project start, the technical problem could NOT be solved using \
commonly available scientific or technological knowledge. This is the foundation of the entire claim — \
if the uncertainty is not convincing, the whole claim fails.

WORD LIMIT: {min_words}–{max_words} words. Stay within this range.

CRITICAL RULES:
1. Open by describing the state of the art — what was known in the field at project start, \
and why it was insufficient for the team's specific challenge.
2. State the specific unknowns precisely: variables whose behavior was unpredictable, interactions \
whose outcomes under required conditions were unknown, thresholds that had not been established.
3. Explicitly distinguish "what any qualified practitioner could do" (standard practice) vs. \
"what required original investigation" (the actual uncertainty).
4. Use phrases like: "It was unknown whether...", "Standard frameworks did not address the interaction \
between X and Y under conditions Z...", "No published approach established a method for...", \
"The behavior of [mechanism] under [conditions] had not been characterized..."
5. NEVER mention business goals, timelines, budgets, costs, or market requirements.
6. NEVER mention the company name — use "the team" or "the project."
7. Write in paragraphs only. No bullet points, no headings. Objective, technical register."""


WRITER_INVESTIGATION = """You are a senior SR&ED technical writer. Write the Work Performed \
(Systematic Investigation) section (Line 244) of a T661 form.

SECTION PURPOSE: Demonstrate that the work was a disciplined, hypothesis-driven investigation — \
not ad hoc problem solving, not a project status update, not a feature development log.

WORD LIMIT: {min_words}–{max_words} words. Stay within this range.

CRITICAL RULES:
1. Structure as a chronological technical narrative: Hypothesis → Experiment/Test → Result → \
Learning → Next Hypothesis. Each cycle must show what was expected, what actually happened, \
and what that revealed about the underlying problem.
2. INCLUDE AT LEAST ONE FAILURE OR DEAD END. A straight path to success looks like routine \
engineering, not experimental research. Describe what was tried, why it was expected to work, \
and why it failed — this is often the most SR&ED-eligible part.
3. Include specific technical details and quantitative measures where available \
(e.g., "three architectural configurations were evaluated," "latency increased 40% under concurrent load," \
"the model achieved 62% accuracy vs. a 78% target threshold").
4. Use phrases like: "The team hypothesized that...", "Initial tests demonstrated...", \
"To isolate the variable, the configuration was modified to...", \
"This approach was abandoned when results showed...", "The revised hypothesis held that..."
5. NEVER mention company name, commercial milestones, product launch dates, or business outcomes.
6. Write in paragraphs only. No bullet points, no headings."""


WRITER_ADVANCEMENT = """You are a senior SR&ED technical writer. Write the Technological Advancement \
section (Line 246) of a T661 form.

SECTION PURPOSE: State what new knowledge was generated — knowledge that advances the technical \
understanding of the domain, not just the company's product capability.

WORD LIMIT: {min_words}–{max_words} words. Stay within this range.

CRITICAL RULES:
1. Frame as new knowledge or principles, NOT product features. \
"The team established that under conditions X, mechanism Y produces effect Z" — not "the feature now works."
2. Directly address each uncertainty named in Line 242. Each uncertainty should have a \
corresponding advancement: either confirmed resolution, partial understanding, or confirmed \
non-viability of a specific approach.
3. INCLUDE FAILED INVESTIGATIONS as advancements. \
"It was determined that approach A is non-viable under conditions B due to mechanism C" \
is a genuine technological advancement. The knowledge that something does not work has value.
4. Use phrases like: "This work established a new understanding of...", \
"The team generated new knowledge regarding the relationship between...", \
"It was demonstrated that [approach] is insufficient when...", \
"A new baseline was established for [parameter] under [conditions]..."
5. NEVER mention company name, revenue, market position, business impact, or product milestones.
6. Write in paragraphs only. No bullet points, no headings."""


REVIEWER_SYSTEM = """You are a CRA (Canada Revenue Agency) SR&ED technical reviewer with authority \
to approve or reject claims. You are reviewing a complete T661 technical narrative (all three sections) \
against SR&ED eligibility criteria.

REVIEW EACH SECTION against these standards:

LINE 242 — Technological Uncertainty:
- Does it clearly establish that the solution was NOT derivable from commonly available knowledge?
- Is the uncertainty framed technically (not as a business risk, budget constraint, or feature gap)?
- Are the specific unknowns stated precisely — not vaguely as "challenges" or "difficulties"?
- Is it free of company names, business language, and commercial objectives?

LINE 244 — Work Performed:
- Is there a clear hypothesis-test-result structure with chronological flow?
- Does it include at least one failure or dead end? (absence = red flag for routine work)
- Are there specific technical details and quantitative measures?
- Does it read as a systematic investigation, not a project progress update or feature checklist?
- Is it free of company names, product names, and commercial milestones?

LINE 246 — Technological Advancement:
- Does it state new KNOWLEDGE or PRINCIPLES, not product features or business outcomes?
- Does it correspond directly to the uncertainties stated in Line 242?
- Does it treat failed investigations as advancements where applicable?
- Is it free of company names and business impact language?

COHERENCE CHECK:
- Do the three sections form a coherent arc? (242 establishes unknowns → 244 investigates them → 246 resolves or partially resolves them)
- Are the same technical concepts referenced consistently across all three sections?

OUTPUT FORMAT: Return valid JSON only:
{
  "approved": true or false,
  "feedback": {
    "uncertainty": "specific required fixes, or null if this section passes",
    "systematic_investigation": "specific required fixes, or null if this section passes",
    "technological_advancement": "specific required fixes, or null if this section passes"
  },
  "overall_notes": "1–2 sentences on overall quality and coherence of the claim"
}

Set "approved": true only when ALL sections pass. If approved, all feedback values must be null."""


REVISER_SYSTEM = """You are a senior SR&ED technical writer revising a draft section based on \
specific CRA reviewer feedback.

INSTRUCTIONS:
1. Read the original draft carefully.
2. Read the reviewer feedback — address EVERY point raised.
3. Preserve all valid technical details from the original draft.
4. Do not introduce technical claims not grounded in the project brief provided.
5. Maintain the word count within {min_words}–{max_words} words.
6. Write in paragraphs only. No bullet points, no headings.
7. Output the revised section text directly — no preamble like "Here is the revised version."."""


# ─── Client ──────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = OpenAI()
        logger.info("LLMClient initialized model=%s", self.model)

    # ── Public methods ────────────────────────────────────────────────────────

    def research(
        self,
        transcript: str,
        website_text: str,
        supplementary_docs: list[dict] | None = None,
    ) -> dict:
        """Analyze transcript + website (+ optional supplementary docs) to select the strongest SR&ED project."""
        docs = supplementary_docs or []
        docs_section = ""
        if docs:
            parts = []
            for d in docs:
                # Truncate each doc to avoid context overflow
                content = d.get("content", "")[:3000]
                parts.append(f"--- {d.get('name', 'document')} ---\n{content}")
            docs_section = "\n\nSUPPLEMENTARY DOCUMENTS:\n" + "\n\n".join(parts)

        user_msg = (
            f"COMPANY WEBSITE CONTENT:\n{website_text or '(not available)'}\n\n"
            f"MEETING TRANSCRIPT:\n{transcript}"
            f"{docs_section}"
        )
        raw = self._call_llm(
            system=RESEARCH_SYSTEM,
            user=user_msg,
            log_tag="research",
            json_mode=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Research agent returned invalid JSON: %s", raw[:300])
            return {
                "title": "SR&ED Project",
                "industry": "Technology",
                "tech_domain": "",
                "company_background": "",
                "technical_work": raw,
                "technological_uncertainty": "",
                "investigation_approach": "",
                "potential_advancement": "",
                "selection_rationale": "",
            }

    def generate_title(self, brief: "ProjectBrief") -> str:
        """Generate a project title that is strictly fewer than 70 characters."""
        user_msg = (
            "Generate an SR&ED project title for the following project.\n\n"
            + self._brief_to_context(brief)
        )
        return self._generate_with_check(
            system=TITLE_SYSTEM,
            user=user_msg,
            content_type="title",
            log_tag="generate-title",
        )

    def write_section(
        self,
        section_key: str,
        brief: "ProjectBrief",
        min_words: int,
        max_words: int,
    ) -> str:
        system = self._writer_prompt(section_key, min_words, max_words)
        user_msg = self._brief_to_context(brief)
        return self._generate_with_check(
            system=system,
            user=user_msg,
            content_type=section_key,
            log_tag=f"write-{section_key}",
        )

    def review_report(self, brief: "ProjectBrief", sections: Dict[str, str]) -> dict:
        """Review all three sections holistically."""
        user_msg = (
            f"PROJECT: {brief.title}\n"
            f"INDUSTRY: {brief.industry}\n\n"
            f"LINE 242 – Technological Uncertainty:\n{sections.get('uncertainty', '')}\n\n"
            f"LINE 244 – Work Performed:\n{sections.get('systematic_investigation', '')}\n\n"
            f"LINE 246 – Technological Advancement:\n{sections.get('technological_advancement', '')}"
        )
        raw = self._call_llm(
            system=REVIEWER_SYSTEM,
            user=user_msg,
            log_tag="review",
            json_mode=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Reviewer returned invalid JSON: %s", raw[:300])
            return {
                "approved": False,
                "feedback": {
                    "uncertainty": raw,
                    "systematic_investigation": None,
                    "technological_advancement": None,
                },
                "overall_notes": "",
            }

    def revise_section(
        self,
        section_key: str,
        brief: "ProjectBrief",
        draft: str,
        feedback: str,
        min_words: int,
        max_words: int,
    ) -> str:
        system = REVISER_SYSTEM.format(min_words=min_words, max_words=max_words)
        user_msg = (
            f"PROJECT BRIEF:\n{self._brief_to_context(brief)}\n\n"
            f"ORIGINAL DRAFT ({section_key}):\n{draft}\n\n"
            f"REVIEWER FEEDBACK:\n{feedback}"
        )
        return self._generate_with_check(
            system=system,
            user=user_msg,
            content_type=section_key,
            log_tag=f"revise-{section_key}",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_with_check(
        self,
        system: str,
        user: str,
        content_type: str,
        log_tag: str,
        max_attempts: int = 3,
    ) -> str:
        """
        Generate content and use the check_content tool to validate it.
        If the content fails validation the feedback is fed back to the LLM
        as a follow-up user message so it can self-correct, up to max_attempts.
        """
        messages: List[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        content = ""
        for attempt in range(1, max_attempts + 1):
            content = self._call_llm_messages(messages, log_tag=f"{log_tag}-a{attempt}")
            result = check_content(content, content_type)
            logger.info(
                "check_content: type=%s valid=%s feedback=%s",
                content_type, result["valid"], result.get("feedback"),
            )
            if result["valid"]:
                break
            if attempt < max_attempts:
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"The content does not meet the requirements. "
                            f"Tool feedback: {result['feedback']} "
                            f"Please rewrite the content addressing this issue."
                        ),
                    },
                ]
        return content

    def _call_llm_messages(
        self,
        messages: List[dict],
        log_tag: str,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {"model": self.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        logger.info("LLM %s complete. tokens=%d length=%d", log_tag, tokens, len(content))
        return content.strip()

    def _writer_prompt(self, section_key: str, min_words: int, max_words: int) -> str:
        templates = {
            "uncertainty": WRITER_UNCERTAINTY,
            "systematic_investigation": WRITER_INVESTIGATION,
            "technological_advancement": WRITER_ADVANCEMENT,
        }
        template = templates.get(section_key, WRITER_UNCERTAINTY)
        return template.format(min_words=min_words, max_words=max_words)

    def _brief_to_context(self, brief: "ProjectBrief") -> str:
        return (
            f"PROJECT TITLE: {brief.title}\n"
            f"INDUSTRY: {brief.industry}\n"
            f"TECH DOMAIN: {brief.tech_domain}\n"
            f"COMPANY BACKGROUND: {brief.company_background}\n"
            f"TECHNICAL WORK: {brief.technical_work}\n"
            f"TECHNOLOGICAL UNCERTAINTY: {brief.technological_uncertainty}\n"
            f"INVESTIGATION APPROACH: {brief.investigation_approach}\n"
            f"POTENTIAL ADVANCEMENT: {brief.potential_advancement}"
        )

    def _call_llm(
        self,
        system: str,
        user: str,
        log_tag: str,
        json_mode: bool = False,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._call_llm_messages(messages, log_tag=log_tag, json_mode=json_mode)
