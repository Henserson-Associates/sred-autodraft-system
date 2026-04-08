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


class ContentConstraintError(ValueError):
    def __init__(
        self,
        *,
        content_type: str,
        feedback: str | None,
        attempt_count: int,
        stats: dict | None = None,
    ) -> None:
        super().__init__(feedback or f"Content failed constraints for {content_type}")
        self.content_type = content_type
        self.feedback = feedback
        self.attempt_count = attempt_count
        self.stats = stats or {}


MAX_COMPLETION_TOKENS: Dict[str, int] = {
    # Heuristic caps. These limit runaway verbosity, but do not guarantee word counts.
    "title": 40,
    "uncertainty": 900,
    "systematic_investigation": 1700,
    "technological_advancement": 900,
}


def max_completion_tokens_for(content_type: str) -> int | None:
    default = MAX_COMPLETION_TOKENS.get(content_type)
    env_key = f"SRED_MAX_COMPLETION_TOKENS_{content_type.upper()}"
    if env_key in os.environ:
        try:
            value = int(os.environ[env_key])
            return value if value > 0 else None
        except ValueError:
            return default
    return default


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

CRITICAL — EXTRACT COMPONENT-LEVEL SPECIFICS:
Your output quality directly determines the quality of the written T661 claim. Vague research \
output produces vague, rejectable claims. You MUST extract:
- Named system components, APIs, models, data sources, or subsystems involved
- The specific interactions or integration points where behavior was unpredictable
- Concrete failure modes: what broke, what was inconsistent, what exceeded acceptable thresholds
- Measurable indicators: error rates, latency spikes, accuracy gaps, variance percentages
- What the team tried first (and why it failed) before finding a working approach
- At least 2–3 distinct technical unknowns, each grounded at the component or mechanism level

DO NOT summarize the project at a conceptual level. Vague statements like "the team faced \
uncertainty about system behavior" or "the material properties were hard to predict" are worthless. \
The required level of precision looks like this — adapted to whatever domain applies:
- Software/Systems: "It was unknown whether [component X] would maintain [property Y] under \
  [concurrent/load/integration condition Z], as no published characterization of this interaction existed."
- Manufacturing/Materials: "It was unknown whether [material/alloy X] would retain [structural/thermal \
  property Y] under [process condition Z], as existing literature only characterized behavior at \
  conditions outside the required operating range."
- Biotech/Chemistry: "It was unknown whether [enzyme/compound X] would maintain [activity/stability Y] \
  when [expressed/synthesized/reacted] under [condition Z], as prior studies had not established \
  behavior at the required [temperature/pH/concentration] range."
- Clean Tech/Hardware: "It was unknown whether [mechanism X] would achieve [efficiency/output threshold Y] \
  under [environmental/load condition Z], as the interaction between [variable A] and [variable B] \
  at this operating point had not been characterized."
Apply this pattern to the actual domain of the project being analyzed.

OUTPUT FORMAT: Return valid JSON only, with these exact keys:
{
  "title": "Concise descriptive project title (e.g., 'Development of Adaptive Cache Invalidation for Distributed Stateful Sessions')",
  "industry": "Industry sector (e.g., 'Software', 'Manufacturing', 'Biotechnology', 'Clean Technology')",
  "tech_domain": "Technical domain (e.g., 'Distributed Systems', 'Machine Learning', 'Materials Science')",
  "company_background": "2–3 sentences describing what the company does, drawn from website content",
  "technical_work": "Detailed description of the specific technical work performed, naming the components, tools, and integration points involved",
  "technological_uncertainty": "2–3 distinct technical unknowns, each naming the specific component/mechanism, the condition under which behavior was unknown, and why existing knowledge could not resolve it. No generic statements like 'AI is unpredictable' — each uncertainty must be grounded in a concrete system failure or interaction gap.",
  "investigation_approach": "For each uncertainty: what hypothesis was formed, what experiment was run (naming configurations, tools, datasets), what result was observed (with metrics where possible), and what was concluded — including at least one approach that failed",
  "potential_advancement": "For each uncertainty: what new technical knowledge or principle was established — including what was learned from failures and dead ends",
  "selection_rationale": "1–2 sentences explaining why this project is the strongest SR&ED candidate, and note if the transcript was sparse so the claim is primarily inferred from the website"
}"""


USER_REVISION_SYSTEM = """You are a senior SR&ED technical writer.

Task: revise a single SR&ED report section based on user instructions, while preserving SR&ED eligibility.

RULES:
1. Follow the user's instructions as long as they do not introduce new technical claims not supported by the existing draft.
2. Keep the content within {min_words}–{max_words} words (hard limit).
3. Maintain CRA-appropriate tone: specific technical uncertainties, hypothesis-driven investigation, and technical advancement.
4. Write in paragraphs only (no bullet points, no headings).
5. Output ONLY the revised section text (no preamble).
"""

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
1. Open with the state of the art — what was established in the field at project start, and exactly \
why it was insufficient for this specific technical challenge. Be precise about the gap.
2. State 2–3 distinct technical unknowns. Each unknown must name: the specific component, material, \
mechanism, subsystem, compound, or process step involved; the condition or interaction under which \
behavior was unpredictable; and why no existing literature, standard practice, or prior art could \
resolve it. The required level of precision:
   "It was unknown whether [specific mechanism/material/component] would [specific behavior/property] \
   under [specific conditions/operating parameters/integration context]."
   This applies equally across all domains — software, manufacturing, biotech, clean tech, or any other.
3. Explicitly distinguish standard practice (what any qualified practitioner could do) from the \
genuine unknown (what required original investigation). CRA rejects claims where the uncertainty \
is just a known limitation dressed up as research.
4. Required phrases: "It was unknown whether...", "No established method addressed...", \
"Standard [framework/tool/approach] did not characterize the interaction between X and Y under Z...", \
"The behavior of [mechanism] under [conditions] had not been established in the literature..."
5. ANTI-PATTERNS — these phrases trigger CRA rejection regardless of domain. Never write:
   - "[technology] is inherently unpredictable / complex / difficult to work with" (known limitation, not uncertainty)
   - "the system / process / material faced challenges" (vague, not technical)
   - "it was difficult to achieve" (difficulty ≠ uncertainty)
   - "the team needed to determine how to" (implementation problem, not research question)
   - "standard approaches were insufficient" without specifying which approach, why it failed, \
     and what specific gap remained
   - Any sentence that a practitioner in the field could resolve by reading existing documentation, \
     vendor specifications, or published literature
6. NEVER mention business goals, timelines, budgets, costs, or market requirements.
7. NEVER mention the company name — use "the team" or "the project."
8. Write in paragraphs only. No bullet points, no headings. Objective, technical register."""


WRITER_INVESTIGATION = """You are a senior SR&ED technical writer. Write the Work Performed \
(Systematic Investigation) section (Line 244) of a T661 form.

SECTION PURPOSE: Demonstrate that the work was a disciplined, hypothesis-driven investigation — \
not ad hoc problem solving, not a project status update, not a feature development log.

WORD LIMIT: {min_words}–{max_words} words. Stay within this range.

STRUCTURE REQUIREMENT — UNCERTAINTY-DRIVEN PARAGRAPHS:
Do NOT write a chronological story. Structure the section so each paragraph directly addresses \
one of the specific uncertainties identified in the project brief. The required paragraph structure is:
  Uncertainty identified → Hypothesis formed → Experiment designed and executed \
  (name the tools, configurations, datasets, or parameters) → Measured result → \
  Technical conclusion and how it informed the next step.
This maps directly to CRA's expectation that work is driven by research questions, not a development roadmap.

CRITICAL RULES:
1. Every paragraph must open by referencing the specific uncertainty it addresses. \
Example: "To investigate whether [specific component] could [specific behavior] under [specific conditions], \
the team hypothesized that..."
2. INCLUDE AT LEAST ONE FAILURE OR DEAD END — described in its own paragraph following the same \
Uncertainty → Hypothesis → Experiment → Result structure. A straight path to success looks like \
routine engineering. The failure paragraph is often the strongest SR&ED evidence in the section.
3. Use specific technical details and quantitative measures: name the tools, configurations, \
materials, instruments, or process parameters tested; include measured outcomes. Examples by domain:
   - Software/Systems: "error rate under peak load exceeded the 0.5% threshold," "four caching \
     strategies were evaluated," "round-trip latency increased 60% above the acceptable ceiling"
   - Manufacturing/Materials: "tensile strength fell 18% below specification at 200°C," \
     "five alloy compositions were tested across three annealing cycles," "surface roughness \
     exceeded the 1.6 μm Ra tolerance under standard feed rates"
   - Biotech/Chemistry: "yield dropped below 40% at pH levels above 7.4," "three catalyst \
     concentrations were evaluated at four reaction temperatures," "binding affinity decreased \
     by 35% when expressed in the alternative host system"
   - Clean Tech/Hardware: "conversion efficiency fell 22% below target under partial-load conditions," \
     "six electrode geometries were tested," "thermal runaway was observed above 55°C ambient"
   Use the unit and measurement types appropriate to the actual project domain.
4. Required phrases: "The team hypothesized that...", "To isolate [variable], the configuration was \
modified to...", "Initial tests demonstrated...", "This approach was abandoned when results showed...", \
"The revised hypothesis held that..."
5. ANTI-PATTERNS — CRA reads these as routine engineering regardless of domain. Never write:
   - "The team adjusted / tuned / optimized the parameters" (standard practice, not research)
   - "Known techniques / standard methods / vendor-recommended settings were applied" \
     (describes routine work, not investigation)
   - "The team iterated on the design / formulation / configuration" (development lifecycle language)
   - "Testing was performed to validate the solution" (validation ≠ experimental investigation)
   - Any reference to a tool, material, or method without specifying what variable was isolated, \
     what was measured, and what the result revealed about the underlying uncertainty
6. NEVER mention company name, commercial milestones, product launch dates, or business outcomes.
7. Write in paragraphs only. No bullet points, no headings."""


WRITER_ADVANCEMENT = """You are a senior SR&ED technical writer. Write the Technological Advancement \
section (Line 246) of a T661 form.

SECTION PURPOSE: State what new knowledge was generated — knowledge that advances the technical \
understanding of the domain, not just the company's product capability.

WORD LIMIT: {min_words}–{max_words} words. Stay within this range.

STRUCTURE REQUIREMENT — ONE ADVANCEMENT PER UNCERTAINTY:
This section must systematically resolve each uncertainty stated in Line 242. For each uncertainty, \
write one paragraph stating the outcome: confirmed resolution (what principle was established and \
under what conditions), partial resolution (what was learned and what remains unresolved), or \
confirmed non-viability (what was proven not to work and why). Do not write a general summary — \
every uncertainty named in 242 must appear here with a corresponding knowledge outcome.

CRITICAL RULES:
1. Frame every statement as new knowledge or a new principle, NOT as a product capability or feature. \
The test: could this finding be published or cited as a technical result independent of any \
specific product? If the sentence describes what the product or process can now do, rewrite it \
as what the team now knows about the underlying mechanism, material, or system behavior. Examples:
   WRONG: "The process now meets production tolerances."
   RIGHT: "It was established that dimensional stability under repeated thermal cycling requires \
   a pre-treatment annealing step at 220°C for a minimum of 4 hours; without this step, \
   dimensional variance exceeded the 0.05 mm tolerance regardless of alloy composition."

   WRONG: "The software module now handles concurrent requests correctly."
   RIGHT: "It was established that consistent state synchronization under concurrent write \
   conditions requires a two-phase commit protocol at the cache layer; optimistic locking \
   alone produced conflict rates exceeding 8% under the target transaction volume."

   WRONG: "The formulation now achieves the required yield."
   RIGHT: "It was determined that acceptable enzyme yield in this host system is achievable \
   only within a pH range of 6.8–7.1; outside this range, misfolding rates increased \
   non-linearly, rendering standard expression protocols insufficient."
2. Failed investigations are advancements. Each dead end from Line 244 must appear here as a \
confirmed non-viability statement: "It was determined that [approach] is non-viable under \
[conditions] due to [technical mechanism]." This is genuine SR&ED advancement.
3. Use specific technical language matching the uncertainties: name the components, thresholds, \
mechanisms, and conditions. Vague advancement statements ("a better understanding was gained") \
are rejected by CRA.
4. Required phrases: "This work established that...", "It was determined that [approach] is \
insufficient when...", "The team generated new knowledge regarding the relationship between...", \
"A new technical understanding was developed of [mechanism] under [conditions]...", \
"It was confirmed that [approach] is non-viable due to..."
5. NEVER mention company name, revenue, market position, business impact, or product milestones.
6. Write in paragraphs only. No bullet points, no headings."""


REVIEWER_SYSTEM = """You are a CRA (Canada Revenue Agency) SR&ED technical reviewer with authority \
to approve or reject claims. You are reviewing a complete T661 technical narrative (all three sections) \
against SR&ED eligibility criteria. You are strict. You do not give the benefit of the doubt. \
If a section is borderline, it fails.

REVIEW LINE 242 — Technological Uncertainty:
PASS criteria (all must be met):
- Each uncertainty names a specific component, mechanism, API, or subsystem — not a general domain.
- Each uncertainty states the condition or interaction under which behavior was unknown.
- Each uncertainty explains why existing knowledge, literature, or standard practice could NOT resolve it.
- The section contains at least 2 distinct uncertainties at this level of specificity.
- No uncertainty is reducible to a known limitation of the technology or domain \
  (e.g., "materials behave differently at high temperatures", "distributed systems have latency", \
  "biological systems are variable") — a known limitation is NOT a technical uncertainty.
FAIL triggers (any one = fail):
- Any uncertainty framed as "it was challenging to" or "the team needed to figure out how to"
- Any uncertainty that a qualified practitioner could resolve using existing documentation
- Vague language: "complex interactions", "unpredictable behavior", "AI limitations"
- Business framing: timelines, costs, market fit, product requirements

REVIEW LINE 244 — Work Performed:
PASS criteria (all must be met):
- Each paragraph addresses one specific uncertainty from Line 242 (not a general project narrative).
- Each paragraph follows: Uncertainty addressed → Hypothesis → Experiment (with named tools, \
  configurations, or parameters) → Measured result → Conclusion.
- At least one failure or dead end is described with the same structure as successful experiments.
- Quantitative measures are present (thresholds, error rates, accuracy percentages, counts of \
  configurations tested, etc.).
FAIL triggers:
- Section reads as a project timeline or feature development log
- Paragraphs describe work without mapping it to a specific uncertainty
- Generic phrasing regardless of domain: "the team adjusted / tuned / optimized parameters", \
  "standard methods were applied and refined", "the team iterated on the design / formulation", \
  "testing was conducted to validate the approach"
- No failed experiment or dead end

REVIEW LINE 246 — Technological Advancement:
PASS criteria (all must be met):
- Every uncertainty stated in Line 242 has a corresponding advancement paragraph.
- Each advancement states a new principle or technical finding — not a product capability.
- Dead ends and failed approaches from Line 244 appear as confirmed non-viability statements.
- Technical specificity matches Line 242: same components, mechanisms, and conditions referenced.
FAIL triggers:
- Any advancement that describes what the product can now do instead of what the team now knows
- Any uncertainty from Line 242 without a corresponding resolution or non-viability statement
- Vague advancement language: "a better understanding was gained", "the team learned about"

COHERENCE CHECK:
- The same technical components, mechanisms, and conditions must appear consistently across all \
  three sections. If a component is named in 242, it must appear in 244 and 246.
- The arc must hold: 242 states unknowns → 244 investigates exactly those unknowns → \
  246 resolves or characterizes each one.

OUTPUT FORMAT: Return valid JSON only:
{
  "approved": true or false,
  "feedback": {
    "uncertainty": "itemized list of specific required fixes, or null if this section passes",
    "systematic_investigation": "itemized list of specific required fixes, or null if this section passes",
    "technological_advancement": "itemized list of specific required fixes, or null if this section passes"
  },
  "overall_notes": "1–2 sentences on overall coherence and the single most important issue to fix"
}

Set "approved": true only when ALL sections pass ALL criteria above. If any section has even one \
fail trigger, it must receive feedback and approved must be false."""


REVISER_SYSTEM = """You are a senior SR&ED technical writer revising a draft section based on \
specific CRA reviewer feedback.

INSTRUCTIONS:
1. Read the reviewer feedback first — it identifies exactly what CRA would reject. \
   Address EVERY point raised. Do not leave any feedback item unresolved.
2. Read the original draft to identify what technical content is valid and salvageable.
3. Apply the appropriate fix for each feedback item:
   - "Too vague / generic": replace with component-level specifics from the project brief
   - "Known limitation, not uncertainty": reframe to name what was specifically unknown about \
     the behavior of a named component under named conditions
   - "Not mapped to uncertainty": restructure the paragraph to open with the specific uncertainty \
     it addresses before describing the experiment
   - "No failure / dead end": add a paragraph describing a failed approach using the \
     Uncertainty → Hypothesis → Experiment → Result structure
   - "Advancement describes product capability": reframe as a technical principle or finding \
     established by the investigation
4. Preserve all valid technical details from the original draft — only rewrite what the \
   feedback flags, not the entire section.
5. Do not introduce technical claims not grounded in the project brief provided.
6. Maintain the word count within {min_words}–{max_words} words.
7. Write in paragraphs only. No bullet points, no headings.
8. Output the revised section text directly — no preamble like "Here is the revised version."."""


# ─── Client ──────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4")
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

    def revise_section_with_user_instructions(
        self,
        section_key: str,
        project_title: str,
        project_summary: str,
        current_section: str,
        instructions: str,
    ) -> str:
        limits = CONTENT_LIMITS.get(section_key)
        if not limits or limits.get("type") != "words":
            raise ValueError(f"Unsupported section_key for user revision: {section_key!r}")
        min_words, max_words = limits["min"], limits["max"]

        system = USER_REVISION_SYSTEM.format(min_words=min_words, max_words=max_words)
        user_msg = (
            f"PROJECT TITLE:\n{project_title}\n\n"
            f"PROJECT SUMMARY:\n{project_summary}\n\n"
            f"CURRENT DRAFT ({section_key}):\n{current_section}\n\n"
            f"USER INSTRUCTIONS:\n{instructions.strip()}"
        )
        return self._generate_with_check(
            system=system,
            user=user_msg,
            content_type=section_key,
            log_tag=f"user-revise-{section_key}",
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
        last_result: dict | None = None
        for attempt in range(1, max_attempts + 1):
            content = self._call_llm_messages(
                messages,
                log_tag=f"{log_tag}-a{attempt}",
                max_completion_tokens=max_completion_tokens_for(content_type),
            )
            result = check_content(content, content_type)
            last_result = result
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
        if last_result and not last_result.get("valid", False):
            raise ContentConstraintError(
                content_type=content_type,
                feedback=last_result.get("feedback"),
                attempt_count=max_attempts,
                stats=last_result,
            )

        return content

    def _call_llm_messages(
        self,
        messages: List[dict],
        log_tag: str,
        json_mode: bool = False,
        max_completion_tokens: int | None = None,
    ) -> str:
        kwargs: dict = {"model": self.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = int(max_completion_tokens)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Backward-compatibility: some older stacks only accept max_tokens.
            msg = str(exc)
            if max_completion_tokens is not None and "max_completion_tokens" in msg:
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = int(max_completion_tokens)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise
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
