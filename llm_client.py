from __future__ import annotations

import logging
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logger = logging.getLogger("sred_app.llm")


# --- SPECIALIZED PROMPTS ---

PROMPTS = {
    "uncertainty": """You are a technical SR&ED consultant.
Write the 'Technological Uncertainty' section. 
Crucial Rules:
1. Focus ONLY on the technical knowledge gap. What did the team NOT know at the start?
2. Explicitly contrast "Standard Practice" (what a typical engineer could do easily) vs. the specific "Unknowns" (variables, interactions, or limitations) that required experimentation.
3. DO NOT discuss business risks (e.g., cost, market timelines, budget) or routine software bugs.
4. Use phrases like: "It was unsure if...", "Standard frameworks did not support...", "The interaction between X and Y was unpredictable."
5. Do NOT mention the company name. Use "The team" or "The project".
""",

    "investigation": """You are a technical SR&ED consultant.
Write the 'Systematic Investigation' section.
Crucial Rules:
1. Structure this chronologically as a technical narrative: Hypothesis -> Experiment/Test -> Result -> Conclusion/Next Step.
2. Highlight the "Iterative Process". Describe failures and what was learned from them. A straight line to success sounds fake.
3. Include quantitative details where possible (e.g., "Tested 3 configurations," "Latency reduced by 15%," "Dataset of 5000 images").
4. Use phrases like: "The team hypothesized...", "Initial tests failed because...", "To isolate the variable, we modified..."
5. Do NOT mention the company name.
""",

    "advancement": """You are a technical SR&ED consultant.
Write the 'Technological Advancement' section.
Crucial Rules:
1. Focus on the NEW KNOWLEDGE gained, not just the new product features.
2. Explain how the company's technical baseline was elevated. What can they do now that they couldn't do before?
3. If the project failed, explain the "knowledge gained through failure" (e.g., knowing that this specific approach is invalid).
4. Use phrases like: "We generated new insight into...", "The team established a new baseline for...", "This work extended the capabilities of [Technology] by..."
5. Do NOT mention the company name.
""",

    "reviewer": """You are a strict CRA (Canada Revenue Agency) technical reviewer.
Critique the following SR&ED draft section.
Your Goal: Identify reasons this might be REJECTED.

Check for:
1. **Business Risks:** Mention of costs, budgets, sales, or marketing (Immediate Fail).
2. **Vague Metrics:** Usage of "significant," "huge," "fast" without numbers.
3. **Routine Work:** Descriptions that sound like standard bug fixing or library integration.
4. **Company Name:** Any mention of the specific company name (must be anonymous).

Output Format:
If the draft is perfect, reply with exactly: "APPROVED"
Otherwise, provide a bulleted list of specific required fixes.
""",

    "refiner": """You are a Senior SR&ED Technical Writer.
Your goal is to rewrite a draft section based on specific Reviewer feedback.

Instructions:
1. Read the Original Draft.
2. Read the Reviewer Feedback.
3. Rewrite the section to address EVERY point in the feedback.
4. Keep the good technical details from the original; only fix the problems.
5. Ensure the tone remains objective and professional.
6. Do not include any preamble (e.g., "Here is the rewritten version"). Just output the text.
""",

    "default": """You are an expert SR&ED technical writer.
Write a clear, specific, CRA-aligned technical narrative.
"""
}


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI()
        logger.info("LLMClient initialized model=%s", self.model)

    def _get_system_prompt(self, section_label: str) -> str:
        label = section_label.lower()
        if "uncertainty" in label:
            return PROMPTS["uncertainty"]
        elif "investigation" in label or "systematic" in label:
            return PROMPTS["investigation"]
        elif "advancement" in label:
            return PROMPTS["advancement"]
        return PROMPTS["default"]

    def generate_section(
        self,
        section_label: str,
        project_description: str,
        industry: str,
        tech_code: str,
        examples: List[str],
        min_words: int,
        max_words: int,
        company_summary: str | None = None,
    ) -> str:
        examples_text = (
            "\n\n---\n\n".join(examples) if examples else "No prior examples available."
        )
        system_message = self._get_system_prompt(section_label)
        system_message += "\n\nFORMATTING: Paragraphs only. No headings. Do not mention you are AI."

        user_message = (
            f"SECTION: {section_label}\n"
            f"CONTEXT: Industry={industry}, Tech Code={tech_code}\n"
            f"DESCRIPTION: {project_description}\n"
            f"COMPANY SUMMARY: {company_summary or 'N/A'}\n\n"
            f"EXAMPLES:\n{examples_text}\n\n"
            f"Write a {min_words}-{max_words} word draft."
        )

        return self._call_llm(system_message, user_message, section_label)

    def review_draft(self, section_label: str, draft_text: str) -> str:
        """Asks the LLM to critique the draft."""
        system_message = PROMPTS["reviewer"]
        user_message = f"SECTION: {section_label}\n\nDRAFT TEXT:\n{draft_text}"
        
        response = self._call_llm(system_message, user_message, f"review-{section_label}")
        return response

    def refine_draft(self, section_label: str, draft_text: str, feedback: str) -> str:
        """Asks the LLM to rewrite the draft based on feedback."""
        system_message = PROMPTS["refiner"]
        user_message = (
            f"SECTION: {section_label}\n\n"
            f"ORIGINAL DRAFT:\n{draft_text}\n\n"
            f"REVIEWER FEEDBACK:\n{feedback}\n\n"
            f"Please write the final corrected version:"
        )
        
        return self._call_llm(system_message, user_message, f"refine-{section_label}")

    def _call_llm(self, system: str, user: str, log_tag: str) -> str:
        """Helper to centralize the API call."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or ""
        logger.info("LLM %s complete. Length=%d", log_tag, len(content))
        return content.strip()