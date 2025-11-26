from __future__ import annotations

import logging
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logger = logging.getLogger("sred_app.llm")


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI()
        logger.info("LLMClient initialized model=%s", self.model)

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

        system_message = """You are an expert SR&ED technical writer specializing in preparing clear, specific, CRA-aligned technical narratives.
            Your goals:
            1. Write objectively, in professional business English, suitable for a CRA technical reviewer.
            2. Do NOT mention or repeat the company name; refer to the organization using neutral terms such as “the team,” “the engineering group,” or “the R&D project.”
            3. Include realistic technical details, sample quantitative data, or plausible metrics (e.g., iteration counts, failure rates, dataset sizes, timing benchmarks, volumes of records, approximate parameters). 
            - These details must sound credible but must never contradict the project description.
            4. Avoid generic or overly “successful” language; realistically describe obstacles, uncertainties, technical limitations, and remaining challenges.
            5. Do not mention that you are an AI model.
            6. Do not reference the prompt, examples, or instructions. Write as a fully self-contained SR&ED section.
            7. Ensure the content is unique and not directly copied from the examples.
        """

        user_message = (
            f"Section: {section_label}\n\n"
            f"Project context:\n"
            f"- Industry: {industry or 'N/A'}\n"
            f"- Tech code: {tech_code or 'N/A'}\n"
            f"- Company summary: {company_summary or 'N/A'}\n"
            f"- Description: {project_description}\n\n"
            f"Use the following approved SR&ED examples as style and content guidance:\n\n"
            f"{examples_text}\n\n"
            f"Now write a single, self-contained SR&ED section focused on {section_label.lower()}. "
            f"Aim for between {min_words} and {max_words} words. "
            f"Format: paragraphs only; no headings or bullet points."
            f"Do NOT mention or repeat the company name. Use neutral phrasing (e.g., “the team,” “the development group,” “the project”)."
            f"Include realistic engineering details, sample metrics, and plausible data points that add credibility."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        logger.info(
            "LLM response received section=%s chars=%d",
            section_label,
            len(response.choices[0].message.content or ""),
        )

        content = response.choices[0].message.content or ""
        return content.strip()
