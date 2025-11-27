from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from llm_client import LLMClient


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "sred_reports"
# UPGRADE: Must match the model used in scripts/ingest.py
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"


SECTION_LABELS = {
    "uncertainty": "Technological Uncertainty",
    "systematic_investigation": "Systematic Investigation",
    "technological_advancement": "Technological Advancement",
}

WORD_LIMITS = {
    "uncertainty": (300, 350),
    "systematic_investigation": (650, 700),
    "technological_advancement": (300, 350),
}

logger = logging.getLogger("sred_app.agents")


@dataclass
class RetrievedExample:
    text: str
    metadata: Dict[str, str]


class RetrievalAgent:
    def __init__(self) -> None:
        # This will download the new model automatically on first run
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(allow_reset=False),
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)

    def retrieve(
        self,
        query: str,
        section: str,
        tech_code: str | None = None,
        industry: str | None = None,
        n_results: int = 5,
    ) -> List[RetrievedExample]:
        # (This uses the robust fallback logic from Step 2)
        query_embedding = self.embedding_model.encode([query])

        def _execute_query(conditions: List[Dict]) -> Dict:
            if len(conditions) == 1:
                where_clause = conditions[0]
            else:
                where_clause = {"$and": conditions}
            return self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_clause,
            )

        strict_conditions = [{"status": "approved"}, {"section": section}]
        if industry: strict_conditions.append({"industry": industry})
        if tech_code: strict_conditions.append({"tech_code": tech_code})

        results = _execute_query(strict_conditions)
        found_docs = results.get("documents", [[]])[0]
        found_metas = results.get("metadatas", [[]])[0]

        # Fallback Logic
        if len(found_docs) < n_results and tech_code:
            logger.info("Retrying with relaxed filters (ignoring tech_code)...")
            relaxed_conditions = [{"status": "approved"}, {"section": section}]
            if industry: relaxed_conditions.append({"industry": industry})
            
            relaxed = _execute_query(relaxed_conditions)
            found_docs.extend(relaxed.get("documents", [[]])[0])
            found_metas.extend(relaxed.get("metadatas", [[]])[0])
        
        examples: List[RetrievedExample] = []
        seen = set()
        for doc, meta in zip(found_docs, found_metas):
            if doc not in seen:
                seen.add(doc)
                examples.append(RetrievedExample(text=doc, metadata=meta or {}))
        
        return examples[:n_results]


class ReviewerAgent:
    """New Agent responsible for Critiquing Drafts."""
    def __init__(self):
        # OPTIMIZATION: Use GPT-4o for the 'Reviewer' to ensure strict, high-quality critique.
        # This agent needs to be smarter than the generator to catch subtle mistakes.
        self.llm = LLMClient(model="gpt-4o")

    def review(self, section: str, draft: str) -> str:
        return self.llm.review_draft(section, draft)


class SimpleGeneratorAgent:
    def __init__(self, retrieval_agent: RetrievalAgent) -> None:
        self.retrieval_agent = retrieval_agent
        # OPTIMIZATION: Use GPT-4o-mini for the 'Generator' for speed and cost-efficiency.
        self.llm = LLMClient(model="gpt-4o-mini")

    def generate_section(
        self,
        section: str,
        industry: str,
        tech_code: str | None,
        project_description: str | None,
        company_summary: str | None = None,
    ) -> str:
        label = SECTION_LABELS.get(section, section)
        min_words, max_words = WORD_LIMITS.get(section, (300, 350))
        
        query = f"Industry: {industry}\nTech: {tech_code}\nDesc: {project_description}"
        
        examples = self.retrieval_agent.retrieve(
            query=query,
            section=section,
            tech_code=tech_code,
            industry=industry,
            n_results=3,
        )

        example_texts = []
        for ex in examples:
            title = ex.metadata.get("project_title", "Example")
            example_texts.append(f"Project: {title}\n\n{ex.text}")

        return self.llm.generate_section(
            section_label=label,
            project_description=project_description or "",
            industry=industry,
            tech_code=tech_code or "",
            examples=example_texts,
            min_words=min_words,
            max_words=max_words,
            company_summary=company_summary,
        )

    def refine_section(self, section: str, draft: str, feedback: str) -> str:
        label = SECTION_LABELS.get(section, section)
        return self.llm.refine_draft(label, draft, feedback)


class ReportGenerator:
    def __init__(self) -> None:
        self.retrieval_agent = RetrievalAgent()
        self.generator = SimpleGeneratorAgent(self.retrieval_agent)
        self.reviewer = ReviewerAgent()

    def generate_report(
        self,
        industry: str,
        tech_code: str | None,
        project_description: str | None,
        company_summary: str | None = None,
    ) -> Dict[str, str]:
        sections = {}
        for key in ["uncertainty", "systematic_investigation", "technological_advancement"]:
            # 1. Generate Draft
            logger.info("Generating draft for %s...", key)
            draft = self.generator.generate_section(
                section=key,
                industry=industry,
                tech_code=tech_code,
                project_description=project_description,
                company_summary=company_summary,
            )

            # 2. Review Draft
            feedback = self.reviewer.review(key, draft)
            
            # 3. Refine (if rejected)
            if "APPROVED" not in feedback:
                logger.info("Refining %s based on feedback...", key)
                final_version = self.generator.refine_section(key, draft, feedback)
                sections[key] = final_version
            else:
                logger.info("%s accepted on first pass.", key)
                sections[key] = draft
                
        return sections