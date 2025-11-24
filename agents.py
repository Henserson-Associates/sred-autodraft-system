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
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


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
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(allow_reset=False),
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)
        logger.info("RetrievalAgent initialized with collection=%s", COLLECTION_NAME)

    def retrieve(
        self,
        query: str,
        section: str,
        tech_code: str | None = None,
        industry: str | None = None,
        n_results: int = 5,
    ) -> List[RetrievedExample]:
        # Build where filter with proper Chroma syntax
        conditions = [
            {"status": "approved"},
            {"section": section},
        ]
        if tech_code:
            conditions.append({"tech_code": tech_code})
        if industry:
            conditions.append({"industry": industry})

        # Combine conditions with $and operator
        if len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        query_embedding = self.embedding_model.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
        )
        logger.info(
            "Retrieved examples for section=%s tech_code=%s industry=%s results=%d",
            section,
            tech_code or "N/A",
            industry or "N/A",
            len(results.get("ids", [[]])[0]),
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        examples: List[RetrievedExample] = []
        for doc, meta in zip(documents, metadatas):
            examples.append(RetrievedExample(text=doc, metadata=meta or {}))
        return examples


class SimpleGeneratorAgent:
    def __init__(self, retrieval_agent: RetrievalAgent) -> None:
        self.retrieval_agent = retrieval_agent
        self.llm = LLMClient()

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
        query_lines = [f"Industry: {industry}"]
        query_lines.append(f"Tech code: {tech_code or 'N/A'}")
        if company_summary:
            query_lines.append(f"Company summary: {company_summary}")
        query_lines.append(f"Description: {project_description or 'N/A'}")
        query = "\n".join(query_lines)

        examples = self.retrieval_agent.retrieve(
            query=query,
            section=section,
            tech_code=tech_code or None,
            industry=industry or None,
            n_results=3,
        )

        example_texts: List[str] = []
        for ex in examples:
            project_title = ex.metadata.get("project_title") or "Example project"
            example_texts.append(f"Project: {project_title}\n\n{ex.text}")

        generated = self.llm.generate_section(
            section_label=label,
            project_description=project_description or "",
            industry=industry,
            tech_code=tech_code or "",
            company_summary=company_summary,
            examples=example_texts,
            min_words=min_words,
            max_words=max_words,
        )
        logger.info(
            "Generated section=%s words_target=%d-%d examples_used=%d",
            section,
            min_words,
            max_words,
            len(example_texts),
        )

        return generated


class ReportGenerator:
    def __init__(self) -> None:
        retrieval_agent = RetrievalAgent()
        generator = SimpleGeneratorAgent(retrieval_agent)
        self.generator = generator

    def generate_report(
        self,
        industry: str,
        tech_code: str | None,
        project_description: str | None,
        company_summary: str | None = None,
    ) -> Dict[str, str]:
        sections = {}
        for section_key in ("uncertainty", "systematic_investigation", "technological_advancement"):
            sections[section_key] = self.generator.generate_section(
                section=section_key,
                industry=industry,
                tech_code=tech_code,
                project_description=project_description,
                company_summary=company_summary,
            )
        return sections
