from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parents[1]
APPROVED_DIR = BASE_DIR / "data" / "approved"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "approved_sections.jsonl"


QUESTION_SECTIONS = {
    "242": "uncertainty",
    "244": "systematic_investigation",
    "246": "technological_advancement",
}


@dataclass
class SectionRecord:
    report_id: str
    project_title: str
    status: str
    industry: str
    tech_code: str
    section: str
    text: str
    source_path: str

    def to_json(self) -> str:
        data = {
            "report_id": self.report_id,
            "project_title": self.project_title,
            "status": self.status,
            "industry": self.industry,
            "tech_code": self.tech_code,
            "section": self.section,
            "text": self.text,
            "source_path": self.source_path,
        }
        return json.dumps(data, ensure_ascii=False)


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def extract_project_title(lines: List[str]) -> str:
    for i, line in enumerate(lines):
        if re.search(r"\*\*200\*\*", line):
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if candidate:
                    return candidate
            break
    return ""


def extract_tech_code(lines: List[str]) -> str:
    for i, line in enumerate(lines):
        if re.search(r"\*\*206\*\*", line):
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                match = re.search(r"(\d{1,2}\.\d{2}\.\d{2})", candidate)
                if match:
                    return match.group(1)
                break
            break
    return ""


def _strip_blank_edges(lines: List[str]) -> List[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def extract_sections(lines: List[str]) -> Dict[str, str]:
    question_indices: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        match = re.search(r"\*\*(24[246])\*\*", line)
        if match:
            question_indices[match.group(1)] = idx

    if not question_indices:
        return {}

    ordered = sorted(question_indices.items(), key=lambda item: item[1])
    ordered.append(("END", len(lines)))

    sections: Dict[str, str] = {}
    for (q_curr, idx_curr), (_, idx_next) in zip(ordered, ordered[1:]):
        section_key = QUESTION_SECTIONS.get(q_curr)
        if not section_key:
            continue
        content_lines = _strip_blank_edges(lines[idx_curr + 1 : idx_next])
        if not content_lines:
            continue
        sections[section_key] = "\n".join(content_lines).strip()

    return sections


def parse_report(path: Path, report_id: str) -> Iterable[SectionRecord]:
    lines = read_lines(path)
    project_title = extract_project_title(lines)
    tech_code = extract_tech_code(lines)
    sections = extract_sections(lines)

    for section_name, text in sections.items():
        yield SectionRecord(
            report_id=report_id,
            project_title=project_title,
            status="approved",
            industry="",  # industry not present in markdown; left blank
            tech_code=tech_code,
            section=section_name,
            text=text,
            source_path=str(path.relative_to(BASE_DIR)),
        )


def main() -> None:
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    markdown_files = sorted(APPROVED_DIR.glob("*.md"))

    records: List[SectionRecord] = []
    for idx, path in enumerate(markdown_files, start=1):
        report_id = f"{idx:03d}"
        records.extend(parse_report(path, report_id))

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json() + "\n")


if __name__ == "__main__":
    main()

