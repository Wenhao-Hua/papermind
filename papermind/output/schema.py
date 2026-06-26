"""Pydantic data models — the single source of truth for PaperMind output.

The in-memory ``Report`` keeps each analysis module in its own section object so
the Python API reads naturally (``report.technical.details``,
``report.connections.related_works``).  ``Report.to_dict`` flattens those into
the JSON shape documented in the README.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from papermind.figures.framework import FrameworkSpec


def _write_text(path: str, text: str) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

Difficulty = Literal["high", "mid", "low"]
Confidence = Literal["high", "mid", "low"]
SegmentKind = Literal["fact", "inference", "out_of_scope"]
FigureType = Literal["original", "ai_generated"]


class Source(BaseModel):
    """A citation back into the source PDF."""

    text: str = Field(..., description="The quoted/paraphrased supporting snippet")
    section: Optional[str] = Field(None, description="e.g. 'Section 3.1' or 'Abstract'")
    page: Optional[int] = Field(None, description="1-based PDF page number")
    verified: bool = Field(True, description="Whether the snippet was matched against the paper text")


class PaperMeta(BaseModel):
    title: str
    arxiv_id: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    pdf_url: Optional[str] = None
    abstract: Optional[str] = None

    def page_link(self, page: Optional[int]) -> Optional[str]:
        """Build a clickable deep link into the PDF at a given page, if possible."""
        if not self.pdf_url or not page:
            return self.pdf_url
        return f"{self.pdf_url}#page={page}"


class Usage(BaseModel):
    """Accumulated token usage (and best-effort cost) for a client session."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def record(self, prompt_tokens: int, completion_tokens: int, cost_usd: float = 0.0) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.cost_usd += cost_usd

    def snapshot(self) -> "Usage":
        return Usage(**self.model_dump())

    def minus(self, prev: "Usage") -> "Usage":
        """Usage incurred since a prior snapshot (for per-turn reporting)."""
        return Usage(
            calls=self.calls - prev.calls,
            prompt_tokens=self.prompt_tokens - prev.prompt_tokens,
            completion_tokens=self.completion_tokens - prev.completion_tokens,
            total_tokens=self.total_tokens - prev.total_tokens,
            cost_usd=self.cost_usd - prev.cost_usd,
        )


# --------------------------------------------------------------------------- #
# Module 1: Contributions
# --------------------------------------------------------------------------- #
class Contributions(BaseModel):
    main_contribution: str = ""
    novelty: str = ""
    problem_solved: str = ""
    sources: List[Source] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Module 2: Technical details (+ figures)
# --------------------------------------------------------------------------- #
class FigureExplain(BaseModel):
    """A short "how to read this figure" walkthrough shown beside the figure."""

    gist: str                                          # 一句话：这张图在讲/定义什么
    parts: List[str] = Field(default_factory=list)     # 各关键部分的含义
    takeaway: str = ""                                 # 一句关键结论 / 怎么读


class Figure(BaseModel):
    type: FigureType = "ai_generated"
    image_path: Optional[str] = None
    mermaid: Optional[str] = None
    svg: Optional[str] = None  # self-contained teaching SVG (architecture-faithful), if generated
    caption: Optional[str] = None
    explain: Optional[FigureExplain] = None  # 读图讲解（结构化），随生成图一起产出


class TechnicalPoint(BaseModel):
    name: str
    explanation: str
    analogy: str = ""
    formula: Optional[str] = None  # key equation in LaTeX (no delimiters), if any
    source_section: Optional[str] = None
    page: Optional[int] = None
    difficulty: Difficulty = "mid"
    figure: Optional[Figure] = None


class TechnicalSection(BaseModel):
    """Wrapper so the public API reads as ``report.technical.details``."""

    details: List[TechnicalPoint] = Field(default_factory=list)

    def __iter__(self):  # convenience: ``for point in report.technical``
        return iter(self.details)

    def __len__(self) -> int:
        return len(self.details)


# --------------------------------------------------------------------------- #
# Module 3: Connections
# --------------------------------------------------------------------------- #
class Connection(BaseModel):
    concept: str
    paper: str = ""
    arxiv_link: Optional[str] = None
    relationship: str = ""


class ConnectionsSection(BaseModel):
    """Wrapper so the public API reads as ``report.connections.related_works``."""

    related_works: List[Connection] = Field(default_factory=list)

    def __iter__(self):
        return iter(self.related_works)

    def __len__(self) -> int:
        return len(self.related_works)


# --------------------------------------------------------------------------- #
# Module 4: Reproduction
# --------------------------------------------------------------------------- #
class SetupStep(BaseModel):
    step: int
    title: str
    desc: str = ""
    command: Optional[str] = None


class Benchmark(BaseModel):
    setting: str
    baseline: Optional[str] = None
    result: Optional[str] = None
    speedup: Optional[str] = None
    memory: Optional[str] = None


class Dataset(BaseModel):
    name: str
    purpose: str = ""
    link: Optional[str] = None


class CommonError(BaseModel):
    error: str
    cause: str = ""
    fix_command: Optional[str] = None


class RepoRef(BaseModel):
    """A *verified* code repository for the paper — found from the paper text or
    PapersWithCode and inspected via the GitHub API, so reproduction is grounded
    in the real repo (deps, run commands) instead of an LLM guess."""

    url: str
    source: str = ""  # how it was found, e.g. "论文原文链接" / "PapersWithCode"
    is_official: bool = False
    stars: Optional[int] = None
    description: Optional[str] = None
    default_branch: str = "main"
    language: Optional[str] = None
    license: Optional[str] = None
    dep_file: Optional[str] = None  # the dependency file actually present in the repo
    install_commands: List[str] = Field(default_factory=list)  # derived from real files
    run_commands: List[str] = Field(default_factory=list)  # detected from the real README


class Reproduction(BaseModel):
    official_code: Optional[str] = None
    code_repo: Optional[RepoRef] = None
    version_tag: Optional[str] = None
    requirements: Optional[str] = None
    recommended_hardware: Optional[str] = None
    key_hyperparams: List[str] = Field(default_factory=list)
    env_setup_steps: List[SetupStep] = Field(default_factory=list)
    performance_benchmarks: List[Benchmark] = Field(default_factory=list)
    datasets: List[Dataset] = Field(default_factory=list)
    common_errors: List[CommonError] = Field(default_factory=list)
    gotchas: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Top-level report
# --------------------------------------------------------------------------- #
class Report(BaseModel):
    paper: PaperMeta
    contributions: Optional[Contributions] = None
    technical: TechnicalSection = Field(default_factory=TechnicalSection)
    connections: ConnectionsSection = Field(default_factory=ConnectionsSection)
    reproduction: Optional[Reproduction] = None
    # Whole-method end-to-end framework diagram (generated alongside the figures).
    framework: Optional[FrameworkSpec] = None
    # Runtime token usage for this analysis; not part of the exported JSON artifact.
    usage: Optional["Usage"] = None

    # -- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Flatten to the documented JSON shape (technical_details / connections as lists)."""
        out: dict = {"paper": self.paper.model_dump(exclude_none=False)}
        if self.contributions is not None:
            out["contributions"] = self.contributions.model_dump()
        out["technical_details"] = [p.model_dump() for p in self.technical.details]
        out["connections"] = [c.model_dump() for c in self.connections.related_works]
        if self.reproduction is not None:
            out["reproduction"] = self.reproduction.model_dump()
        if self.framework is not None:
            out["framework"] = self.framework.model_dump()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Report":
        """Reconstruct a Report from the flattened JSON shape."""
        return cls(
            paper=PaperMeta(**data["paper"]),
            contributions=Contributions(**data["contributions"]) if data.get("contributions") else None,
            technical=TechnicalSection(
                details=[TechnicalPoint(**p) for p in data.get("technical_details", [])]
            ),
            connections=ConnectionsSection(
                related_works=[Connection(**c) for c in data.get("connections", [])]
            ),
            reproduction=Reproduction(**data["reproduction"]) if data.get("reproduction") else None,
            framework=FrameworkSpec.model_validate(data["framework"]) if data.get("framework") else None,
        )

    def to_json(self, path: Optional[str] = None) -> str:
        from papermind.output.json_export import to_json as _to_json

        return _to_json(self, path)

    def to_markdown(self, path: Optional[str] = None) -> str:
        from papermind.output.markdown import to_markdown as _to_markdown

        return _to_markdown(self, path)

    def to_html(self, path: Optional[str] = None) -> str:
        from papermind.output.html import to_html as _to_html

        return _to_html(self, path)

    def to_setup_script(self, path: Optional[str] = None) -> str:
        from papermind.output.reproduce_export import to_setup_script, write_setup_script

        return write_setup_script(self, path) if path else to_setup_script(self)

    def to_notebook(self, path: Optional[str] = None) -> dict:
        from papermind.output.reproduce_export import to_notebook, write_notebook

        return write_notebook(self, path) if path else to_notebook(self)


def report_reading_minutes(report: Report) -> int:
    """Estimate reading time in minutes for the generated report at 200 WPM."""
    texts: List[str] = []
    if report.contributions:
        c = report.contributions
        texts += [c.main_contribution, c.novelty, c.problem_solved]
    for p in report.technical.details:
        texts += [p.explanation, p.analogy]
    for w in report.connections.related_works:
        texts += [w.concept, w.relationship]
    if report.reproduction:
        r = report.reproduction
        texts += [r.requirements or "", r.recommended_hardware or ""]
        texts += list(r.gotchas)
        for e in r.common_errors:
            texts += [e.error, e.cause]
        for d in r.datasets:
            texts.append(d.purpose)
        for s in r.env_setup_steps:
            texts += [s.title, s.desc]
    word_count = sum(len(t.split()) for t in texts if t)
    return max(1, round(word_count / 200))


# --------------------------------------------------------------------------- #
# Q&A / reasoning / tutoring
# --------------------------------------------------------------------------- #
class Summary(BaseModel):
    """A one-call TL;DR of a paper."""

    title: str = ""
    tldr: str = ""
    key_points: List[str] = Field(default_factory=list)


class ComparedPaper(BaseModel):
    """One paper's row in a multi-paper comparison."""

    title: str = ""
    arxiv_id: Optional[str] = None
    year: Optional[int] = None
    main_contribution: str = ""
    novelty: str = ""
    methods: List[str] = Field(default_factory=list)
    benchmark: Optional[str] = None
    hardware: Optional[str] = None
    official_code: Optional[str] = None


class Comparison(BaseModel):
    """Side-by-side comparison of 2+ papers."""

    papers: List[ComparedPaper] = Field(default_factory=list)
    synthesis: str = ""
    usage: Optional["Usage"] = None

    def to_dict(self) -> dict:
        return {"papers": [p.model_dump() for p in self.papers], "synthesis": self.synthesis}

    def to_json(self, path: Optional[str] = None) -> str:
        import json

        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path:
            _write_text(path, text)
        return text

    def to_markdown(self, path: Optional[str] = None) -> str:
        from papermind.output.compare_render import to_markdown as _md

        text = _md(self)
        if path:
            _write_text(path, text)
        return text

    def to_html(self, path: Optional[str] = None) -> str:
        from papermind.output.compare_render import to_html as _html

        text = _html(self)
        if path:
            _write_text(path, text)
        return text


class AnswerSegment(BaseModel):
    kind: SegmentKind
    text: str
    confidence: Optional[Confidence] = None
    reasoning: Optional[str] = Field(
        None, description="For inference segments: the basis the inference rests on"
    )


class EvidenceItem(BaseModel):
    text: str
    section: Optional[str] = None
    page: Optional[int] = None
    verified: bool = True  # True if the snippet was matched against a retrieved passage


class SourceRef(BaseModel):
    section: Optional[str] = None
    page: Optional[int] = None


class Answer(BaseModel):
    question: str
    segments: List[AnswerSegment] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    usage: Optional["Usage"] = None  # token usage for this turn

    @property
    def text(self) -> str:
        """Plain concatenation of all segment texts."""
        return "\n\n".join(seg.text for seg in self.segments)
