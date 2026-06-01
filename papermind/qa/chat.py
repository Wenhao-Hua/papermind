"""PaperChat: grounded multi-turn Q&A with fact/inference layering, plus tutor
and debug modes.

Indexing happens once (then cached). Every answer is built from retrieved
passages and labeled by source:论文事实 (fact) / 基于论文的推理 (inference, with a
confidence level) / 超出论文范围 (out_of_scope). The model is instructed never to
fabricate when the paper lacks support.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from papermind.config import Config, load_config
from papermind.llm.base import LLMClient
from papermind.llm import prompts
from papermind.modules._util import int_or_none, str_or_none
from papermind.output.schema import Answer, AnswerSegment, EvidenceItem, Report, SourceRef
from papermind.parser.arxiv import resolve
from papermind.parser.pdf import parse_pdf
from papermind.qa.index import PaperIndex
from papermind.qa.retriever import Retriever

_VALID_KINDS = {"fact", "inference", "out_of_scope"}
_VALID_CONFIDENCE = {"high", "mid", "low"}
_HISTORY_TURNS = 6  # number of prior messages (user+assistant) kept for context


class PaperChat:
    """Stateful chat session bound to one paper."""

    def __init__(
        self,
        source: str,
        model: Optional[str] = None,
        mode: str = "balanced",
        config: Optional[Config] = None,
        console=None,
    ) -> None:
        self.config = config or load_config()
        self.mode = mode
        notice = (lambda msg: console.print(f"[dim]{msg}[/dim]")) if console is not None else None
        self.client = LLMClient(model=model, config=self.config, on_notice=notice)

        resolved = resolve(source, self.config)
        # Chat only needs text for indexing — skip the (potentially costly) figure work.
        self.parsed = parse_pdf(
            resolved.pdf_path, resolved.meta, resolved.cache_dir, extract_figures=False
        )
        self.report = Report(paper=self.parsed.meta)

        if console is not None:
            with console.status("[cyan]建立/加载向量索引…[/cyan]"):
                self.index = PaperIndex.load_or_build(self.parsed, self.client)
        else:
            self.index = PaperIndex.load_or_build(self.parsed, self.client)
        self.retriever = Retriever(self.index, self.client)

        self._history: List[dict] = []
        self._last_user: str = ""

    @property
    def usage(self):
        """Cumulative token usage for this chat session."""
        return self.client.usage

    # -- public API --------------------------------------------------------- #
    def ask(self, question: str, k: int = 5, on_delta=None, section: Optional[str] = None) -> Answer:
        focused = f"{question}（请聚焦 Section {section}）" if section else question
        return self._answer(
            system=prompts.qa_system(self.mode),
            make_user=lambda passages: prompts.qa_user(focused, passages),
            retrieval_query=question,
            question_text=question,
            k=k,
            on_delta=on_delta,
            section=section,
        )

    def tutor(self, question: str, k: int = 5, on_delta=None) -> Answer:
        return self._answer(
            system=prompts.tutor_system(self.mode if self.mode != "strict" else "balanced"),
            make_user=lambda passages: prompts.qa_user(question, passages),
            retrieval_query=question,
            question_text=question,
            k=k,
            on_delta=on_delta,
        )

    def debug(self, error: str, k: int = 6, on_delta=None) -> Answer:
        return self._answer(
            system=prompts.tutor_system("explore"),
            make_user=lambda passages: prompts.debug_user(error, passages),
            retrieval_query=error,
            question_text=f"[debug] {error[:80]}",
            k=k,
            on_delta=on_delta,
        )

    # -- internals ---------------------------------------------------------- #
    def _answer(
        self,
        system: str,
        make_user: Callable[[str], str],
        retrieval_query: str,
        question_text: str,
        k: int,
        on_delta=None,
        section: Optional[str] = None,
    ) -> Answer:
        # Follow-up questions retrieve better with a touch of prior context.
        query = f"{self._last_user}\n{retrieval_query}" if self._last_user else retrieval_query
        results = self.retriever.retrieve(query, k=k, section=section)
        passages = Retriever.format_passages(results)

        messages = (
            [{"role": "system", "content": system}]
            + self._history[-_HISTORY_TURNS:]
            + [{"role": "user", "content": make_user(passages)}]
        )
        before = self.client.usage.snapshot()
        data = self.client.complete_json_messages(messages, on_delta=on_delta)
        answer = self._parse_answer(question_text, data)
        _verify_evidence(answer.evidence, [chunk for chunk, _ in results])
        answer.usage = self.client.usage.minus(before)

        self._history.append({"role": "user", "content": retrieval_query})
        self._history.append({"role": "assistant", "content": answer.text})
        self._last_user = retrieval_query
        return answer

    def _parse_answer(self, question: str, data) -> Answer:
        if not isinstance(data, dict):
            return Answer(
                question=question,
                segments=[AnswerSegment(kind="out_of_scope", text="未能从模型获得结构化回答。")],
            )

        segments = []
        for seg in data.get("segments", []) or []:
            if not isinstance(seg, dict) or not seg.get("text"):
                continue
            kind = str(seg.get("kind", "")).lower()
            if kind not in _VALID_KINDS:
                kind = "inference"
            if self.mode == "strict" and kind == "inference":
                continue  # strict mode: drop derived content entirely
            confidence = str(seg.get("confidence") or "").lower() or None
            if confidence not in _VALID_CONFIDENCE:
                confidence = None
            segments.append(
                AnswerSegment(
                    kind=kind,
                    text=str(seg["text"]),
                    confidence=confidence,
                    reasoning=str_or_none(seg.get("reasoning")),
                )
            )

        if not segments:
            segments = [AnswerSegment(kind="out_of_scope", text="原文未提及，且无法据此推理。")]

        evidence = [
            EvidenceItem(
                text=str(e["text"]),
                section=str_or_none(e.get("section")),
                page=int_or_none(e.get("page")),
            )
            for e in (data.get("evidence", []) or [])
            if isinstance(e, dict) and e.get("text")
        ]
        sources = [
            SourceRef(section=str_or_none(s.get("section")), page=int_or_none(s.get("page")))
            for s in (data.get("sources", []) or [])
            if isinstance(s, dict)
        ]
        return Answer(question=question, segments=segments, evidence=evidence, sources=sources)


# --------------------------------------------------------------------------- #
# Citation verification: check each cited snippet actually came from a retrieved
# passage, and snap its section/page to that passage (parser-derived, reliable).
# Shared with the analysis-module citations — see papermind.qa.verify.
# --------------------------------------------------------------------------- #
def _verify_evidence(evidence, chunks) -> None:
    from papermind.qa.verify import verify_items

    verify_items(evidence, chunks)
