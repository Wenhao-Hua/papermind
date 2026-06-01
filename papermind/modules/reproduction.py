"""Module 4: reproduction guide (the most detailed module)."""

from __future__ import annotations

from papermind.llm.base import LLMClient
from papermind.llm.prompts import ANALYSIS_SYSTEM, reproduction_user
from papermind.modules._util import int_or_none, str_list, str_or_none
from papermind.output.schema import (
    Benchmark,
    CommonError,
    Dataset,
    Reproduction,
    SetupStep,
)
from papermind.parser.pdf import ParsedPaper


def run(parsed: ParsedPaper, client: LLMClient, context: str) -> Reproduction:
    data = client.complete_json(ANALYSIS_SYSTEM, reproduction_user(context))
    if not isinstance(data, dict):
        return Reproduction()

    return Reproduction(
        official_code=str_or_none(data.get("official_code")),
        version_tag=str_or_none(data.get("version_tag")),
        requirements=str_or_none(data.get("requirements")),
        recommended_hardware=str_or_none(data.get("recommended_hardware")),
        key_hyperparams=str_list(data.get("key_hyperparams")),
        env_setup_steps=_setup_steps(data.get("env_setup_steps")),
        performance_benchmarks=_benchmarks(data.get("performance_benchmarks")),
        datasets=_datasets(data.get("datasets")),
        common_errors=_common_errors(data.get("common_errors")),
        gotchas=str_list(data.get("gotchas")),
    )


def _setup_steps(value) -> list:
    if not isinstance(value, list):
        return []
    steps = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict) or not item.get("title"):
            continue
        steps.append(
            SetupStep(
                step=int_or_none(item.get("step")) or idx,
                title=str(item["title"]),
                desc=str(item.get("desc", "") or ""),
                command=str_or_none(item.get("command")),
            )
        )
    return steps


def _benchmarks(value) -> list:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict) or not item.get("setting"):
            continue
        out.append(
            Benchmark(
                setting=str(item["setting"]),
                baseline=str_or_none(item.get("baseline")),
                result=str_or_none(item.get("result")),
                speedup=str_or_none(item.get("speedup")),
                memory=str_or_none(item.get("memory")),
            )
        )
    return out


def _datasets(value) -> list:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        out.append(
            Dataset(
                name=str(item["name"]),
                purpose=str(item.get("purpose", "") or ""),
                link=str_or_none(item.get("link")),
            )
        )
    return out


def _common_errors(value) -> list:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict) or not item.get("error"):
            continue
        out.append(
            CommonError(
                error=str(item["error"]),
                cause=str(item.get("cause", "") or ""),
                fix_command=str_or_none(item.get("fix_command")),
            )
        )
    return out
