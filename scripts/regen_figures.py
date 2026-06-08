"""Regenerate ONLY the teaching SVG figures of the gallery examples, using the current
(fixed) SVG figure prompt — no re-analysis. Loads each cached report, clears its
AI-generated SVG figures (keeps matched original figures), regenerates them with the
figure model, and rewrites examples/<slug>.md via the make_example processor.

Usage:  python scripts/regen_figures.py [slug ...]   (default: all)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_example as mk  # noqa: E402

PAIRS = [
    ("https://arxiv.org/abs/1706.03762", "transformer"),
    ("https://arxiv.org/abs/2307.08691", "flashattention2"),
    ("https://arxiv.org/abs/2307.09288", "llama2"),
    ("https://arxiv.org/abs/2010.11929", "vit"),
    ("https://arxiv.org/abs/2112.10752", "latent-diffusion"),
    ("https://arxiv.org/abs/1707.06347", "ppo"),
    ("https://arxiv.org/abs/2106.09685", "lora"),
]


def regen(url: str, slug: str) -> None:
    from papermind.analyze import ALL_MODULES, _build_context
    from papermind.cache import load_cached_report, report_cache_path
    from papermind.config import load_config
    from papermind.figures.generate import generate_svg_diagrams
    from papermind.llm.base import LLMClient
    from papermind.parser.arxiv import resolve
    from papermind.parser.pdf import parse_pdf

    cfg = load_config()
    resolved = resolve(url, cfg)
    cache_path = report_cache_path(resolved.cache_dir, LLMClient(config=cfg).model, ALL_MODULES, "svg")
    report = load_cached_report(cache_path)
    if report is None:
        print(f"[{slug}] NO cached svg report — skip (run make_example analyze first)")
        return
    parsed = parse_pdf(resolved.pdf_path, resolved.meta, resolved.cache_dir, extract_figures=True)
    ctx = _build_context(parsed)  # no client -> extractive sampling, no model call

    cleared = 0
    for p in report.technical.details:
        if p.figure is not None and getattr(p.figure, "svg", None):
            p.figure = None  # regenerate AI SVGs; keep matched 'original' figures
            cleared += 1
    fig_client = LLMClient(model=cfg.figure_model, config=cfg)
    generate_svg_diagrams(report.technical.details, fig_client, ctx, reasoning_effort=None, max_tokens=18000)

    mk._clear_old(slug)
    md = mk._process(report.to_markdown(), slug)
    (mk.EXAMPLES / f"{slug}.md").write_text(md, encoding="utf-8")
    n_svg = len(list(mk.FIGURES.glob(f"{slug}-fig*.svg")))
    print(f"[{slug}] cleared {cleared} AI-SVG -> regenerated; {n_svg} teaching SVGs now")


if __name__ == "__main__":
    want = sys.argv[1:]
    for url, slug in PAIRS:
        if want and slug not in want:
            continue
        try:
            regen(url, slug)
        except Exception as exc:  # noqa: BLE001
            print(f"[{slug}] FAILED: {exc!r}")
        time.sleep(20)  # pace to avoid DeepSeek burst rate-limiting
    print("REGEN-FIGURES DONE")
