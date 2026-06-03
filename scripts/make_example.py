"""Build / fix gallery example reports so they render on GitHub.

`papermind analyze --format md` embeds teaching SVGs as base64 data-URIs and
original paper figures by their *absolute* local cache path — neither renders in
GitHub's markdown view. This tool rewrites a report into a self-contained gallery
entry: every figure is extracted/copied into ``examples/figures/<slug>-*.{svg,png}``
and referenced by a repo-relative path.

Usage:
  python scripts/make_example.py analyze <arxiv_url> <slug>   # run the pipeline (live) + process
  python scripts/make_example.py fix <slug>                   # repair an existing examples/<slug>.md in place
"""

from __future__ import annotations

import base64
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
FIGURES = EXAMPLES / "figures"

# alt text may itself contain ']' (e.g. citation markers "[39]"), so match lazily up
# to the real "](" rather than forbidding ']' in the alt.
_IMG_RE = re.compile(r"!\[(.*?)\]\(([^)]+)\)")


def _process(md: str, slug: str) -> str:
    """Rewrite every image reference to a repo-relative examples/figures/ path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    svg_n = 0
    png_map: dict = {}  # resolved source path -> relative name, so a figure referenced twice
    #                     (same original matched to two points) reuses ONE file, not orig1==orig2.

    def repl(m: re.Match) -> str:
        nonlocal svg_n
        caption, target = m.group(1), m.group(2).strip()

        if target.startswith("data:image/svg+xml;base64,"):
            svg_n += 1
            raw = base64.b64decode(target.split(",", 1)[1]).decode("utf-8", "replace")
            name = f"{slug}-fig{svg_n}.svg"
            (FIGURES / name).write_text(raw, encoding="utf-8")
            return f"![{caption}](figures/{name})"

        if target.startswith("figures/"):
            return m.group(0)  # already a relative gallery path -> leave it

        src = Path(target)
        if src.is_absolute() and src.exists():  # original paper figure copied from the cache
            key = str(src.resolve())
            if key not in png_map:
                name = f"{slug}-orig{len(png_map) + 1}{src.suffix.lower()}"
                shutil.copyfile(src, FIGURES / name)
                png_map[key] = name
            return f"![{caption}](figures/{png_map[key]})"

        return m.group(0)  # unknown / missing -> leave untouched (don't fabricate a link)

    return _IMG_RE.sub(repl, md)


def _clear_old(slug: str) -> None:
    for f in FIGURES.glob(f"{slug}-*"):
        f.unlink()


def cmd_analyze(url: str, slug: str, refresh: bool = False) -> None:
    from papermind.analyze import analyze

    print(f"[make_example] analyzing {url} (live, with SVG figures, refresh={refresh})…", flush=True)
    report = analyze(url, with_figures=True, svg_figures=True, refresh=refresh)
    _clear_old(slug)
    md = _process(report.to_markdown(), slug)
    (EXAMPLES / f"{slug}.md").write_text(md, encoding="utf-8")
    n_svg = len(list(FIGURES.glob(f"{slug}-fig*.svg")))
    n_png = len(list(FIGURES.glob(f"{slug}-orig*")))
    print(f"[make_example] wrote examples/{slug}.md  (+{n_svg} teaching SVG, +{n_png} original figure)")


def cmd_fix(slug: str) -> None:
    path = EXAMPLES / f"{slug}.md"
    md = path.read_text(encoding="utf-8")
    before = len(_IMG_RE.findall(md))
    md = _process(md, slug)
    path.write_text(md, encoding="utf-8")
    abs_left = sum(1 for _, t in _IMG_RE.findall(md) if Path(t.strip()).is_absolute())
    print(f"[make_example] fixed examples/{slug}.md  ({before} images, {abs_left} absolute paths remaining)")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "analyze":
        cmd_analyze(sys.argv[2], sys.argv[3], refresh="--refresh" in sys.argv[4:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "fix":
        cmd_fix(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
