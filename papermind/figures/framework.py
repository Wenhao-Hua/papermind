"""The paper's whole-method *framework diagram* as an editable structured spec.

Unlike the per-point teaching SVGs (which the LLM draws as raw SVG), the framework
diagram is a node/edge **spec** — a single source of truth that powers three things
from one representation:

  * generation       — the LLM emits the spec (figures.generate side stays untouched);
  * conversational edit — the LLM rewrites the spec from an instruction;
  * a visual editor  — the browser drags/renames/adds nodes by mutating the spec.

This module owns the spec schema and a deterministic ``spec -> paper-style SVG``
renderer (so the editor can re-render after every change without an LLM call).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Spec schema
# --------------------------------------------------------------------------- #
_KINDS = ("io", "box", "group")          # input/output · main module · (inferred) panel
_STYLES = ("solid", "emph", "dashed")     # data flow · highlighted/selected · inferred/reference


class FNode(BaseModel):
    id: str
    label: str = ""                       # bold first line
    lines: List[str] = Field(default_factory=list)  # extra lines (sub-labels, formulas)
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    kind: str = "box"
    col: int = 0                          # 0 = main vertical flow, 1 = right-hand panel
    inferred: bool = False                # un-illustrated step the paper only implies -> dashed


class FEdge(BaseModel):
    src: str
    dst: str
    style: str = "solid"
    label: str = ""


class FLegend(BaseModel):
    style: str = "solid"
    text: str = ""


class FrameworkSpec(BaseModel):
    title: str = ""
    width: int = 960
    height: int = 680
    nodes: List[FNode] = Field(default_factory=list)
    edges: List[FEdge] = Field(default_factory=list)
    legend: List[FLegend] = Field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------------------- #
# Palette (matches the per-point teaching SVGs: indigo accent + neutral grays)
# --------------------------------------------------------------------------- #
_FILL = {"io": "#f1f5f9", "box": "#eef2ff", "group": "#fbfcfe"}
_STROKE = {"io": "#475569", "box": "#6366f1", "group": "#94a3b8"}
_EDGE = {"solid": ("#475569", 2.0, ""), "emph": ("#6366f1", 2.6, ""), "dashed": ("#94a3b8", 1.8, "6,5")}
_INK, _SUB = "#1e293b", "#334155"
_PAD, _LABEL_FS, _LINE_FS, _LH = 12.0, 15.0, 12.5, 1.5
_FONT = "'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif"


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _char_w(ch: str, fs: float) -> float:
    return fs * (1.0 if ord(ch) > 0x2E7F else 0.55)  # CJK ~ full-width, latin ~0.55


def _wrap(text: str, max_w: float, fs: float) -> List[str]:
    """Greedy wrap to ``max_w`` px; break at spaces when possible, else mid-run (CJK)."""
    out, line, w, last_space = [], "", 0.0, -1
    for ch in str(text):
        cw = _char_w(ch, fs)
        if ch == " ":
            last_space = len(line)
        if w + cw > max_w and line:
            if last_space > 0 and last_space < len(line):
                out.append(line[:last_space]); line = line[last_space + 1:]
            else:
                out.append(line); line = ""
            w = sum(_char_w(c, fs) for c in line)
            last_space = -1
        line += ch
        w += cw
    if line:
        out.append(line)
    return out or [""]


def _node_rows(node: FNode):
    """Wrapped (text, font_size, bold) rows for a node, and the box height they need."""
    inner = max(node.w - 2 * _PAD, 40.0)
    rows = []
    for piece in _wrap(node.label, inner, _LABEL_FS):
        rows.append((piece, _LABEL_FS, True))
    for ln in node.lines:
        for piece in _wrap(ln, inner, _LINE_FS):
            rows.append((piece, _LINE_FS, False))
    height = 2 * _PAD + sum(fs * _LH for _, fs, _ in rows)
    return rows, max(height, 44.0)


def _box(node: FNode):
    """(x, y, w, h, cx, cy) for a node, height derived from its content."""
    _, h = _node_rows(node)
    return node.x, node.y, node.w, h, node.x + node.w / 2, node.y + h / 2


def _edge_point(box, tx: float, ty: float):
    """Where the ray from a box's center toward (tx, ty) crosses the box boundary."""
    x, y, w, h, cx, cy = box
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, y + h
    sx = (w / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def render_framework_svg(spec: FrameworkSpec) -> str:
    """Render a :class:`FrameworkSpec` to a self-contained, paper-style ``<svg>``."""
    W = spec.width or 960
    boxes = {n.id: _box(n) for n in spec.nodes}
    bottom = max((b[1] + b[3] for b in boxes.values()), default=120.0)
    legend_y = bottom + 40
    note_rows = _wrap(spec.note, W - 80, 12.0) if spec.note else []
    H = int(legend_y + (28 if spec.legend else 0) + len(note_rows) * 18 + 36)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="{_FONT}" width="{W}" height="{H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    if spec.title:
        out.append(
            f'<text x="{W/2}" y="44" text-anchor="middle" font-size="25" font-weight="700" '
            f'fill="{_INK}">{_esc(spec.title)}</text>'
        )
        out.append(f'<line x1="80" y1="62" x2="{W-80}" y2="62" stroke="#e2e8f0" stroke-width="1.5"/>')

    # edges first (so nodes sit on top of arrow tails)
    for e in spec.edges:
        if e.src not in boxes or e.dst not in boxes:
            continue
        a, b = boxes[e.src], boxes[e.dst]
        x1, y1 = _edge_point(a, b[4], b[5])
        x2, y2 = _edge_point(b, a[4], a[5])
        color, sw, dash = _EDGE.get(e.style, _EDGE["solid"])
        da = f' stroke-dasharray="{dash}"' if dash else ""
        # shorten the line so the arrowhead tip (not the line) lands on the box edge
        import math

        ang = math.atan2(y2 - y1, x2 - x1)
        hx, hy = x2 - 11 * math.cos(ang), y2 - 11 * math.sin(ang)
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{color}" stroke-width="{sw}"{da}/>')
        p = 6
        out.append(
            f'<polygon points="{x2:.1f},{y2:.1f} '
            f'{x2-11*math.cos(ang)+p*math.sin(ang):.1f},{y2-11*math.sin(ang)-p*math.cos(ang):.1f} '
            f'{x2-11*math.cos(ang)-p*math.sin(ang):.1f},{y2-11*math.sin(ang)+p*math.cos(ang):.1f}" fill="{color}"/>'
        )
        if e.label:
            out.append(
                f'<text x="{(x1+x2)/2:.1f}" y="{(y1+y2)/2-4:.1f}" text-anchor="middle" '
                f'font-size="11" fill="{_SUB}">{_esc(e.label)}</text>'
            )

    # nodes
    for n in spec.nodes:
        rows, h = _node_rows(n)
        kind = n.kind if n.kind in _KINDS else "box"
        fill = _FILL[kind]
        stroke = _STROKE[kind]
        dash = ' stroke-dasharray="7,5"' if (n.inferred or kind == "group") else ""
        sw = 1.5 if kind == "group" else 1.8
        out.append(
            f'<rect x="{n.x:.1f}" y="{n.y:.1f}" width="{n.w:.1f}" height="{h:.1f}" rx="11" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>'
        )
        ty = n.y + _PAD + rows[0][1]
        for text, fs, bold in rows:
            weight = ' font-weight="700"' if bold else ""
            color = _INK if bold else _SUB
            out.append(
                f'<text x="{n.x+n.w/2:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="{fs}"{weight} '
                f'fill="{color}">{_esc(text)}</text>'
            )
            ty += fs * _LH

    # legend
    if spec.legend:
        lx = 60.0
        for item in spec.legend:
            color, sw, dash = _EDGE.get(item.style, _EDGE["solid"])
            da = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<line x1="{lx:.0f}" y1="{legend_y:.0f}" x2="{lx+34:.0f}" y2="{legend_y:.0f}" stroke="{color}" stroke-width="{sw}"{da}/>')
            out.append(f'<text x="{lx+42:.0f}" y="{legend_y+4:.0f}" font-size="12" fill="{_SUB}">{_esc(item.text)}</text>')
            lx += 44 + sum(_char_w(c, 12) for c in item.text) + 28
    # note
    ny = legend_y + (28 if spec.legend else 0) + 14
    for row in note_rows:
        out.append(f'<text x="60" y="{ny:.0f}" font-size="12" fill="{_SUB}">{_esc(row)}</text>')
        ny += 18

    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Layout + generation
# --------------------------------------------------------------------------- #
def _auto_layout(spec: FrameworkSpec) -> FrameworkSpec:
    """Assign non-overlapping x/y from each node's column + order (the LLM supplies the
    logical structure, not pixel coordinates). A later visual edit overrides these."""
    width = spec.width or 960
    gap = 44.0
    y0 = 86.0 if spec.title else 30.0
    y_main, y_side = y0, y0 + 120.0
    for n in spec.nodes:
        if not n.w:
            n.w = 280.0 if n.kind == "io" else (220.0 if n.col else 360.0)
    for n in spec.nodes:
        _, h = _node_rows(n)
        if n.col:  # right-hand column (typically the inferred panels)
            n.x, n.y = width - n.w - 40.0, y_side
            y_side += h + gap
        else:      # main vertical flow, centered
            n.x, n.y = (width - n.w) / 2, y_main
            y_main += h + gap
    return spec


def _spec_from_data(data) -> Optional[FrameworkSpec]:
    """Build a :class:`FrameworkSpec` from loosely-shaped LLM JSON (tolerates from/to
    instead of src/dst, missing fields, extra keys). Returns None if unusable."""
    if not isinstance(data, dict):
        return None
    nodes = []
    for n in data.get("nodes") or []:
        if not isinstance(n, dict) or not n.get("id"):
            continue
        nodes.append(FNode(
            id=str(n["id"]),
            label=str(n.get("label") or ""),
            lines=[str(x) for x in (n.get("lines") or []) if str(x).strip()],
            kind=str(n.get("kind") or "box"),
            col=1 if n.get("col") in (1, "1", True) else 0,
            inferred=bool(n.get("inferred")),
        ))
    if not nodes:
        return None
    ids = {n.id for n in nodes}
    edges = []
    for e in data.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src, dst = e.get("src") or e.get("from"), e.get("dst") or e.get("to")
        if src in ids and dst in ids:
            edges.append(FEdge(src=str(src), dst=str(dst), style=str(e.get("style") or "solid"),
                               label=str(e.get("label") or "")))
    legend = [FLegend(style=str(li.get("style") or "solid"), text=str(li.get("text") or ""))
              for li in (data.get("legend") or []) if isinstance(li, dict) and li.get("text")]
    return FrameworkSpec(title=str(data.get("title") or ""), nodes=nodes, edges=edges,
                         legend=legend, note=str(data.get("note") or ""))


def generate_framework(context: str, client, on_notice=None) -> Optional[FrameworkSpec]:
    """Ask the model for the paper's whole-method framework as a spec, lay it out, and
    return it. Best-effort: any failure returns None (the report simply has no diagram)."""
    from papermind.errors import LLMError
    from papermind.llm.prompts import FRAMEWORK_SYSTEM, framework_user

    try:
        data = client.complete_json(FRAMEWORK_SYSTEM, framework_user(context), max_tokens=4000)
    except LLMError as exc:
        if on_notice:
            on_notice(f"框架图生成失败：{exc}")
        return None
    spec = _spec_from_data(data)
    if spec is None:
        if on_notice:
            on_notice("框架图生成失败：模型未返回有效结构。")
        return None
    return _auto_layout(spec)


def edit_framework(spec: FrameworkSpec, instruction: str, client, on_notice=None) -> Optional[FrameworkSpec]:
    """Rewrite a framework spec from a natural-language instruction (conversational edit).
    Re-lays-out the result; returns None on failure (the caller keeps the old spec)."""
    from papermind.errors import LLMError
    from papermind.llm.prompts import FRAMEWORK_EDIT_SYSTEM, framework_edit_user

    try:
        data = client.complete_json(
            FRAMEWORK_EDIT_SYSTEM, framework_edit_user(spec.model_dump(), instruction), max_tokens=4000
        )
    except LLMError as exc:
        if on_notice:
            on_notice(f"改图失败：{exc}")
        return None
    new = _spec_from_data(data)
    return _auto_layout(new) if new is not None else None


# --------------------------------------------------------------------------- #
# Persistence (one framework.json per paper, in its cache dir)
# --------------------------------------------------------------------------- #
def _framework_path(cache_dir):
    from pathlib import Path

    return Path(cache_dir) / "framework.json"


def save_framework_spec(cache_dir, spec: FrameworkSpec) -> None:
    _framework_path(cache_dir).write_text(spec.model_dump_json(indent=2), encoding="utf-8")


def load_framework_spec(cache_dir) -> Optional[FrameworkSpec]:
    path = _framework_path(cache_dir)
    if not path.exists():
        return None
    try:
        return FrameworkSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupt cache -> regenerate
        return None
