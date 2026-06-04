"""Framework-diagram spec: deterministic SVG renderer, loose JSON parser, auto-layout."""

from __future__ import annotations

from xml.etree import ElementTree as ET

from papermind.figures.framework import (
    FEdge,
    FLegend,
    FNode,
    FrameworkSpec,
    _auto_layout,
    _spec_from_data,
    _wrap,
    render_framework_svg,
)


def _sample() -> FrameworkSpec:
    return _auto_layout(FrameworkSpec(
        title="T",
        nodes=[
            FNode(id="a", kind="io", label="输入 x₀"),
            FNode(id="b", kind="box", label="阶段", lines=["xₜ = √āₜ x₀ + ε"]),
            FNode(id="c", kind="group", inferred=True, col=1, label="训练目标", lines=["L = ‖ε-ε_θ‖²"]),
        ],
        edges=[FEdge(src="a", dst="b"), FEdge(src="b", dst="c", style="dashed")],
        legend=[FLegend(style="solid", text="数据流")],
        note="āₜ = Πα",
    ))


def test_render_is_well_formed_svg():
    svg = render_framework_svg(_sample())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    ET.fromstring(svg)  # raises if not valid XML
    assert "输入 x₀" in svg and "训练目标" in svg and "数据流" in svg


def test_render_escapes_xml():
    svg = render_framework_svg(_auto_layout(FrameworkSpec(nodes=[FNode(id="n", label="A & B < C")])))
    ET.fromstring(svg)
    assert "&amp;" in svg and "&lt;" in svg
    assert "A & B" not in svg  # the raw ampersand must not survive


def test_inferred_node_renders_dashed():
    svg = render_framework_svg(_auto_layout(FrameworkSpec(nodes=[FNode(id="n", inferred=True, label="x")])))
    assert "stroke-dasharray" in svg


def test_spec_from_data_tolerates_from_to_and_missing_fields():
    spec = _spec_from_data({
        "title": "X",
        "nodes": [
            {"id": "a", "label": "A"},
            {"id": "b", "kind": "io", "col": 1, "lines": ["l1", ""]},
            {"label": "no id -> dropped"},
        ],
        "edges": [
            {"from": "a", "to": "b", "style": "emph"},  # from/to accepted
            {"src": "a", "dst": "ghost"},               # bad dst -> dropped
        ],
        "legend": [{"style": "solid", "text": "flow"}],
    })
    assert spec is not None
    assert [n.id for n in spec.nodes] == ["a", "b"]  # the id-less node is dropped
    assert spec.nodes[1].col == 1 and spec.nodes[1].lines == ["l1"]  # blank line stripped
    assert len(spec.edges) == 1 and spec.edges[0].src == "a" and spec.edges[0].style == "emph"


def test_spec_from_data_rejects_garbage():
    assert _spec_from_data("not a dict") is None
    assert _spec_from_data({"nodes": []}) is None  # no usable nodes


def test_auto_layout_positions():
    spec = _auto_layout(FrameworkSpec(width=960, nodes=[
        FNode(id="a", kind="io", label="in"),
        FNode(id="b", kind="box", label="mid"),
        FNode(id="c", kind="group", col=1, label="side"),
    ]))
    a, b, c = spec.nodes
    assert a.y < b.y                       # main column stacks downward
    assert abs((a.x + a.w / 2) - 480) < 1  # main column centered on a 960 canvas
    assert c.x > 480                       # side column sits on the right
    assert a.w and b.w and c.w             # default widths assigned by kind


def test_wrap_breaks_long_text():
    lines = _wrap("word " * 40, 120, 12.0)
    assert len(lines) > 1 and all(lines)
