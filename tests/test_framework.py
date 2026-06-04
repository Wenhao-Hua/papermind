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


# --------------------------------------------------------------------------- #
# LLM generation + persistence
# --------------------------------------------------------------------------- #
class _FakeJSONClient:
    def __init__(self, value=None, raises=None):
        self._value, self._raises = value, raises

    def complete_json(self, system, user, max_tokens=4000):
        if self._raises is not None:
            raise self._raises
        return self._value


def test_generate_framework_builds_and_lays_out():
    from papermind.figures.framework import generate_framework

    client = _FakeJSONClient({"title": "T", "nodes": [{"id": "a", "label": "In", "kind": "io"},
                                                      {"id": "b", "label": "Stage"}],
                              "edges": [{"src": "a", "dst": "b"}]})
    spec = generate_framework("ctx", client)
    assert spec is not None and spec.title == "T"
    assert [n.id for n in spec.nodes] == ["a", "b"]
    assert all(n.w and n.x for n in spec.nodes)  # auto-layout assigned widths + positions


def test_generate_framework_none_on_garbage():
    from papermind.figures.framework import generate_framework

    notices = []
    assert generate_framework("ctx", _FakeJSONClient(["not", "a", "spec"]), on_notice=notices.append) is None
    assert notices  # the user is told it failed


def test_generate_framework_none_on_llm_error():
    from papermind.errors import LLMError
    from papermind.figures.framework import generate_framework

    notices = []
    assert generate_framework("ctx", _FakeJSONClient(raises=LLMError("boom")), on_notice=notices.append) is None
    assert notices


def test_framework_spec_persistence_roundtrip(tmp_path):
    from papermind.figures.framework import load_framework_spec, save_framework_spec

    spec = _sample()
    save_framework_spec(tmp_path, spec)
    loaded = load_framework_spec(tmp_path)
    assert loaded is not None
    assert loaded.title == spec.title and [n.id for n in loaded.nodes] == [n.id for n in spec.nodes]
    assert abs(loaded.nodes[0].x - spec.nodes[0].x) < 0.01  # dragged/laid-out positions preserved


def test_framework_spec_load_missing_and_corrupt(tmp_path):
    from papermind.figures.framework import _framework_path, load_framework_spec

    assert load_framework_spec(tmp_path) is None  # no file yet
    _framework_path(tmp_path).write_text("{not valid json", encoding="utf-8")
    assert load_framework_spec(tmp_path) is None  # corrupt -> None so the caller regenerates
