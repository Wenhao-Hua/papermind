"""Resolve an input source (arXiv URL / id / local PDF) to a downloaded PDF + metadata.

Results are cached under ``~/.papermind/cache/<key>/`` so re-running analysis or
chat on the same paper never re-downloads or re-queries arXiv.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from papermind.config import Config, load_config
from papermind.errors import SourceError
from papermind.output.schema import PaperMeta

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "PaperMind/0.1 (https://github.com/Wenhao-Hua/papermind)"

# 2307.08691 / 2307.08691v2  (new) and hep-th/9901001 (old) schemes.
_NEW_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD_ID = r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ID_RE = re.compile(rf"(?:arxiv:)?({_NEW_ID}|{_OLD_ID})", re.IGNORECASE)
_ABS_URL_RE = re.compile(rf"arxiv\.org/(?:abs|pdf)/({_NEW_ID}|{_OLD_ID})", re.IGNORECASE)


@dataclass
class ResolvedSource:
    """Everything downstream stages need to start parsing a paper."""

    meta: PaperMeta
    pdf_path: Path
    cache_key: str
    cache_dir: Path


def _query_arxiv(params: dict) -> str:
    """GET the arXiv API with retry/backoff on 429 (rate limit) and 5xx."""
    import time

    import httpx

    for attempt in range(3):
        try:
            resp = httpx.get(
                ARXIV_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
            )
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                retry_after = resp.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else 3.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            if attempt < 2:
                time.sleep(3.0 * (attempt + 1))
                continue
            raise SourceError(f"arXiv 请求失败（可能被限流，请稍后重试）：{exc}") from exc
    raise SourceError("arXiv 多次请求失败（可能被限流），请稍后再试。")


def search_arxiv(query: str, max_results: int = 10):
    """Search arXiv and return a list of :class:`PaperMeta` (no PDF download)."""
    text = _query_arxiv(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results, "sortBy": "relevance"}
    )
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise SourceError(f"arXiv 搜索返回无法解析的响应: {exc}") from exc

    results = []
    for entry in root.findall("atom:entry", ARXIV_ATOM_NS):
        id_node = entry.find("atom:id", ARXIV_ATOM_NS)
        if id_node is None or not id_node.text:
            continue
        m = re.search(r"abs/(.+)$", id_node.text.strip())
        arxiv_id = m.group(1) if m else None
        published = _text(entry.find("atom:published", ARXIV_ATOM_NS))
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [_text(a.find("atom:name", ARXIV_ATOM_NS)) for a in entry.findall("atom:author", ARXIV_ATOM_NS)]
        results.append(
            PaperMeta(
                title=_text(entry.find("atom:title", ARXIV_ATOM_NS)),
                arxiv_id=arxiv_id,
                authors=[a for a in authors if a],
                year=year,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None,
            )
        )
    return results


def parse_arxiv_id(source: str) -> Optional[str]:
    """Extract an arXiv id from an arXiv URL, or None.

    Bare ids and ``arxiv:ID`` are intentionally NOT accepted — paste a URL
    (``https://arxiv.org/abs/...``). Inputs are unified on URLs.
    """
    match = _ABS_URL_RE.search((source or "").strip())
    return match.group(1) if match else None


def cache_dir_for(source: str, config: Optional[Config] = None) -> Optional[Path]:
    """The cache dir a source *would* use — computed WITHOUT any network, so a
    cached-only lookup (e.g. demo mode) never downloads. Mirrors the keys used by
    the _resolve_* functions. Returns None for an unrecognized source."""
    config = config or load_config()
    source = (source or "").strip()
    arxiv_id = parse_arxiv_id(source)
    if arxiv_id:
        return config.paper_cache(arxiv_id.replace("/", "_"))
    if re.match(r"https?://", source, re.IGNORECASE):
        return config.paper_cache(f"url-{hashlib.sha1(source.encode('utf-8')).hexdigest()[:12]}")
    path = Path(source).expanduser()
    if path.exists():
        return config.paper_cache(f"local-{hashlib.sha1(str(path.resolve()).encode('utf-8')).hexdigest()[:10]}")
    return None


def resolve(source: str, config: Optional[Config] = None) -> ResolvedSource:
    """Resolve a source into a local PDF + best-effort metadata.

    Accepts: an arXiv URL (rich metadata), any other PDF URL (downloaded), or a
    local .pdf path. Bare arXiv ids are no longer accepted — give a URL.
    """
    config = config or load_config()
    source = (source or "").strip()
    if not source:
        raise SourceError("请提供论文 URL（arXiv 链接或 PDF 直链），或本地 PDF 路径。")

    arxiv_id = parse_arxiv_id(source)
    if arxiv_id:
        return _resolve_arxiv(arxiv_id, config)
    if re.match(r"https?://", source, re.IGNORECASE):
        return _resolve_url(source, config)
    if Path(source).expanduser().exists():
        return _resolve_local(source, config)
    raise SourceError(
        f"无法识别来源 {source!r}。请粘贴论文 URL（arXiv 链接或 PDF 直链）或本地 PDF 路径。"
        "已不再支持裸 arXiv id —— 例如用 https://arxiv.org/abs/2307.08691 代替 2307.08691。"
    )


# --------------------------------------------------------------------------- #
# Local PDF
# --------------------------------------------------------------------------- #
def _resolve_local(source: str, config: Config) -> ResolvedSource:
    path = Path(source).expanduser()
    if not path.exists():
        raise SourceError(
            f"无法识别来源 {source!r}：既不是可识别的 arXiv URL，该路径下也没有文件。"
        )
    if path.suffix.lower() != ".pdf":
        raise SourceError(f"本地来源必须是 .pdf 文件，收到的是：{path}")

    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    cache_key = f"local-{digest}"
    cache_dir = config.paper_cache(cache_key)
    meta = PaperMeta(title=path.stem, pdf_url=path.resolve().as_uri())
    return ResolvedSource(meta=meta, pdf_path=path, cache_key=cache_key, cache_dir=cache_dir)


# --------------------------------------------------------------------------- #
# Any other paper, by URL (non-arXiv PDF link)
# --------------------------------------------------------------------------- #
def _resolve_url(url: str, config: Config) -> ResolvedSource:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    cache_key = f"url-{digest}"
    cache_dir = config.paper_cache(cache_key)
    pdf_path = cache_dir / "paper.pdf"

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        _download_pdf(url, pdf_path)
    with open(pdf_path, "rb") as fh:
        if not fh.read(5).startswith(b"%PDF"):
            pdf_path.unlink(missing_ok=True)
            raise SourceError(f"{url} 看起来不是 PDF（可能是网页）。请提供 PDF 直链或 arXiv 链接。")

    meta = PaperMeta(title=_title_from_pdf(pdf_path, url), pdf_url=url)
    return ResolvedSource(meta=meta, pdf_path=pdf_path, cache_key=cache_key, cache_dir=cache_dir)


def _title_from_pdf(pdf_path: Path, url: str) -> str:
    """Best-effort title: infer from the PDF's first page, else the URL filename."""
    try:
        import fitz

        from papermind.parser.pdf import _infer_title

        doc = fitz.open(pdf_path)
        first = doc[0].get_text() if doc.page_count else ""
        doc.close()
        inferred = _infer_title([first])
        if inferred:
            return inferred
    except Exception:  # noqa: BLE001 - title is best-effort
        pass
    from urllib.parse import unquote, urlparse

    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = name.replace("_", " ").replace("-", " ").strip()
    return name or urlparse(url).netloc or "Untitled paper"


# --------------------------------------------------------------------------- #
# arXiv
# --------------------------------------------------------------------------- #
def _resolve_arxiv(arxiv_id: str, config: Config) -> ResolvedSource:
    cache_key = arxiv_id.replace("/", "_")
    cache_dir = config.paper_cache(cache_key)
    pdf_path = cache_dir / "paper.pdf"
    meta_path = cache_dir / "metadata.json"

    if meta_path.exists():
        meta = PaperMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
    else:
        meta = _fetch_arxiv_metadata(arxiv_id)
        meta_path.write_text(
            json.dumps(meta.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        _download_pdf(meta.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf", pdf_path)

    return ResolvedSource(meta=meta, pdf_path=pdf_path, cache_key=cache_key, cache_dir=cache_dir)


def _fetch_arxiv_metadata(arxiv_id: str) -> PaperMeta:
    text = _query_arxiv({"id_list": arxiv_id, "max_results": 1})
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise SourceError(f"arXiv 对 {arxiv_id!r} 返回了无法解析的响应：{exc}") from exc

    entry = root.find("atom:entry", ARXIV_ATOM_NS)
    if entry is None or entry.find("atom:id", ARXIV_ATOM_NS) is None:
        raise SourceError(f"找不到 arXiv id 为 {arxiv_id!r} 的论文。")

    title = _text(entry.find("atom:title", ARXIV_ATOM_NS)) or arxiv_id
    abstract = _text(entry.find("atom:summary", ARXIV_ATOM_NS))
    authors = [
        _text(a.find("atom:name", ARXIV_ATOM_NS))
        for a in entry.findall("atom:author", ARXIV_ATOM_NS)
    ]
    authors = [a for a in authors if a]
    published = _text(entry.find("atom:published", ARXIV_ATOM_NS))
    year = int(published[:4]) if published and published[:4].isdigit() else None

    return PaperMeta(
        title=title,
        arxiv_id=arxiv_id,
        authors=authors,
        year=year,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        abstract=abstract,
    )


def _download_pdf(url: str, dest: Path) -> None:
    import httpx

    try:
        with httpx.stream(
            "GET", url, headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True
        ) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        dest.unlink(missing_ok=True)
        raise SourceError(f"从 {url} 下载 PDF 失败：{exc}") from exc

    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise SourceError(f"从 {url} 下载到的是空 PDF。")


def _text(node: Optional[ET.Element]) -> str:
    if node is None or node.text is None:
        return ""
    return re.sub(r"\s+", " ", node.text).strip()
