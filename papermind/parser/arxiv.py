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
# A DOI: bare (10.1145/...) or with a doi: prefix. doi.org URLs are handled as URLs.
_DOI_RE = re.compile(r"^(?:doi:\s*)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)


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
        if not m:
            continue  # non-standard <id> (no abs/ form) -> skip; else the web renders an abs/None dead link
        arxiv_id = m.group(1)
        published = _text(entry.find("atom:published", ARXIV_ATOM_NS))
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [_text(a.find("atom:name", ARXIV_ATOM_NS)) for a in entry.findall("atom:author", ARXIV_ATOM_NS)]
        results.append(
            PaperMeta(
                title=_text(entry.find("atom:title", ARXIV_ATOM_NS)),
                arxiv_id=arxiv_id,
                authors=[a for a in authors if a],
                year=year,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
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


def _parse_doi(source: str) -> Optional[str]:
    """A bare DOI (``10.1145/...``) or its ``doi:`` form -> the DOI. (``doi.org`` URLs
    are handled as ordinary URLs — they 302 to the publisher's landing page.)"""
    m = _DOI_RE.match(source.strip())
    return m.group(1) if m else None


def _looks_like_title(source: str) -> bool:
    """Free text we can search arXiv for, vs. a stray token or a broken URL/path."""
    return len(source) >= 4 and "/" not in source and "\\" not in source


def _is_existing_path(source: str) -> bool:
    try:
        return Path(source).expanduser().exists()
    except (OSError, ValueError):  # a title with ':' etc. is not a valid Windows path
        return False


OPENALEX_API = "https://api.openalex.org/works"


def _search_openalex(query: str) -> Optional[dict]:
    """Top title match from OpenAlex (free, keyless, lenient — unlike the arXiv and
    Semantic Scholar search APIs, which throttle cloud IPs hard). Returns
    ``{title, pdf, doi}`` (any value may be None; ``doi`` is a full doi.org URL), or
    None if unreachable / rate-limited / no match."""
    import httpx

    try:
        resp = httpx.get(
            OPENALEX_API,
            params={"search": query, "per_page": 1, "mailto": "papermind@try2026.cn"},
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except (httpx.HTTPError, ValueError):
        return None
    if not results:
        return None
    work = results[0]
    pdf = (work.get("best_oa_location") or {}).get("pdf_url") or (work.get("open_access") or {}).get("oa_url")
    return {"title": work.get("title"), "pdf": pdf, "doi": work.get("doi")}


def _resolve_title(query: str, config: Config) -> ResolvedSource:
    """Find a paper by title and resolve it. OpenAlex first (free, reliable, covers
    non-arXiv, hands back the open-access PDF); the arXiv search API is the fallback.
    Prefer the free open-access PDF over the (possibly paywalled) DOI."""
    hit = _search_openalex(query)
    if hit:
        if hit.get("pdf"):
            return _resolve_url(hit["pdf"], config)
        if hit.get("doi"):  # already a full https://doi.org/... URL
            return _resolve_url(hit["doi"], config)
    results = search_arxiv(query, max_results=1)  # fallback: arXiv search API
    if results and results[0].arxiv_id:
        return _resolve_arxiv(results[0].arxiv_id, config)
    raise SourceError(f"没找到匹配「{query}」的论文。请换个更准确的标题，或粘贴论文链接 / DOI。")


def resolve(source: str, config: Optional[Config] = None) -> ResolvedSource:
    """Resolve a source into a local PDF + best-effort metadata.

    Accepts: an arXiv URL, a DOI (bare or ``doi:``/``doi.org``), any paper page or
    PDF URL, a paper *title* (searched on arXiv), or a local ``.pdf`` path. Bare
    arXiv ids are not accepted — give a URL.
    """
    config = config or load_config()
    source = (source or "").strip()
    if not source:
        raise SourceError("请提供论文链接（arXiv / 论文页面 / PDF / DOI）、论文标题，或本地 PDF 路径。")

    arxiv_id = parse_arxiv_id(source)
    if arxiv_id:
        return _resolve_arxiv(arxiv_id, config)
    if _ID_RE.fullmatch(source):  # a bare arXiv id -> guide to a URL, don't title-search it
        raise SourceError(f"请用完整链接而不是裸 id，例如 https://arxiv.org/abs/{source.split(':')[-1]}")
    doi = _parse_doi(source)
    if doi:
        return _resolve_url(f"https://doi.org/{doi}", config)
    if re.match(r"https?://", source, re.IGNORECASE):
        return _resolve_url(source, config)
    if _is_existing_path(source):
        return _resolve_local(source, config)
    if _looks_like_title(source):  # not an id/URL/DOI/path -> find the paper by title
        return _resolve_title(source, config)
    raise SourceError(
        f"无法识别来源 {source!r}。请粘贴论文链接（arXiv / 论文页面 / PDF / DOI）、论文标题，或本地 PDF 路径。"
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
        _download_pdf(_resolve_pdf_url(url), pdf_path)
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
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    meta: Optional[PaperMeta] = None
    if meta_path.exists():
        meta = PaperMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
    else:
        try:
            meta = _fetch_arxiv_metadata(arxiv_id)
            meta_path.write_text(
                json.dumps(meta.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except SourceError:
            # arXiv's metadata API is unreachable or rate-limiting this IP (cloud hosts
            # routinely get 429'd even with retries). The PDF lives on a different
            # endpoint — fall back to downloading it and inferring the title, so analysis
            # proceeds instead of hard-failing. metadata.json is intentionally NOT written,
            # so a later run still retries the API once it recovers.
            meta = None

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        _download_pdf((meta.pdf_url if meta else None) or pdf_url, pdf_path)

    if meta is None:
        meta = PaperMeta(
            title=_title_from_pdf(pdf_path, f"https://arxiv.org/abs/{arxiv_id}"),
            arxiv_id=arxiv_id,
            pdf_url=pdf_url,
        )

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


def _resolve_pdf_url(url: str) -> str:
    """Resolve a paper URL to a downloadable PDF URL.

    A direct PDF link is returned unchanged. An academic *landing page* (HTML) is
    parsed for its ``citation_pdf_url`` meta tag — the Highwire / Google-Scholar
    standard embedded by arXiv, bioRxiv, OpenReview, ACL, PMLR, NeurIPS, IEEE,
    Springer and most publishers — so a user can paste the page they're reading,
    not just a raw PDF link. Raises :class:`SourceError` for a page with no
    discoverable PDF. The fetch (and the resolved PDF) go through the SSRF guard.
    """
    import httpx

    from papermind.net import BlockedURLError, safe_stream

    try:
        with safe_stream("GET", url, headers={"User-Agent": USER_AGENT}, timeout=30.0) as resp:
            resp.raise_for_status()
            html = b""
            for chunk in resp.iter_bytes(chunk_size=65536):
                if not html and chunk.startswith(b"%PDF"):
                    return url  # already a direct PDF — don't slurp the whole file
                html += chunk
                if len(html) > 2_000_000:  # 2 MB of markup is far more than enough
                    break
    except BlockedURLError as exc:
        raise SourceError(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise SourceError(f"无法打开 {url}：{exc}") from exc

    pdf_url = _citation_pdf_url(html.decode("utf-8", "replace"), base=url)
    if pdf_url:
        return pdf_url
    raise SourceError(
        f"{url} 是网页而非 PDF，且未能从页面找到论文 PDF 链接。"
        "请粘贴 PDF 直链或 arXiv 链接，或上传 PDF 文件。"
    )


def _citation_pdf_url(html: str, base: str) -> Optional[str]:
    """The PDF link from a page's ``citation_pdf_url`` meta tag. Scans every ``<meta>``
    and parses its attributes, tolerating either attribute order and quoted *or*
    unquoted values (ACL Anthology, for one, leaves ``name`` unquoted)."""
    from urllib.parse import urljoin

    attr_re = r"""([a-zA-Z_:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s">]+))"""
    for tag in re.finditer(r"<meta\b([^>]*)>", html, re.IGNORECASE):
        attrs = {}
        for m in re.finditer(attr_re, tag.group(1)):
            attrs[m.group(1).lower()] = m.group(2) or m.group(3) or m.group(4) or ""
        if attrs.get("name", "").lower() == "citation_pdf_url" and attrs.get("content"):
            return urljoin(base, attrs["content"].strip())
    return None


def _download_pdf(url: str, dest: Path) -> None:
    import httpx

    from papermind.net import MAX_DOWNLOAD_BYTES, BlockedURLError, safe_stream

    cap_mb = MAX_DOWNLOAD_BYTES // (1024 * 1024)
    # Download to a sibling .part and atomically rename on success: an interrupted
    # download then leaves only the .part (never a truncated paper.pdf that the
    # size>0 / %PDF-prefix reuse checks would accept and feed to the parser).
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)  # clear any leftover from a previously interrupted run
    try:
        with safe_stream("GET", url, headers={"User-Agent": USER_AGENT}, timeout=60.0) as resp:
            resp.raise_for_status()
            clen = resp.headers.get("content-length", "")
            if clen.isdigit() and int(clen) > MAX_DOWNLOAD_BYTES:
                raise SourceError(f"{url} 的 PDF 过大（>{cap_mb}MB），拒绝下载。")
            total, first = 0, True
            with open(part, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    if first and chunk and not chunk.startswith(b"%PDF"):
                        raise SourceError(f"{url} 看起来不是 PDF（可能是网页）。请提供 PDF 直链或 arXiv 链接。")
                    first = False
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise SourceError(f"{url} 的 PDF 过大（>{cap_mb}MB），已中断下载。")
                    fh.write(chunk)
        if part.stat().st_size == 0:
            raise SourceError(f"从 {url} 下载到的是空 PDF。")
        part.replace(dest)  # atomic on the same filesystem
    except SourceError:
        part.unlink(missing_ok=True)
        raise
    except BlockedURLError as exc:
        part.unlink(missing_ok=True)
        raise SourceError(str(exc)) from exc
    except httpx.HTTPError as exc:
        part.unlink(missing_ok=True)
        raise SourceError(f"从 {url} 下载 PDF 失败：{exc}") from exc


def _text(node: Optional[ET.Element]) -> str:
    if node is None or node.text is None:
        return ""
    return re.sub(r"\s+", " ", node.text).strip()
