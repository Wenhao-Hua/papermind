"""Find a paper's *real* code repository and inspect it.

This makes reproduction grounded: instead of an LLM-guessed ``git clone`` URL and
imagined steps, we (1) find the repo from a link in the paper text or via
PapersWithCode, then (2) read its actual dependency file and README via the
GitHub API to derive real install/run commands.

Everything here is best-effort. Any network or parsing failure degrades
gracefully — we return whatever we have (or ``None``); nothing raises.
"""

from __future__ import annotations

import base64
import os
import re
from typing import List, Optional, Tuple

from papermind.output.schema import RepoRef

# Match github.com/owner/repo even when PDF extraction mangled the printed URL:
# the scheme is often lost/garbled ("p://www.github.com/...") so we don't require it,
# and line-wrapping can put whitespace around the path separators.
_GH_RE = re.compile(r"github\.com\s*/\s*([A-Za-z0-9_.-]+)\s*/\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
# PDF text often renders "fl"/"fi" etc. as single ligature glyphs (tensorﬂow), which
# break the character class above — fold them back before scanning.
_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"})
# github.com paths that are not user/repo
_SKIP_OWNERS = {
    "blog", "about", "features", "topics", "sponsors", "marketplace", "apps",
    "settings", "orgs", "collections", "explore", "pricing", "readme", "site",
}
_SKIP_REPOS = {"blob", "tree", "issues", "pulls", "wiki", "releases", ""}

# Dependency files, in the order we prefer them, with the real install command.
_DEP_PRIORITY: List[Tuple[str, List[str]]] = [
    ("requirements.txt", ["pip install -r requirements.txt"]),
    ("environment.yml", ["conda env create -f environment.yml"]),
    ("environment.yaml", ["conda env create -f environment.yaml"]),
    ("pyproject.toml", ["pip install -e ."]),
    ("setup.py", ["pip install -e ."]),
    ("Pipfile", ["pipenv install"]),
]

# A line in a README that is plausibly a "run the experiment" command.
_RUN_RE = re.compile(
    r"^(python3?\s+\S+\.py(\s|$)|torchrun\b|accelerate\s+launch\b|deepspeed\b"
    r"|bash\s+\S+\.sh|sh\s+\S+\.sh|make\s+\w+)",
    re.IGNORECASE,
)

_TIMEOUT = 8.0


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def find_and_inspect_repo(parsed) -> Optional[RepoRef]:
    """Best-effort: locate and inspect the paper's code repo. Never raises."""
    try:
        text = parsed.full_text
        meta = parsed.meta
        url, source, official = find_repo_url(text, getattr(meta, "abstract", None), getattr(meta, "arxiv_id", None))
    except Exception:
        return None
    if not url:
        return None
    try:
        ref = inspect_repo(url, source)
    except Exception:
        ref = RepoRef(url=url, source=source)
    ref.is_official = official or ref.is_official
    return ref


# --------------------------------------------------------------------------- #
# Finding the URL
# --------------------------------------------------------------------------- #
def find_repo_url(text: str, abstract: Optional[str], arxiv_id: Optional[str]) -> Tuple[Optional[str], str, bool]:
    """Return (url, source, is_official). Prefers a link printed in the paper
    (author-provided, authoritative) over PapersWithCode."""
    url = _scan_github(text) or _scan_github(abstract or "")
    if url:
        return url, "论文原文链接", True
    url, official = _pwc_repo(arxiv_id)
    if url:
        return url, "PapersWithCode", official
    return None, "", False


def _owner_repo(m) -> Optional[Tuple[str, str]]:
    """Clean a ``_GH_RE`` match into (owner, repo), or None if it's not a real repo path."""
    owner = m.group(1).strip("-.")
    repo = m.group(2).strip().rstrip(".")
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo or owner.lower() in _SKIP_OWNERS or repo.lower() in _SKIP_REPOS:
        return None
    return owner, repo


def _scan_github(text: str) -> Optional[str]:
    text = (text or "").translate(_LIGATURES)  # fold ligatures (tensorﬂow -> tensorflow)
    for m in _GH_RE.finditer(text):
        pair = _owner_repo(m)
        if pair:
            return f"https://github.com/{pair[0]}/{pair[1]}"
    return None


def _pwc_repo(arxiv_id: Optional[str]) -> Tuple[Optional[str], bool]:
    if not arxiv_id:
        return None, False
    base = "https://paperswithcode.com/api/v1"
    r = _http_get(f"{base}/papers/?arxiv_id={arxiv_id}")
    if r is None:
        return None, False
    try:
        results = r.json().get("results") or []
        pid = results[0].get("id") if results else None
    except Exception:
        return None, False
    if not pid:
        return None, False
    r = _http_get(f"{base}/papers/{pid}/repositories/")
    if r is None:
        return None, False
    try:
        repos = [x for x in (r.json().get("results") or []) if x.get("url")]
    except Exception:
        return None, False
    if not repos:
        return None, False
    repos.sort(key=lambda x: (not x.get("is_official"), -(x.get("stars") or 0)))
    best = repos[0]
    return best.get("url"), bool(best.get("is_official"))


# --------------------------------------------------------------------------- #
# Inspecting the repo via the GitHub API
# --------------------------------------------------------------------------- #
def inspect_repo(url: str, source: str) -> RepoRef:
    ref = RepoRef(url=url, source=source)
    m = _GH_RE.search(url)  # .match anchors at pos 0 -> never matches an "https://..." URL
    pair = _owner_repo(m) if m else None
    if pair is None:
        return ref
    owner, repo = pair
    api = f"https://api.github.com/repos/{owner}/{repo}"

    r = _http_get(api, headers=_gh_headers())
    if r is not None:
        try:
            d = r.json()
            ref.stars = d.get("stargazers_count")
            ref.description = d.get("description")
            ref.default_branch = d.get("default_branch") or "main"
            ref.language = d.get("language")
            lic = d.get("license")
            ref.license = lic.get("spdx_id") if isinstance(lic, dict) else None
        except Exception:
            pass

    names = _top_level_files(api)
    lower = {n.lower(): n for n in names}
    for fname, cmds in _DEP_PRIORITY:
        if fname.lower() in lower:
            ref.dep_file = lower[fname.lower()]
            ref.install_commands = list(cmds)
            break

    readme = _readme(owner, repo)
    if readme:
        ref.run_commands = _extract_run_commands(readme)
    return ref


def _top_level_files(api: str) -> List[str]:
    r = _http_get(f"{api}/contents", headers=_gh_headers())
    if r is None:
        return []
    try:
        return [it.get("name", "") for it in r.json() if isinstance(it, dict)]
    except Exception:
        return []


def _readme(owner: str, repo: str) -> Optional[str]:
    r = _http_get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=_gh_headers())
    if r is None:
        return None
    try:
        d = r.json()
        if d.get("encoding") == "base64" and d.get("content"):
            return base64.b64decode(d["content"]).decode("utf-8", "replace")
    except Exception:
        return None
    return None


def _extract_run_commands(readme: str) -> List[str]:
    cmds: List[str] = []
    seen = set()
    for line in readme.splitlines():
        cand = line.strip()
        if cand.startswith(("$ ", "> ")):
            cand = cand[2:].strip()
        if _RUN_RE.match(cand) and cand not in seen and 4 <= len(cand) <= 200:
            seen.add(cand)
            cmds.append(cand)
        if len(cmds) >= 6:
            break
    return cmds


# --------------------------------------------------------------------------- #
def _gh_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "PaperMind"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get(url: str, headers: Optional[dict] = None):
    try:
        import httpx

        r = httpx.get(url, headers=headers, timeout=_TIMEOUT, follow_redirects=True)
        return r if r.status_code == 200 else None
    except Exception:
        return None
