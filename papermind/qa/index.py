"""Chunking + embedding + FAISS index, cached per paper.

Chunks follow paragraph blocks (section + page preserved for citations) grouped
into sliding windows. The built index and chunk metadata are cached under the
paper's cache dir and reused on subsequent runs, keyed by the embedding model so
a model change transparently rebuilds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from papermind.errors import PaperMindError
from papermind.llm.base import LLMClient
from papermind.parser.pdf import ParsedPaper

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

_TARGET_CHARS = 1100
_OVERLAP_BLOCKS = 1
_MIN_BLOCK_CHARS = 20
_EMBED_BATCH = 64


@dataclass
class Chunk:
    idx: int
    text: str
    section: str
    page: int


class PaperIndex:
    """A FAISS cosine-similarity index over a paper's chunks."""

    def __init__(self, chunks: List[Chunk], index, dim: int) -> None:
        self.chunks = chunks
        self._index = index
        self.dim = dim

    # -- construction ------------------------------------------------------- #
    @classmethod
    def load_or_build(cls, parsed: ParsedPaper, client: LLMClient) -> "PaperIndex":
        cache_dir = parsed.cache_dir
        embed_model = client.config.resolved_embedding_model()
        cached = cls._try_load(cache_dir, embed_model)
        if cached is not None:
            return cached
        built = cls._build(parsed, client)
        built._save(cache_dir, embed_model)
        return built

    @classmethod
    def _build(cls, parsed: ParsedPaper, client: LLMClient) -> "PaperIndex":
        import numpy as np

        faiss = _import_faiss()
        chunks = build_chunks(parsed)
        if not chunks:
            raise PaperMindError("No text chunks could be built for indexing.")

        vectors = _embed_all(client, [c.text for c in chunks])
        dim = vectors.shape[1]
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return cls(chunks, index, dim)

    # -- search ------------------------------------------------------------- #
    def search(self, query_vec: "np.ndarray", k: int = 5) -> List[Tuple[Chunk, float]]:
        import numpy as np

        faiss = _import_faiss()
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        k = min(k, len(self.chunks))
        scores, ids = self._index.search(q, k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            out.append((self.chunks[int(idx)], float(score)))
        return out

    # -- caching ------------------------------------------------------------ #
    def _save(self, cache_dir: Path, embed_model: str) -> None:
        faiss = _import_faiss()
        cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(cache_dir / "index.faiss"))
        (cache_dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        (cache_dir / "index_meta.json").write_text(
            json.dumps({"embed_model": embed_model, "dim": self.dim, "n": len(self.chunks)}),
            encoding="utf-8",
        )

    @classmethod
    def _try_load(cls, cache_dir: Path, embed_model: str) -> Optional["PaperIndex"]:
        meta_path = cache_dir / "index_meta.json"
        idx_path = cache_dir / "index.faiss"
        chunks_path = cache_dir / "chunks.json"
        if not (meta_path.exists() and idx_path.exists() and chunks_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("embed_model") != embed_model:
                return None  # embedding model changed -> rebuild
            faiss = _import_faiss()
            index = faiss.read_index(str(idx_path))
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in raw]
            return cls(chunks, index, meta.get("dim", index.d))
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            return None


# --------------------------------------------------------------------------- #
def build_chunks(
    parsed: ParsedPaper, target_chars: int = _TARGET_CHARS, overlap_blocks: int = _OVERLAP_BLOCKS
) -> List[Chunk]:
    """Group consecutive paragraph blocks into overlapping, section-aware chunks."""
    blocks = [b for b in parsed.blocks if len(b.text) >= _MIN_BLOCK_CHARS]
    chunks: List[Chunk] = []
    i = 0
    n = len(blocks)
    while i < n:
        size = 0
        j = i
        parts = []
        while j < n and size < target_chars:
            parts.append(blocks[j].text)
            size += len(blocks[j].text)
            j += 1
        chunks.append(
            Chunk(idx=len(chunks), text=" ".join(parts), section=blocks[i].section, page=blocks[i].page)
        )
        if j >= n:
            break
        i = max(j - overlap_blocks, i + 1)
    return chunks


def _embed_all(client: LLMClient, texts: List[str]) -> "np.ndarray":
    import numpy as np

    batches = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batches.append(client.embed(texts[start : start + _EMBED_BATCH]))
    return np.vstack(batches).astype("float32")


def _import_faiss():
    try:
        import faiss

        return faiss
    except ImportError as exc:  # pragma: no cover
        raise PaperMindError("faiss-cpu is required for Q&A. Install with: pip install faiss-cpu") from exc
