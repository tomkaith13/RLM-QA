import hashlib
import json
import sys
import threading
import time
from pathlib import Path

from main import DATA_DIR, format_call, load_transcripts

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
INDEX_DIR = Path(__file__).parent / "index"
INDEX_NAME = "transcripts"
ENCODE_BATCH_SIZE = 16


def compute_corpus_hash() -> str:
    """SHA-256 of all JSON files in the data directory."""
    h = hashlib.sha256()
    for path in sorted(DATA_DIR.glob("*.json")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build_corpus(calls: list[dict]) -> tuple[list[str], list[str]]:
    """Return (doc_ids, doc_texts) parallel lists, one entry per call."""
    doc_ids = []
    doc_texts = []
    for call in calls:
        doc_ids.append(call["id"])
        doc_texts.append(format_call(call))
    return doc_ids, doc_texts


def index_is_current(corpus_hash: str) -> bool:
    """Check if the index exists and matches the given corpus hash."""
    hash_path = INDEX_DIR / "corpus_hash.txt"
    index_path = INDEX_DIR / INDEX_NAME
    if not hash_path.exists() or not index_path.exists():
        return False
    stored_hash = hash_path.read_text().strip()
    return stored_hash == corpus_hash


def build_index(force: bool = False) -> None:
    """Build the PLAID index from transcript data."""
    corpus_hash = compute_corpus_hash()
    if not force and index_is_current(corpus_hash):
        print("Index is up to date.")
        return

    from pylate import indexes, models

    _, calls = load_transcripts()
    doc_ids, doc_texts = build_corpus(calls)
    print(f"Corpus: {len(doc_ids)} documents")

    print(f"Loading model {MODEL_NAME} (first run downloads ~130MB)...")
    model = models.ColBERT(model_name_or_path=MODEL_NAME)

    print("Encoding documents...")
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=ENCODE_BATCH_SIZE,
        is_query=False,
        show_progress_bar=True,
    )
    print(f"Encoded {len(doc_embeddings)} documents, dim={doc_embeddings[0].shape[-1]}")

    print("Building PLAID index...")
    INDEX_DIR.mkdir(exist_ok=True)
    index = indexes.PLAID(
        index_folder=str(INDEX_DIR),
        index_name=INDEX_NAME,
        override=True,
    )
    index.add_documents(
        documents_ids=doc_ids,
        documents_embeddings=doc_embeddings,
    )

    # Write metadata sidecar
    metadata = []
    for call in calls:
        attrs = {a["label"]: a["value"] for a in call.get("attributes", [])}
        metadata.append({"doc_id": call["id"], "attrs": attrs})
    metadata_path = INDEX_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Write corpus hash (last — acts as sentinel for complete build)
    hash_path = INDEX_DIR / "corpus_hash.txt"
    hash_path.write_text(corpus_hash)

    print(f"Index built: {len(doc_ids)} documents → {INDEX_DIR / INDEX_NAME}")


# ── Search ──────────────────────────────────────────────────────────────────

_model = None
_retriever = None
_calls_by_id = None
_load_lock = threading.Lock()


def _load_search_state() -> None:
    """Lazy-load the ColBERT model, PLAID index, and transcript data."""
    global _model, _retriever, _calls_by_id
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return

        from pylate import indexes, models, retrieve

        index_path = INDEX_DIR / INDEX_NAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}. Run `uv run build_index.py` first."
            )

        model = models.ColBERT(model_name_or_path=MODEL_NAME)
        index = indexes.PLAID(
            index_folder=str(INDEX_DIR),
            index_name=INDEX_NAME,
            override=False,
        )
        retriever = retrieve.ColBERT(index=index)

        _, calls = load_transcripts()
        calls_by_id = {call["id"]: call for call in calls}

        # Assign globals last so partially-initialized state is never visible
        _calls_by_id = calls_by_id
        _retriever = retriever
        _model = model


def search_transcripts(query: str, top_k: int = 10) -> str:
    """Semantically search interview transcripts. Returns the top-k most
    relevant transcript calls ranked by similarity to the query.
    Use this to quickly find transcripts about a specific topic, brand,
    theme, or sentiment instead of iterating through all transcripts.
    top_k controls how many results to return (default 10)."""
    _load_search_state()

    query_embeddings = _model.encode([query], batch_size=1, is_query=True)
    results = _retriever.retrieve(queries_embeddings=query_embeddings, k=top_k)

    output = []
    for rank, hit in enumerate(results[0], 1):
        call = _calls_by_id.get(hit["id"])
        if call is None:
            continue
        text = format_call(call)
        output.append(f"[Rank {rank} | Score: {hit['score']:.2f}]{text}")

    if not output:
        return "No matching transcripts found."

    return "\n\n---\n\n".join(output)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    force = "--force" in sys.argv
    start = time.time()
    build_index(force=force)
    duration = time.time() - start
    print(f"--- Duration: {duration:.1f}s ---")


if __name__ == "__main__":
    main()
