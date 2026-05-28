import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, TypedDict

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from langgraph.graph import END, StateGraph

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db")
COLLECTION_NAME = "hvrag"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

L0_SIZE = 3000    # ~750 tokens — large subject sections
L0_OVERLAP = 500
L1_SIZE = 600     # ~150 tokens — paragraph-level voters
L1_OVERLAP = 100
TOP_K = 10

_openai_client: Optional[OpenAI] = None
_collection = None


def _client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def get_collection():
    global _collection
    if _collection is None:
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=EMBED_MODEL,
        )
        chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = chroma.get_or_create_collection(
            COLLECTION_NAME, embedding_function=ef
        )
    return _collection


def corpus_stats() -> dict:
    col = get_collection()
    total = col.count()
    if total == 0:
        return {"total": 0, "l0": 0, "l1": 0}
    l0 = col.get(where={"level": {"$eq": 0}}, include=[])
    l1 = col.get(where={"level": {"$eq": 1}}, include=[])
    return {"total": total, "l0": len(l0["ids"]), "l1": len(l1["ids"])}


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return p.read_text(encoding="utf-8", errors="ignore")


# ── Chunking ─────────────────────────────────────────────────────────────────

def _split(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return [c for c in chunks if c.strip()]


def build_chunks(doc_name: str, text: str) -> list[dict]:
    records = []
    for si, l0_text in enumerate(_split(text, L0_SIZE, L0_OVERLAP)):
        l0_id = f"{doc_name}::l0::{si}"
        records.append(dict(
            id=l0_id,
            text=l0_text,
            metadata=dict(
                level=0,
                doc_name=doc_name,
                parent_id="",
                section_index=si,
                source=doc_name,
            ),
        ))
        for ci, l1_text in enumerate(_split(l0_text, L1_SIZE, L1_OVERLAP)):
            records.append(dict(
                id=f"{doc_name}::l1::{si}::{ci}",
                text=l1_text,
                metadata=dict(
                    level=1,
                    doc_name=doc_name,
                    parent_id=l0_id,
                    section_index=si,
                    child_index=ci,
                    source=doc_name,
                ),
            ))
    return records


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_file(path: str, doc_name: Optional[str] = None) -> int:
    doc_name = doc_name or Path(path).name
    text = extract_text(path)
    if not text.strip():
        raise ValueError(f"Could not extract text from {doc_name}")
    records = build_chunks(doc_name, text)
    col = get_collection()

    existing = col.get(where={"doc_name": {"$eq": doc_name}}, include=[])
    if existing["ids"]:
        col.delete(ids=existing["ids"])

    for i in range(0, len(records), 100):
        batch = records[i: i + 100]
        col.upsert(
            ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
        )
    return len(records)


# ── Retrieval + RRF voting ────────────────────────────────────────────────────

def retrieve_and_vote(question: str, top_k: int = TOP_K) -> dict:
    col = get_collection()

    # Count available Level 1 chunks to cap n_results safely
    l1_probe = col.get(where={"level": {"$eq": 1}}, limit=top_k, include=[])
    n_l1 = len(l1_probe["ids"])
    if n_l1 == 0:
        raise ValueError("No documents indexed. Upload and index files first.")

    n_results = min(top_k, n_l1)
    results = col.query(
        query_texts=[question],
        n_results=n_results,
        where={"level": {"$eq": 1}},
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # RRF: score += 1/rank for each Level 1 hit's parent
    rrf: dict[str, float] = defaultdict(float)
    for rank, meta in enumerate(metas, 1):
        rrf[meta["parent_id"]] += 1.0 / rank

    winner_id = max(rrf, key=rrf.__getitem__)

    parent = col.get(ids=[winner_id], include=["documents", "metadatas"])
    parent_text = parent["documents"][0]
    parent_meta = parent["metadatas"][0]

    supporting = [
        {
            "text": docs[i],
            "meta": metas[i],
            "rank": i + 1,
            "rrf_contrib": 1.0 / (i + 1),
            "voted_for_winner": metas[i]["parent_id"] == winner_id,
            "parent_rrf_total": rrf[metas[i]["parent_id"]],
        }
        for i in range(len(docs))
    ]

    return dict(
        parent_text=parent_text,
        parent_meta=parent_meta,
        supporting=supporting,
        rrf_scores=dict(rrf),
    )


# ── LangGraph flow: retrieve → generate ──────────────────────────────────────

class RAGState(TypedDict):
    question: str
    context: str
    parent_meta: dict
    supporting: list
    rrf_scores: dict
    answer: str


def _retrieve_node(state: RAGState) -> RAGState:
    r = retrieve_and_vote(state["question"])
    return {
        **state,
        "context": r["parent_text"],
        "parent_meta": r["parent_meta"],
        "supporting": r["supporting"],
        "rrf_scores": r["rrf_scores"],
    }


def _generate_node(state: RAGState) -> RAGState:
    resp = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise assistant. Answer the question using only "
                    "the provided context. If the answer is not in the context, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{state['context']}\n\nQuestion: {state['question']}",
            },
        ],
    )
    return {**state, "answer": resp.choices[0].message.content}


def _build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("generate", _generate_node)
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    g.set_entry_point("retrieve")
    return g.compile()


_graph = None


def ask(question: str) -> dict:
    global _graph
    if _graph is None:
        _graph = _build_graph()
    state = _graph.invoke({
        "question": question,
        "context": "",
        "parent_meta": {},
        "supporting": [],
        "rrf_scores": {},
        "answer": "",
    })
    return {
        "answer": state["answer"],
        "source_meta": state["parent_meta"],
        "context_text": state["context"],
        "supporting": state["supporting"],
        "rrf_scores": state["rrf_scores"],
    }
