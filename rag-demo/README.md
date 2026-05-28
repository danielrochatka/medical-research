# HV-RAG Demo

Hierarchical Voting RAG: paragraph-level chunks vote via Reciprocal Rank Fusion to elect the best subject section, which is passed to the LLM for generation.

## Quick start

```bash
cd rag-demo

# 1. Copy and fill in your OpenAI key
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-...

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. Upload one or more PDF / TXT / MD files in the sidebar.
2. Click **Index Corpus** — chunks are stored in `data/chroma_db/`.
3. Type a question and click **Ask**.
4. See the generated answer, the cited Level 0 section, and the Level 1 vote snippets that drove the selection.

## Architecture (HV-RAG)

| Component | Detail |
|-----------|--------|
| Level 0 chunks | ~3 000 chars, 500-char overlap — full subject sections sent to LLM |
| Level 1 chunks | ~600 chars, 100-char overlap — paragraph voters, embedded only |
| Retrieval | Top-10 Level 1 chunks by cosine similarity (`text-embedding-3-small`) |
| Voting | `score(parent) += 1/rank` for each Level 1 hit — pure Python, no LLM |
| Generation | Winning Level 0 text → `gpt-4o-mini` |

## Files

```
rag-demo/
├── app.py           # Streamlit UI
├── hvrag.py         # HV-RAG core logic + LangGraph flow
├── requirements.txt
├── .env.example
└── data/
    └── chroma_db/   # Persisted vector store (auto-created)
```
