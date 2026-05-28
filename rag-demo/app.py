import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import hvrag

st.set_page_config(page_title="HV-RAG Demo", layout="wide")

st.title("HV-RAG Demo")
st.caption(
    "Hierarchical Voting RAG — paragraph chunks vote via RRF to select "
    "the best subject section, which is passed to the LLM for generation."
)

# ── Sidebar: upload & index ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Corpus")

    try:
        stats = hvrag.corpus_stats()
        st.metric("Indexed chunks", stats["total"], help=f"Level 0: {stats['l0']}  |  Level 1: {stats['l1']}")
    except Exception:
        st.metric("Indexed chunks", 0)

    st.divider()
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Index Corpus", type="primary", disabled=not files):
        progress = st.progress(0, text="Starting...")
        results = []
        for idx, f in enumerate(files):
            progress.progress((idx) / len(files), text=f"Indexing {f.name}…")
            suffix = Path(f.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                n = hvrag.index_file(tmp_path, doc_name=f.name)
                results.append((f.name, n, None))
            except Exception as e:
                results.append((f.name, 0, str(e)))
            finally:
                os.unlink(tmp_path)
        progress.progress(1.0, text="Done")

        for name, n, err in results:
            if err:
                st.error(f"{name}: {err}")
            else:
                st.success(f"{name} — {n} chunks")

        st.session_state.uploader_key += 1
        st.rerun()

# ── Main: query ───────────────────────────────────────────────────────────────
st.header("Ask a Question")

question = st.text_input(
    "Question",
    placeholder="e.g. What is Reciprocal Rank Fusion?",
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([1, 6])
with col_btn:
    ask_clicked = st.button("Ask", type="primary", disabled=not question)
with col_hint:
    if not question:
        st.caption("Index at least one document first, then type your question.")

if ask_clicked and question:
    with st.spinner("Retrieving…  voting…  generating…"):
        try:
            result = hvrag.ask(question)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # Answer
    st.subheader("Answer")
    st.success(result["answer"])

    # Cited source
    meta = result["source_meta"]
    st.subheader("Cited Level 0 Source")
    st.info(
        f"**Document:** {meta.get('doc_name', 'unknown')}  \n"
        f"**Section index:** {meta.get('section_index', '?')}  \n"
        f"**Chunk ID:** `{meta.get('source', '')} · section {meta.get('section_index', '?')}`"
    )
    with st.expander("View full Level 0 context sent to LLM"):
        st.text_area(
            "level0_context",
            result["context_text"],
            height=300,
            label_visibility="collapsed",
        )

    # Supporting Level 1 votes
    st.subheader("Supporting Level 1 Vote Snippets")
    supporting = result["supporting"]
    if supporting:
        for hit in supporting:
            m = hit["meta"]
            label = (
                f"Rank {hit['rank']}  |  RRF contrib {hit['rrf_contrib']:.3f}  |  "
                f"{m.get('doc_name')} · section {m.get('section_index')} · child {m.get('child_index')}"
            )
            with st.expander(label):
                snippet = hit["text"]
                st.text(snippet[:500] + ("…" if len(snippet) > 500 else ""))
    else:
        st.info(
            "No Level 1 chunks from the winning section appeared in the top-k results. "
            "The winner was selected by votes from other sections."
        )
