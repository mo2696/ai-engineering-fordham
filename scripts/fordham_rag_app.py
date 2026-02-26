# fordham_rag_app.py
import sys
import os
from pathlib import Path

# Make sure imports work from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import json
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fordham University Assistant",
    page_icon="🎓",
    layout="centered"
)

# ── Load resources (cached so they only load once) ────────────────────────────
@st.cache_resource
def load_resources():
    client     = OpenAI()
    tokenizer  = tiktoken.get_encoding("cl100k_base")
    embeddings = np.load("temp/embeddings.npy")
    with open("temp/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return client, tokenizer, embeddings, chunks

client, tokenizer, embeddings, all_chunks = load_resources()

EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS  = 8000

# ── Helpers ───────────────────────────────────────────────────────────────────
def truncate_to_tokens(text: str) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= MAX_TOKENS:
        return text
    return tokenizer.decode(tokens[:MAX_TOKENS])

def get_query_embedding(question: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[truncate_to_tokens(question)]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    query_vec  = get_query_embedding(question)
    norms      = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    similarities = (embeddings @ query_vec) / (norms * query_norm + 1e-10)
    top_indices  = np.argsort(similarities)[::-1][:top_k]
    return [
        {
            "text":       all_chunks[i]["text"],
            "source_url": all_chunks[i]["source_url"],
            "score":      float(similarities[i])
        }
        for i in top_indices
    ]

def generate(question: str, chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i} - {chunk['source_url']}]:\n{chunk['text']}\n\n"

    prompt = (
        "You are a helpful assistant for Fordham University.\n"
        "Answer the user's question using ONLY the provided context below.\n"
        "If the context does not contain enough information, say so — do not make anything up.\n"
        "At the end of your answer, list the full URLs of the sources you used under a 'Sources:' section.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful Fordham University assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600
    )
    return response.choices[0].message.content

def rag(question: str) -> tuple[str, list[dict]]:
    chunks = retrieve(question, top_k=5)
    answer = generate(question, chunks)
    return answer, chunks

# ── UI ────────────────────────────────────────────────────────────────────────
# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎓 Fordham University Assistant")
st.markdown("Ask anything about Fordham — programs, admissions, tuition, campus life, and more.")

st.divider()

question = st.text_input(
    label="Your question",
    placeholder="e.g. What programs does the Gabelli School of Business offer?"
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching Fordham's website..."):
            answer, chunks = rag(question)

        st.markdown("### Answer")
        st.markdown(answer)

        st.divider()
        st.markdown("### 📄 Sources Used")
        seen_urls = set()
        for chunk in chunks:
            url = chunk["source_url"]
            if url not in seen_urls:
                seen_urls.add(url)
                score = chunk["score"]
                with st.expander(f"🔗 {url}  (relevance: {score:.3f})"):
                    st.markdown(chunk["text"][:600] + "...")