import os
import time
import random
from typing import List, Dict, Any

import numpy as np          # MMR matrix math
import ollama               # local embedding path
import chromadb             
from dotenv import load_dotenv
from google import genai    # Gemini client for query rewriting + generation

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
# Free-tier chat model used for query rewriting and the one-shot RAG demo.
GEMINI_CHAT_MODEL = "gemini-3.6-flash"

# Local fallback brain for the quality/latency comparison (8 GB RAM friendly).
OLLAMA_CHAT_MODEL = "gemma3:4b"

# Embedding model 
# Mixing embedders between ingest and query silently destroys retrieval quality.
OLLAMA_EMBED_MODEL = "nomic-embed-text"     # 768-dim, local, offline

# Collection created by ingest.py on Day 2 — the capstone corpus.
COLLECTION_NAME = "capstone_corpus"

# When MMR is ON we fetch this many candidates per requested result so the
# re-ranker has a pool to trade relevance for diversity from.
FETCH_MULTIPLIER = 3

# One Gemini client for the whole file (connection reuse).
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Embedding helpers 
# module-level demo output. The one invariant that matters is the model name.

def embed_query_ollama(query: str) -> List[float]:
    """Embed one query string with nomic-embed-text → 768-dim float vector."""
    resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=query)
    return resp["embeddings"][0]          # one text in → one vector in a list


def embed_ollama(texts: List[str]) -> List[List[float]]:
    """Embed a small batch of texts (used later to re-embed MMR candidates)."""
    vectors = []
    for text in texts:
        resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=text)
        vectors.append(resp["embeddings"][0])
    return vectors


# ── Gemini call with exponential backoff ───────────

def gemini_text(prompt: str, model: str = GEMINI_CHAT_MODEL, max_retries: int = 5) -> str:
    """
    Single-turn Gemini text call with exponential backoff on 429 rate limits.
    Waits 2^attempt seconds + random jitter (jitter prevents the thundering-herd
    problem when many clients retry at the same instant).
    Non-rate-limit errors (bad key, bad request) raise immediately.
    """
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return (resp.text or "").strip()
        except Exception as e:
            # google-genai surfaces 429 as "429" or "RESOURCE_EXHAUSTED" in the message.
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper()
            if not is_rate_limit or attempt == max_retries - 1:
                raise                      # non-retryable, or retries exhausted
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"  [backoff] 429 on attempt {attempt+1}. Waiting {wait:.1f}s...")
            time.sleep(wait)


print("[retriever] Ready — embedding model:", OLLAMA_EMBED_MODEL, "| chat model:", GEMINI_CHAT_MODEL)

# **************************************************************************************************************
# Why rewrite? Retrieval matches the query string as-is. Conversational or
# anaphoric questions ("what about its fine?") embed poorly because the vector
# store has no memory of what "its" refers to. ONE extra Gemini call before
# retrieval converts them into standalone search queries.
# REWRITE_PROMPT = """You are a search-query rewriter for a document retrieval system.
# Rewrite the user question as a single standalone search query that could be
# matched against a campus handbook. Resolve pronouns and conversation references.
# Do NOT answer the question. Do NOT add facts that are not implied by the question.
# If the question is already standalone, return it unchanged.
#
# User question: {question}
#
# Standalone search query:"""
#
#
# def rewrite_query(question: str, model: str = GEMINI_CHAT_MODEL) -> str:
#     """
#     Return a retrieval-friendly standalone version of the question.
#     Cost: exactly 1 Gemini call (with backoff). Falls back to the original
#     question if the model returns an empty string — an empty rewrite would
#     silently destroy retrieval quality.
#     """
#     prompt = REWRITE_PROMPT.format(question=question)
#     rewritten = gemini_text(prompt, model=model)
#     return rewritten if rewritten else question
#
# print(rewrite_query("what about its fine?"))                 # → standalone version
# print(rewrite_query("How many books can a UG student borrow?"))  # → returned unchanged
# print(rewrite_query("and the lab rules too?"))               # → resolved into a full query
# *************************************************************************************************
def mmr_rerank(
    query_emb: List[float],
    candidate_embeddings: List[List[float]],
    k: int,
    lam: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance — returns the INDICES of the k selected candidates.

        score(d) = lam * sim(query, d) - (1 - lam) * max_{s already selected} sim(d, s)

    lam = 1.0 → pure relevance (identical to plain top-k)
    lam = 0.0 → pure diversity (spreads selections across the candidate pool)
    Default 0.7 → keep relevance dominant, suppress near-duplicates.

    Input candidates are expected roughly relevance-sorted (Chroma returns them
    that way), but the algorithm does not depend on the order.
    """
    # Degenerate cases: nothing to choose from, or fewer candidates than k.
    if len(candidate_embeddings) <= k:
        return list(range(len(candidate_embeddings)))

    # (N, D) matrix of L2-normalised candidate vectors.
    # After normalisation, a row-dot-row product IS cosine similarity.
    mat = np.asarray(candidate_embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)      # guard against zero vectors

    # Normalise the query the same way so mat @ q = cosine sims to the query.
    q = np.asarray(query_emb, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)

    sim_q = mat @ q                 # shape (N,) — similarity of every candidate to the query

    selected: List[int] = []        # indices chosen so far
    remaining = list(range(len(mat)))  # indices still in the pool

    while remaining and len(selected) < k:
        if not selected:
            # First pick: nothing to be redundant with → plain best match.
            best = remaining[int(np.argmax(sim_q[remaining]))]
        else:
            # Similarity of each remaining candidate to each already-selected one.
            redundancy = mat[remaining] @ mat[selected].T   # shape (len(remaining), len(selected))
            max_red = redundancy.max(axis=1)                # worst-case duplication per candidate
            mmr_scores = lam * sim_q[remaining] - (1.0 - lam) * max_red
            best = remaining[int(np.argmax(mmr_scores))]

        selected.append(best)
        remaining.remove(best)      # list.remove is O(N) — fine at k*3 ≤ ~30 candidates

    return selected


# **Execution steps:** *****IMPORTANT****
# Run these three commands separately to demonstrate the effect of lambda:

# Experiment 1 – Pure relevance
# python -c "from P1 import mmr_rerank; print(mmr_rerank([1,0], [[1,0],[0.99,0.1],[0,1],[0.1,0.99]], k=2, lam=1.0))"

# Expected: [0, 1]

# Experiment 2 – Balanced relevance and diversity
# python -c "from P1 import mmr_rerank; print(mmr_rerank([1,0], [[1,0],[0.99,0.1],[0,1],[0.1,0.99]], k=2, lam=0.7))"

# Expected: [0, 1]

# Experiment 3 – More diversity
# python -c "from P1 import mmr_rerank; print(mmr_rerank([1,0], [[1,0],[0.99,0.1],[0,1],[0.1,0.99]], k=2, lam=0.3))"

# Expected: [0, 2]
# ***************************************************************************************************************************
_collection = None


def _get_collection():
    """
    Return the capstone_corpus Chroma collection, failing fast with an
    actionable message if Day 2's ingest was never run.
    """
    global _collection
    if _collection is None:
        chroma = chromadb.PersistentClient(path="./chroma_db")
        try:
            _collection = chroma.get_collection(COLLECTION_NAME)
        except Exception:
            raise SystemExit(
                f"[Error] Collection '{COLLECTION_NAME}' not found in ./chroma_db.\n"
                f"        Run Day 2's P4.py first:\n"
                f"        python P4.py --backend chroma --embedder ollama --input ./sample_corpus"
            )
    return _collection


def retrieve(
    question: str,
    k: int = 5,
    mmr: bool = False,
    rewrite: bool = False,
    lam: float = 0.7,
) -> Dict[str, Any]:
    """
    One retrieval call with three switches. This is Session 1's Chroma-direct
    lab version (raw vectors needed for MMR); Session 2's rag_engine.py routes
    through the Day 2 VectorStore abstraction instead so --backend works.

    Returns:
        {
          "query_used": str,            # original or rewritten question
          "results": [ {id, document, metadata, score, embedding}, ... ],
          "timings": {rewrite_s, embed_s, retrieve_s}
        }
    score = cosine similarity (1 - cosine distance); higher = more relevant.
    """
    col = _get_collection()
    timings: Dict[str, float] = {}

    # ── Switch: rewrite (0 or 1 Gemini call) ────────────────────────────────
    t0 = time.perf_counter()
    search_q = rewrite_query(question) if rewrite else question
    timings["rewrite_s"] = round(time.perf_counter() - t0, 3) if rewrite else 0.0

    # ── Always: embed the (possibly rewritten) query ────────────────────────
    # Same model as ingest — nomic-embed-text — no exceptions.
    t0 = time.perf_counter()
    q_emb = embed_query_ollama(search_q)
    timings["embed_s"] = round(time.perf_counter() - t0, 3)

    # ── Fetch: over-fetch when MMR needs a pool to choose from ──────────────
    t0 = time.perf_counter()
    fetch_k = k * FETCH_MULTIPLIER if mmr else k
    resp = col.query(
        query_embeddings=[q_emb],
        n_results=fetch_k,
        # ids come back automatically; we also want text, metadata, distance AND
        # the stored vectors (MMR needs the candidate embeddings).
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    # Chroma returns nested lists (one list per query vector) — we sent one query.
    raw: List[Dict[str, Any]] = []
    for id_, doc, meta, dist, emb in zip(
        resp["ids"][0],
        resp["documents"][0],
        resp["metadatas"][0],
        resp["distances"][0],
        resp["embeddings"][0],
    ):
        raw.append({
            "id":        id_,
            "document":  doc,
            "metadata":  meta,
            "score":     1.0 - dist,      # cosine distance → cosine similarity
            "embedding": list(emb),       # kept only so MMR can operate on it
        })

    # ── Switch: mmr (re-rank the pool, keep top k) ──────────────────────────
    if mmr and len(raw) > k:
        order = mmr_rerank(q_emb, [r["embedding"] for r in raw], k, lam)
        results = [raw[i] for i in order]
    else:
        results = raw[:k]
    timings["retrieve_s"] = round(time.perf_counter() - t0, 3)

    return {"query_used": search_q, "results": results, "timings": timings}

# Execution of the code: )
# python -c "from P1 import retrieve; out = retrieve('What is the fine for late book return?', k=3, mmr=True); print('Query used:', out['query_used']); print('Results:'); [print('Score:', round(r['score'], 3), '|', r['document'][:90]) for r in out['results']]"