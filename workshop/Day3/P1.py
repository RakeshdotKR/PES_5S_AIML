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