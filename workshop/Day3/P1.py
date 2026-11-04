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

REWRITE_PROMPT = """You are a search-query rewriter for a document retrieval system.
Rewrite the user question as a single standalone search query that could be
matched against a campus handbook. Resolve pronouns and conversation references.
Do NOT answer the question. Do NOT add facts that are not implied by the question.
If the question is already standalone, return it unchanged.

User question: {question}

Standalone search query:"""


def rewrite_query(question: str, model: str = GEMINI_CHAT_MODEL) -> str:
    """
    Return a retrieval-friendly standalone version of the question.
    Cost: exactly 1 Gemini call (with backoff). Falls back to the original
    question if the model returns an empty string — an empty rewrite would
    silently destroy retrieval quality.
    """
    prompt = REWRITE_PROMPT.format(question=question)
    rewritten = gemini_text(prompt, model=model)
    return rewritten if rewritten else question

print(rewrite_query("what about its fine?"))                 # → standalone version
print(rewrite_query("How many books can a UG student borrow?"))  # → returned unchanged
print(rewrite_query("and the lab rules too?"))               # → resolved into a full query