# Import OS utilities for environment variables.
import os

# Import JSON utilities (currently unused in this snippet).
import json

# Import time utilities (currently unused in this snippet).
import time

# Import type hints for lists.
from typing import List

# Load environment variables from a .env file.
from dotenv import load_dotenv

# Import the Google GenAI SDK client.
from google import genai

# Import the Ollama Python client.
import ollama


# Load .env variables into the environment.
load_dotenv()


# Read the Gemini API key from the environment.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create one reusable Gemini client.
client = genai.Client(api_key=GEMINI_API_KEY)


# ── Embedding model constants ────────────────────────────────────────────────
# gemini-embedding is the free-tier embedding model — check AI Studio for current name.
# nomic-embed-text produces 768-dimensional vectors; gemini-embedding produces 768 as well
# (verify in AI Studio console — the dimension must match your Pinecone index on Day 2 S2).

# Gemini embedding model name.
GEMINI_EMBED_MODEL  = "gemini-embedding-exp-03-07"   # free-tier; check AI Studio for latest name

# Local Ollama embedding model name.
OLLAMA_EMBED_MODEL  = "nomic-embed-text"              # 768-dim; fast; fully offline

# Expected vector dimension for both models.
EMBED_DIM           = 768                             # shared dimension for both models



def embed_gemini(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using the Gemini free-tier embedding model.
    Returns a list of float vectors (one per input text).
    
    The task_type parameter hints to the model how the embedding will be used:
    - RETRIEVAL_DOCUMENT  → for chunks being stored in the vector DB
    - RETRIEVAL_QUERY     → for the user's question at query time
    Using the correct task_type improves retrieval quality.
    """

    # List to collect one vector per input text.
    embeddings = []

    # Embed each text individually.
    for text in texts:
        # Request a document embedding from Gemini.
        resp = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )

        # resp.embeddings is a list of ContentEmbedding objects — take the first (one text in).
        # Store the numeric vector for this text.
        embeddings.append(resp.embeddings[0].values)

    # Return all document vectors.
    return embeddings



def embed_query_gemini(query: str) -> List[float]:
    """Embed a single query string with task_type=RETRIEVAL_QUERY for retrieval."""

    # Request a query embedding from Gemini.
    resp = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY"},
    )

    # Return the query's vector.
    return resp.embeddings[0].values



def embed_ollama(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using nomic-embed-text via Ollama.
    Fully offline — no API key, no rate limits.
    Ollama's embed endpoint accepts a single prompt at a time; we loop.
    """

    # List to collect one vector per input text.
    embeddings = []

    # Embed each text individually.
    for text in texts:
        # Request an embedding from the local Ollama model.
        resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=text)

        # resp["embeddings"] is a list with one entry (we passed one text).
        # Store the numeric vector for this text.
        embeddings.append(resp["embeddings"][0])

    # Return all document vectors.
    return embeddings



def embed_query_ollama(query: str) -> List[float]:
    """Embed a single query string via Ollama nomic-embed-text."""

    # Request a query embedding from Ollama.
    resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=query)

    # Return the query's vector.
    return resp["embeddings"][0]



# Confirm that both embedding paths are configured.
print("[Setup] Embedding helpers ready — Gemini + Ollama paths defined.")

# Print the Gemini embedding model.
print(f"        Gemini model : {GEMINI_EMBED_MODEL}")

# Print the Ollama embedding model.
print(f"        Ollama model : {OLLAMA_EMBED_MODEL}")

# Print the expected embedding dimension.
print(f"        Expected dim : {EMBED_DIM}")

# Chroma DB
# Import the Chroma vector database client.
import chromadb


# PersistentClient stores data on disk at the given path.
# On subsequent runs, it LOADS existing collections — no need to re-add documents.
# Use this for the capstone corpus (you want it to survive restarts).

# Create a disk-backed Chroma client using ./chroma_db.
chroma_persistent = chromadb.PersistentClient(path="./chroma_db")


# EphemeralClient stores data only in RAM — wiped when the process exits.
# Useful for one-off experiments or unit tests where you don't want leftovers.

# Create an in-memory Chroma client.
chroma_ephemeral = chromadb.EphemeralClient()


# Confirm the persistent client's storage location.
print(f"\n[Chroma] Persistent client ready → ./chroma_db")

# Confirm the ephemeral client is memory-only.
print(f"[Chroma] Ephemeral client ready  → in-memory only")

# List names of collections already saved in the persistent database.
print(f"[Chroma] Existing collections    : {[c.name for c in chroma_persistent.list_collections()]}")