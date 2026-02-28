# endee_client.py
# Handles all communication with the Endee Vector Database REST API

import requests
import os
from dotenv import load_dotenv

load_dotenv()

ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")

HEADERS = {
    "Content-Type": "application/json",
    **({"Authorization": ENDEE_TOKEN} if ENDEE_TOKEN else {})
}


def create_index(index_name: str, dimension: int = 384):
    """Create a new vector index in Endee. Skips if already exists."""
    # First check if index already exists
    existing = list_indexes()
    if index_name in existing:
        print(f"[Endee] Index '{index_name}' already exists. Skipping creation.")
        return True

    url = f"{ENDEE_URL}/api/v1/index/create"
    payload = {
        "name": index_name,
        "dimension": dimension,       # 384 = all-MiniLM-L6-v2 output size
        "metric": "cosine"            # cosine similarity for semantic search
    }
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code in [200, 201]:
        print(f"[Endee] Index '{index_name}' created successfully.")
        return True
    else:
        print(f"[Endee] Failed to create index: {response.text}")
        return False


def insert_vectors(index_name: str, vectors: list, metadata: list):
    """
    Insert embeddings into Endee.
    vectors: list of float lists [[0.1, 0.2, ...], ...]
    metadata: list of dicts [{"text": "chunk text", "source": "filename"}, ...]
    """
    url = f"{ENDEE_URL}/api/v1/index/{index_name}/insert"
    payload = {
        "vectors": [
            {
                "id": str(i),
                "values": vectors[i],
                "metadata": metadata[i]
            }
            for i in range(len(vectors))
        ]
    }
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code in [200, 201]:
        print(f"[Endee] Inserted {len(vectors)} vectors into '{index_name}'.")
        return True
    else:
        print(f"[Endee] Insert failed: {response.text}")
        return False


def query_index(index_name: str, query_vector: list, top_k: int = 5):
    """
    Query Endee for top-k similar vectors.
    Returns list of metadata dicts from nearest neighbors.
    """
    url = f"{ENDEE_URL}/api/v1/index/{index_name}/query"
    payload = {
        "vector": query_vector,
        "top_k": top_k
    }
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code == 200:
        results = response.json()
        # Extract metadata text from results
        return [r.get("metadata", {}) for r in results.get("results", [])]
    else:
        print(f"[Endee] Query failed: {response.text}")
        return []


def list_indexes():
    """List all existing indexes in Endee."""
    url = f"{ENDEE_URL}/api/v1/index/list"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("indexes", [])
    return []


def delete_index(index_name: str):
    """Delete an index from Endee (useful for cleanup)."""
    url = f"{ENDEE_URL}/api/v1/index/{index_name}/delete"
    response = requests.delete(url, headers=HEADERS)
    if response.status_code == 200:
        print(f"[Endee] Index '{index_name}' deleted.")
        return True
    return False
