# endee_client.py
# Endee Vector Database client using official Python SDK
# Docs: https://docs.endee.io/python-sdk/quickstart

from endee import Endee, Precision
import os
from dotenv import load_dotenv

load_dotenv()

ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")

# Initialize client — connects to localhost:8080 by default
client = Endee(ENDEE_TOKEN) if ENDEE_TOKEN else Endee()


def create_index(index_name: str, dimension: int = 384):
    """Create a new vector index. Skips if already exists."""
    existing = list_indexes()
    if index_name in existing:
        print(f"[Endee] Index '{index_name}' already exists. Skipping.")
        return True

    try:
        client.create_index(
            name=index_name,
            dimension=dimension,
            space_type="cosine",
            precision="float32"    # Correct enum from official docs
        )
        print(f"[Endee] Index '{index_name}' created successfully.")
        return True
    except Exception as e:
        print(f"[Endee] Failed to create index: {e}")
        return False


def insert_vectors(index_name: str, vectors: list, metadata: list):
    """
    Insert embeddings into Endee using SDK upsert.
    SDK handles batching and timeouts internally.
    """
    try:
        index = client.get_index(name=index_name)

        payload = [
            {
                "id": f"vec_{i}",
                "vector": vectors[i],
                "meta": metadata[i]
            }
            for i in range(len(vectors))
        ]

        index.upsert(payload)
        print(f"[Endee] Inserted {len(vectors)} vectors into '{index_name}'.")
        return True

    except Exception as e:
        print(f"[Endee] Insert failed: {e}")
        return False


def query_index(index_name: str, query_vector: list, top_k: int = 5):
    """
    Query Endee for top-k similar vectors.
    Returns list of metadata dicts from nearest neighbors.
    """
    try:
        index = client.get_index(name=index_name)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_vectors=False
        )
        print(f"[Endee] Search returned {len(results)} results.")
        return [r.get("meta", {}) for r in results]

    except Exception as e:
        print(f"[Endee] Query failed: {e}")
        return []


def list_indexes():
    """
    List all existing index names in Endee.
    Per official docs — list_indexes() returns plain list of strings.
    """
    try:
        indexes = client.list_indexes()
        print(f"[Endee] Active indexes: {indexes}")
        return indexes if isinstance(indexes, list) else []
    except Exception as e:
        print(f"[Endee] List failed: {e}")
        return []


def delete_index(index_name: str):
    """Permanently delete an index and all its vectors."""
    try:
        client.delete_index(name=index_name)
        print(f"[Endee] Index '{index_name}' deleted.")
        return True
    except Exception as e:
        print(f"[Endee] Delete failed: {e}")
        return False