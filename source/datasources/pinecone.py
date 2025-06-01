import os
import logging
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, Vector

from source.helper_function import set_logger

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(name=os.getenv("PINECONE_INDEX_NAME", "rag"))

logger = set_logger("pinecone")


def upload_vectors(vectors: List[Vector], namespace: str = "default"):
    try:
        index.upsert(
            vectors=vectors,
            namespace=namespace,
        )
        logger.info(msg=f"Uploaded {len(vectors)} vectors to namespace '{namespace}'.")
    except Exception as e:
        logger.error(msg=f"Failed to upload vectors to namespace '{namespace}': {e}")


def delete_vector(id: str, namespace: str = "default"):
    try:
        index.delete(
            ids=[id],
            namespace=namespace,
        )
        logger.warning(msg=f"Deleted vector with id '{id}' from namespace '{namespace}'.")
    except Exception as e:
        logger.error(msg=f"Failed to delete vector with id '{id}' from namespace '{namespace}': {e}")


def delete_namespace(namespace: str = "default"):
    try:
        index.delete(delete_all=True, namespace=namespace)
        logger.warning(msg=f"Deleted namespace '{namespace}'.")
    except Exception as e:
        logger.error(msg=f"Failed to delete namespace '{namespace}': {e}")


def search_vectors(vector: list[float] | None, top_k: int = 3, namespace: str = "default"):
    return index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_values=True,
        include_metadata=False
    )
