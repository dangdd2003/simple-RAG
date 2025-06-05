import os
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from source.helper_function import set_logger

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
model_name = "models/text-embedding-004"

logger = set_logger("gemini-embedding")


def get_embeddings(contents: List) -> List[types.ContentEmbedding] | None:
    embeddings = []
    batch_size = 100

    try:
        logger.info(f"Start getting list embeddings from list documents using '{model_name}'")
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            response = client.models.embed_content(
                model=model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=768,
                ),
            )
            embeddings.extend(response.embeddings)
        logger.info(f"Finish getting list embeddings from list documents using '{model_name}'")
        return embeddings
    except Exception as e:
        logger.error(
            f"Failed to get list embeddings from list documents using '{model_name}': {e}"
        )
        return None


def get_embedding(content: str) -> list[float] | None:
    try:
        logger.info(f"Start getting embedding from a document using '{model_name}'")
        embedding_model = GoogleGenerativeAIEmbeddings(
            client=client,
            model=model_name,
            task_type="semantic_similarity",
        )
        embedding = embedding_model.embed_query(content, output_dimensionality=768)
        logger.info(f"Finish getting embedding from a document using {model_name}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to get embedding from a document using '{model_name}': {e}")
        return None


def semantic_chunking(contents: str) -> List[Document]:
    try:
        logger.info(f"Start performing semantic chunking using '{model_name}'")
        text_spliter = SemanticChunker(
            GoogleGenerativeAIEmbeddings(client=client, model=model_name, task_type="semantic_similarity"),
        )
        chunks = text_spliter.create_documents([contents])
        logger.info(
            f"Finish getting list of chunks by performing semantic chunking using {model_name}"
        )
        return chunks
    except Exception as e:
        logger.error(f"Failed to perform semantic chunking using '{model_name}': {e}")
        return []
