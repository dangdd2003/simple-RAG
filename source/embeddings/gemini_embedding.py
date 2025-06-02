import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_embeddings(contents: List) -> types.EmbedContentResponse:
    embeddings = client.models.embed_content(
        model="models/text-embedding-004",
        contents=contents,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=768,
        ),
    )
    return embeddings


def get_embedding(content: str) -> list[float]:
    embedding_model = GoogleGenerativeAIEmbeddings(
        client=client,
        model="models/text-embedding-004",
        task_type="semantic_similarity",
    )
    embedding = embedding_model.embed_query(content, output_dimensionality=768)
    return embedding
