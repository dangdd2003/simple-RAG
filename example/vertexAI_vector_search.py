import os
import uuid
import requests

import vertexai
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_vertexai import VertexAIEmbeddings
from vertexai.language_models import TextEmbeddingModel

from source.datasources.utils import extract_text_from_pdf


def main():
    # Install a sample story (Les Misérables in this example)
    try:
        req = requests.get(
            "https://cleveracademy.vn/wp-content/uploads/2016/10/Les-Miserables.pdf",
            stream=True,
        )
        req.raise_for_status()
        with open("./data/documents/Les-Miserables.pdf", "wb") as file:
            for chunk in req.iter_content(chunk_size=8192):
                file.write(chunk)
    except Exception as e:
        print(f"Failed to download the sample document: {e}")

    content = extract_text_from_pdf("./data/documents/Les-Miserables.pdf")

    load_dotenv()

    vertexai.init(
        project=os.getenv("VECTOR_SEARCH_PROJECT_ID"),
        location=os.getenv("VECTOR_SEARCH_LOCATION"),
        credentials=service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        ),
    )

    # Perform semantic chunking
    model_name = "text-embedding-005"
    text_splitter = SemanticChunker(VertexAIEmbeddings(model_name=model_name))
    chunks = text_splitter.create_documents([content])
    chunk_texts = [chunk.page_content for chunk in chunks]

    # Get embeddings for chunks
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings = model.get_embeddings(chunk_texts)
    embeddings = [embedding.values for embedding in embeddings]

    index = aiplatform.MatchingEngineIndex(
        index_name=os.getenv("VECTOR_SEARCH_INDEX_NAME"),
    )

    # Create a metadata mapping
    metadata = {str(uuid.uuid4()): text for text in chunk_texts}

    # Upsert datapoints
    datapoints = [
        {"datapoint_id": dp_id, "feature_vector": embedding}
        for (dp_id, _), embedding in zip(metadata.items(), embeddings)
    ]
    # index.upsert_datapoints(datapoints=datapoints)

    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=os.getenv("VECTOR_SEARCH_INDEX_ENDPOINT_NAME"),
    )

    # Find neighbors
    res = endpoint.find_neighbors(
        deployed_index_id=os.getenv("VECTOR_SEARCH_DEPLOYED_INDEX_ID"),
        queries=[embeddings[0]],
        num_neighbors=10,
        return_full_datapoint=True,
    )
    print(res)

    # You can now map the result IDs back to your metadata
    for neighbor in res[0]:

        print(
            f"Neighbor ID: {neighbor.id}, Text: {metadata.get(neighbor.id, 'Metadata not found')}"
        )

    # Remove datapoints
    # index.remove_datapoints(datapoint_ids=list(metadata.keys()))
