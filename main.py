from pinecone import Vector
from source.datasources.pinecone import search_vectors, delete_namespace, upload_vectors
from source.datasources.utils import extract_text_from_pdf
from source.embeddings.gemini_embedding import (
    get_embedding,
    get_embeddings,
    semantic_chunking,
)
from source.helper_function import set_logger
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = set_logger("main")


def main():
    # content = extract_text_from_pdf(
    #     "./data/documents/the-strange-case-of-doctor-jekyll-and-mr-hyde-robert-louis-stevenson.pdf"
    # )
    # chunks = semantic_chunking(content)
    # list_of_chunks = []
    # for chunk in chunks:
    #     list_of_chunks.append(chunk.page_content)
    # embeddings = get_embeddings(list_of_chunks)
    # vectors = []
    # for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    #     vectors.append(
    #         Vector(
    #             id=str(i),
    #             values=embedding.values,
    #             metadata={"text": chunk.page_content},
    #         )
    #     )
    # upload_vectors(vectors=vectors, namespace="temp")
    vector = get_embedding(
        "The central theme is the eternal dilemma between good and \nevil, which is so relevant and at the same time makes this a \ntimeless book."
    )
    res = search_vectors(vector=vector, top_k=3, namespace="temp")
    print(res)


if __name__ == "__main__":
    main()
