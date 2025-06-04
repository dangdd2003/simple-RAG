from source.datasources.pinecone import search_vectors, delete_namespace
from source.embeddings.gemini_embedding import get_embedding
from source.helper_function import set_logger
from source.models.google_llm import query_rag

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
    # prompt = "Who are Dr. Jekyll and Mr. Hyde"
    # vector = get_embedding(
    #     prompt
    # )
    # res = search_vectors(vector=vector, top_k=30, namespace="temp")
    # score_text_pairs = [(match.score, match.metadata["text"]) for match in res.matches]
    # score_text_pairs.sort(key=lambda x: x[0], reverse=True)
    # top_10_chunks = score_text_pairs[:10]
    # top_10_texts = [text for _, text in top_10_chunks]
    # context = "\n\n".join(top_10_texts)
    # output = query_rag(prompt, context)
    # print(output)
    pass


if __name__ == "__main__":
    main()
