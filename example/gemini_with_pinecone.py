import requests
from pinecone import Vector

from source.datasources.pinecone import (
    search_vectors,
    upload_batches_vectors,
    delete_namespace,
)
from source.datasources.utils import extract_text_from_pdf
from source.embeddings.gemini_embedding import (
    get_embedding,
    semantic_chunking,
    get_embeddings,
)
from source.models.google_llm import query_rag, query_contexts


def main():
    # Install a sample story (Les Misérables in this example)
    # try:
    #     req = requests.get("https://cleveracademy.vn/wp-content/uploads/2016/10/Les-Miserables.pdf", stream=True)
    #     req.raise_for_status()
    #     with open("./data/documents/Les-Miserables.pdf", "wb") as file:
    #         for chunk in req.iter_content(chunk_size=8192):
    #             file.write(chunk)
    # except Exception as e:
    #     print(f"Failed to download the sample document: {e}")

    # content = extract_text_from_pdf(
    #     "./data/documents/Les-Miserables.pdf"
    # )

    namespace = "les-miserables"
    top_k = 50  # Retrieve top k most similar vectors
    top_n = 20  # Reranking top n the highest cosine similarity score

    # # Perform semantic chunking
    # chunks = semantic_chunking(content)

    # # Perform embedding chunks
    # list_of_chunks = []
    # for chunk in chunks:
    #     list_of_chunks.append(chunk.page_content)
    # embeddings = get_embeddings(list_of_chunks)

    # # Upload vector embeddings to Pinecone
    # vectors = []
    # for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    #     vectors.append(
    #         Vector(
    #             id=str(i),
    #             values=embedding.values,
    #             metadata={"text": chunk.page_content},
    #         )
    #     )
    # upload_batches_vectors(vectors=vectors,
    #                        namespace=namespace)  # Run upload vectors only once to prevent duplicate upload

    # Prepare prompt and contexts
    prompt = "Who is Jean Valjean? Tell me about his life and background in Les Misérables by Victor Hugo."
    rag_contexts = query_contexts(prompt)
    vector = get_embedding(rag_contexts)
    res = search_vectors(vector=vector, top_k=top_k, namespace=namespace)

    # Perform reranking (Pinecone has already calculated cosine similarity score,
    # just top n result and add to contexts)
    score_text_pairs = [(match.score, match.metadata["text"]) for match in res.matches]
    score_text_pairs.sort(key=lambda x: x[0], reverse=True)
    top_texts = score_text_pairs[:top_n]
    context = "\n\n".join(text for _, text in top_texts)

    # Use RAG for querying LLM
    output = query_rag(prompt, context)
    # output = query(prompt)
    print(output)
    # Sample output after fully querying:
    # Jean Valjean is the main character in Victor Hugo's novel *Les Misérables*. Here's some information about him:
    #
    # *   **Early Life and Background:** Jean Valjean came from a poor peasant family. He had a sister, a widow with seven children, and helped support them.
    # *   **Imprisonment:** He was imprisoned for stealing a loaf of bread to feed his sister's starving children. He was sentenced to five years in the galleys, but his attempts to escape extended his sentence to nineteen years.
    # *   **Release and Transformation:** Upon his release, Jean Valjean is embittered and distrustful. He struggles to find work due to his convict status.
    # *   **The Bishop's Kindness:** He encounters the Bishop of Digne, who shows him unexpected kindness, offering him food and shelter. This act of grace begins to transform Jean Valjean.
    # *   **Stealing and Redemption:** Jean Valjean steals silverware from the Bishop, but is caught by the police. The Bishop, instead of condemning him, tells the police that he gave Jean Valjean the silverware and includes two silver candlesticks, urging him to use the silver to become an honest man. This encounter profoundly impacts Jean Valjean.
    # *   **Adopting a New Identity:** He breaks his parole and adopts a new identity as Monsieur Madeleine, becoming a successful industrialist and mayor in the town of M. sur M.
    # *   **Moral Struggles:** Throughout the novel, Jean Valjean is constantly grappling with his past and his conscience, torn between his desire to do good and the fear of being discovered and returned to the galleys.
    # *   **Javert's Pursuit:** He is relentlessly pursued by Inspector Javert, a police officer obsessed with upholding the law and who believes Jean Valjean must be brought to justice.
    # *   **Relationship with Cosette:** Jean Valjean becomes a father figure to Cosette, a young girl whose mother he promises to care for. Their relationship is central to the novel.
    # *   **Sacrifice and Redemption:** Jean Valjean makes numerous sacrifices to protect Cosette and help her find happiness. He ultimately finds redemption through his acts of love, kindness, and selflessness.
    # *   **Death:** Jean Valjean dies peacefully, surrounded by Cosette and Marius, having finally found love and a sense of peace.

    # Delete vector embeddings uploaded to Pinecone by deleting namespace
    # NOTE: Vector embeddings belong to a namespace, and namespaces belong to a index (an index is a database)
    # delete_namespace(namespace)
