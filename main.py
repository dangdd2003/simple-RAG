from source.datasources.pinecone import search_vectors
from source.embeddings.gemini_embedding import get_embedding
from source.helper_function import set_logger

logger = set_logger("main")


def main():
    # embedding = get_embeddings(
    #     [
    #         "Hello, world! This is a test for Gemini embedding.",
    #         "This is a second test message for embedding generation.",
    #     ]
    # ).embeddings
    # upload_vectors([
    #                    {
    #                        "id": "1",
    #                        "values": embedding[0].values,
    #                        "metadata": {"source": "gemini"}
    #                    },
    #                    {
    #                        "id": "2",
    #                        "values": embedding[1].values,
    #                        "metadata": {"source": "gemini"}
    #                    }
    #                ], namespace="gemini_test")

    query = get_embedding("Hello, world! This is a test for Gemini embedding.")
    res = search_vectors(query, namespace="gemini_test")
    # print(res.matches[0].values)
    print(res)
    # delete_namespace("gemini_test")
    # text = extract_text_from_pdf(
    #     "./Gray-scale image enhancement as an automatic process driven by evolution.pdf"
    # )
    # print(text[:1000])


if __name__ == "__main__":
    main()
