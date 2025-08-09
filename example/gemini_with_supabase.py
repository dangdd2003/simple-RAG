import os

import requests
from supabase import Client, create_client

from source.datasources.utils import extract_text_from_pdf
from source.embeddings.gemini_embedding import (
    get_embedding,
    get_embeddings,
    semantic_chunking,
)
from source.models.google_llm import query_contexts, query_rag


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

    chunks = semantic_chunking(content)

    # Perform embedding chunks
    list_of_chunks = [chunk.page_content for chunk in chunks]
    embeddings = get_embeddings(list_of_chunks)

    # Prepare datasources
    url: str = os.environ.get("SUPABASE_URL", "")
    key: str = os.environ.get("SUPABASE_KEY", "")

    supabase: Client = create_client(url, key)

    # Upload vector embeddings to Supabase
    row = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        row.append(
            {
                "content": chunk.page_content,
                "embedding": embedding.values,
            }
        )
    supabase.schema("public").table("les_miserables").insert(row).execute()

    # Prepare the prompt
    prompt = "Who is Jean Valjean? Tell me about his life and background in Les Misérables by Victor Hugo."
    rag_contexts = query_contexts(prompt)
    embedding = get_embedding(rag_contexts)
    res = supabase.rpc(
        "match_documents",
        {
            "query_embedding": embedding,
            "match_count": 30,
        },
    ).execute()
    context = "\n\n".join(chunk["content"] for chunk in res.data)

    # Query the model
    output = query_rag(prompt, context)
    print(output)
