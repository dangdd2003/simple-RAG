import os

import requests
import vertexai
from dotenv import load_dotenv
from google.oauth2 import service_account
from vertexai import rag

from source.datasources.utils import extract_text_from_pdf


def main():
    # Install a sample story (Les Misérables in this example)
    # try:
    #     req = requests.get(
    #         "https://cleveracademy.vn/wp-content/uploads/2016/10/Les-Miserables.pdf",
    #         stream=True,
    #     )
    #     req.raise_for_status()
    #     with open("./data/documents/Les-Miserables.pdf", "wb") as file:
    #         for chunk in req.iter_content(chunk_size=8192):
    #             file.write(chunk)
    # except Exception as e:
    #     print(f"Failed to download the sample document: {e}")

    load_dotenv()

    vertexai.init(
        project=os.getenv("RAG_PROJECT_ID"),
        location=os.getenv("RAG_LOCATION"),
        credentials=service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        ),
    )
    print("Vertex AI initialized.")

    corpus_name = os.getenv("RAG_CORPUS_NAME")

    rag.upload_file(
        corpus_name=corpus_name,
        path="./data/documents/Les-Miserables.pdf",
        display_name="Les-Miserables",
    )
    print("File uploaded to RAG corpus.")

    rag.list_files(
        corpus_name=corpus_name,
    )


if __name__ == "__main__":
    main()
