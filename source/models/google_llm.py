import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from source.helper_function import set_logger

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# model_name = "models/gemini-2.5-flash-preview-05-20"
model_name = "models/gemini-2.0-flash-lite"

logger = set_logger("google-LLM")


def query(prompt: str) -> str | None:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=f"{prompt}",
        )
        logger.info(msg=f"Query LLM using '{model_name}'")
        return response.text
    except Exception as e:
        logger.warning(msg=f"Failed to query LLM using '{model_name}': {e}")
        return None


def query_contexts(prompt: str) -> str | None:
    try:
        rag_contexts = client.models.generate_content(
            model=model_name,
            contents=f"""
            User prompt:
            {prompt}
            Return the texts only, no conversation.
""",
            config=types.GenerateContentConfig(
                system_instruction="You are RAG tool helper. Extract key contexts from the user questions to use for RAG."
            ),
        )
        logger.info(msg=f"Query contexts for RAG using '{model_name}'")
        return rag_contexts.text
    except Exception as e:
        logger.warning(
            msg=f"Failed to query contexts for RAG using '{model_name}': {e}"
        )
        return None


def query_rag(prompt: str, context: str) -> str | None:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=f"""
            Contexts:
            {context}
            
            User question:
            {prompt}
    """,
            config=types.GenerateContentConfig(
                system_instruction="""
                You are helpful assistant with RAG tool. Answer the user questions based on the provided contexts.
                """
            ),
        )
        logger.info(msg=f"Query LLM (RAG enabled) using '{model_name}'")
        return response.text
    except Exception as e:
        logger.warning(
            msg=f"Failed to query LLM (RAG enabled) using '{model_name}': {e}"
        )
        return None
