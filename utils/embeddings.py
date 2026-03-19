import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import List

# Load environment variables to get the API key
load_dotenv()

# Initialize the ASYNC OpenAI client once when the module is imported
# This is efficient and best practice for FastAPI.
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def get_openai_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Create embeddings for a batch of texts using OpenAI async client.
    """
    try:
        response = await client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"[ERROR] Embedding creation failed: {e}")
        return []