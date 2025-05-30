import os
import re
import pickle
from typing import List, Dict, Any, Union

from tqdm import tqdm
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient


# Load environment variables
load_dotenv()
OPENAI_EMBEDDING_MODEL = os.environ["OPENAI_EMBEDDING_MODEL"]


@st.cache_resource
def get_milvus_client() -> MilvusClient:
    """
    Get a cached Milvus client instance.

    Returns:
        MilvusClient: A Milvus client instance connected to the local database.
    """
    return MilvusClient("milvus_demo.db")


def semantic_search(client: MilvusClient, user_query: str) -> List[Dict[str, Any]]:
    """
    Perform semantic search on the FAQ collection using the user's query.

    Args:
        client (MilvusClient): Milvus client instance
        user_query (str): User's search query

    Returns:
        List[Dict[str, Any]]: List of search results containing query and answer texts
    """
    query_embeddings = get_query_embeddings([user_query])

    search_result = client.search(
        collection_name="smart_store_faq",
        data=query_embeddings,
        limit=5,
        output_fields=[
            "query_text",
            "answer_text",
        ],
    )
    return search_result


def get_query_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for the given texts using OpenAI's embedding model.

    Args:
        texts (List[str]): List of texts to generate embeddings for

    Returns:
        List[List[float]]: List of embedding vectors
    """
    model = OpenAI()
    response = model.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    query_embeddings = [item.embedding for item in response.data]

    return query_embeddings


def clean_query(text: str) -> str:
    """
    Clean the query text by removing brackets and extra whitespace.

    Args:
        text (str): Raw query text

    Returns:
        str: Cleaned query text
    """
    # Remove [text] pattern
    text = re.sub(r"\[.*?]\s*", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_answer(text: str) -> str:
    """
    Clean the answer text by removing unwanted characters and system messages.

    Args:
        text (str): Raw answer text

    Returns:
        str: Cleaned answer text
    """
    # Unicode characters to remove
    unwanted_chars = [
        "\u00a0",  # non-breaking space (xa0 or &nbsp;)
        "\u3000",  # ideographic space (full-width space)
        "\u200b",  # zero-width space
        "\u200d",  # zero-width joiner
        "\ufeff",  # BOM
    ]

    # Remove unwanted characters
    for ch in unwanted_chars:
        text = text.replace(ch, " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove system messages
    text = re.sub("위 도움말이 도움이 되었나요?.*", "", text, flags=re.DOTALL)

    return text


def get_document_embeddings(
    texts: List[str], batch_size: int = 64
) -> List[List[float]]:
    """
    Generate embeddings for a list of documents in batches.

    Args:
        texts (List[str]): List of documents to generate embeddings for
        batch_size (int, optional): Size of each batch. Defaults to 64.

    Returns:
        List[List[float]]: List of embedding vectors for all documents
    """
    all_embeddings = []

    model = OpenAI()
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i : i + batch_size]
        response = model.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


if __name__ == "__main__":
    # Load and process FAQ data
    with open("data/final_result.pkl", "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(list(data.items()), columns=["query", "answer"])

    # Clean query and answer texts
    df["cleaned_query"] = df["query"].apply(clean_query)
    df["cleaned_answer"] = df["answer"].apply(clean_answer)

    # Generate embeddings for all queries
    embeddings = get_document_embeddings(df["cleaned_query"].tolist())

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)

    # Prepare data for Milvus insertion
    data = [
        {
            "id": i,
            "vector": embeddings_array[i],
            "query_text": df["cleaned_query"][i],
            "answer_text": df["cleaned_answer"][i],
        }
        for i in range(len(df))
    ]

    # Initialize Milvus client and create collection
    client = MilvusClient("milvus_demo.db")

    if client.has_collection(collection_name="smart_store_faq"):
        client.drop_collection(collection_name="smart_store_faq")

    client.create_collection(
        collection_name="smart_store_faq",
        dimension=1536,
    )
    res = client.insert(collection_name="smart_store_faq", data=data)
