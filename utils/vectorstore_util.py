import os
import re
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient


load_dotenv()
OPENAI_EMBEDDING_MODEL = os.environ["OPENAI_EMBEDDING_MODEL"]


def get_query_embeddings(texts):
    model = OpenAI()
    response = model.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    query_embeddings = [item.embedding for item in response.data]

    return query_embeddings


def clean_query(text):
    # [text] 패턴 전체 삭제
    text = re.sub(r"\[.*?]\s*", "", text)

    # 여러 개의 공백을 하나로 줄이고 strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_answer(text):
    # 제거할 유니코드 문자
    unwanted_chars = [
        "\u00a0",  # non-breaking space (xa0 or &nbsp;)
        "\u3000",  # ideographic space (full-width space)
        "\u200b",  # zero-width space
        "\u200d",  # zero-width joiner
        "\ufeff",  # BOM
    ]

    # 유니코드 문자 제거
    for ch in unwanted_chars:
        text = text.replace(ch, " ")

    # 여러 개의 공백을 하나로 줄이고 strip
    text = re.sub(r"\s+", " ", text).strip()

    # 시스템 문장 제거
    text = re.sub("위 도움말이 도움이 되었나요?.*", "", text, flags=re.DOTALL)

    return text


def get_document_embeddings(texts, batch_size=64):
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
    with open("data/final_result.pkl", "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(list(data.items()), columns=["query", "answer"])

    df["cleaned_query"] = df["query"].apply(clean_query)
    df["cleaned_answer"] = df["answer"].apply(clean_answer)

    # Generate embeddings for all queries
    embeddings = get_document_embeddings(df["cleaned_query"].tolist())

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)

    data = [
        {
            "id": i,
            "vector": embeddings_array[i],
            "query_text": df["cleaned_query"][i],
            "answer_text": df["cleaned_answer"][i],
        }
        for i in range(len(df))
    ]

    # Milvus
    client = MilvusClient("milvus_demo.db")

    if client.has_collection(collection_name="smart_store_faq"):
        client.drop_collection(collection_name="smart_store_faq")

    client.create_collection(
        collection_name="smart_store_faq",
        dimension=1536,
    )
    res = client.insert(collection_name="smart_store_faq", data=data)
