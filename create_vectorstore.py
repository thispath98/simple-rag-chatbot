import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient


load_dotenv()


def get_query_embeddings(texts):
    response = OpenAI().embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
        encoding_format="float",
    )
    query_embeddings = [item.embedding for item in response.data]

    return query_embeddings


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv", encoding="utf-8-sig")

    with open("data/embeddings.pkl", "rb") as f:
        loaded_embeddings = pickle.load(f)

    data = [
        {
            "id": i,
            "vector": loaded_embeddings[i],
            "query_text": df["cleaned_query"][i],
            "answer_text": df["cleaned_answer"][i],
        }
        for i in range(len(df))
    ]

    # Milvus
    client = MilvusClient("milvus_demo.db")

    if client.has_collection(collection_name="demo_collection"):
        client.drop_collection(collection_name="demo_collection")

    client.create_collection(
        collection_name="demo_collection",
        dimension=1536,  # The vectors we will use in this demo have 768 dimensions
    )
    res = client.insert(collection_name="demo_collection", data=data)

    q_e = get_query_embeddings(["스마트스토어센터 회원가입은 어떻게 하나요?"])

    res = client.search(
        collection_name="demo_collection",  # target collection
        data=q_e,  # query vectors
        limit=2,  # number of returned entities
        output_fields=["query_text", "answer_text"],  # specifies fields to be returned
    )

    print(res)
