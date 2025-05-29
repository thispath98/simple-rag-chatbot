import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_embeddings(texts, batch_size=128):
    all_embeddings = []

    # 배치 단위로 처리
    client = OpenAI()
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch,
            encoding_format="float",
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv", encoding="utf-8-sig")

    # Generate embeddings for all queries
    embeddings = get_embeddings(df["cleaned_query"].tolist())

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)

    # Save embeddings in pickle format (more universal format)
    with open("data/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_array, f)

    # Check saved embedding vector
    print(f"Saved embedding vector shape: {embeddings_array.shape}")
