import os
import time
from typing import List, Dict, Any, Generator

import streamlit as st
from datetime import datetime


def display_chat_history(history: List[Dict[str, Any]]) -> None:
    """
    Display the chat history in the Streamlit interface.

    Args:
        history (List[Dict[str, Any]]): List of chat entries containing role and content
    """
    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])


def add_to_chat_history(role: str, content: str) -> None:
    """
    Add a new chat entry to the session state chat history.

    Args:
        role (str): Role of the message sender (e.g., 'user', 'assistant')
        content (str): Content of the message
    """
    chat_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": st.session_state.OPENAI_MODEL,
        "role": role,
        "content": content,
    }
    st.session_state.chat_history.append(chat_entry)


def get_retrieved_docs(
    search_result: List[List[Dict[str, Any]]], rank_indices: List[int]
) -> str:
    """
    Format retrieved documents from search results into a readable string.

    Args:
        search_result (List[List[Dict[str, Any]]]): Search results from vector store
        rank_indices (List[int]): Indices of the documents to retrieve

    Returns:
        str: Formatted string containing questions and answers from retrieved documents
    """
    retrieved_docs = "\n\n".join(
        [
            f"Question: {search_result[0][i]['query_text']}\nAnswer: {search_result[0][i]['answer_text']}"
            for i in rank_indices
        ]
    )
    return retrieved_docs


def get_input_message(history: List[Dict[str, Any]], window: int = 5) -> str:
    """
    Convert recent chat history into a formatted message string.

    Args:
        history (List[Dict[str, Any]]): List of chat entries
        window (int, optional): Number of recent exchanges to include. Defaults to 5.

    Returns:
        str: Formatted string containing recent chat history
    """
    # Convert recent chat history into a formatted message
    input_message = "\n\n".join(
        [
            f"{entry['role']}:\n{entry['content']}"
            for entry in history[-(2 * window + 1) :]
        ]
    )

    return input_message


def stream_data(text: str) -> Generator[str, None, None]:
    """
    Stream text word by word with a small delay between each word.

    Args:
        text (str): Text to be streamed

    Yields:
        str: Words from the input text with spaces
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)
