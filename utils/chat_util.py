import os

import streamlit as st
from datetime import datetime


def display_chat_history(history):
    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])


def add_to_chat_history(role, content):
    chat_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": st.session_state.OPENAI_MODEL,
        "role": role,
        "content": content,
    }
    st.session_state.chat_history.append(chat_entry)


# 최근 5개의 대화를 messages 형식으로 변환
def get_recent_messages(history, window=5):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. 주어진 대화를 바탕으로 이전 문맥을 참고해서 답변해줘.",
        }
    ]
    messages.extend(
        [
            {"role": entry["role"], "content": entry["content"]}
            for entry in history[-(2 * window + 1) :]
        ]
    )
    return messages
