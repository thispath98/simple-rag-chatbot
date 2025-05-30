import os
import json
from datetime import datetime

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]

# 페이지 설정
st.set_page_config(page_title="AI 채팅 인터페이스", page_icon="💬", layout="wide")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")

    st.write(st.session_state)
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.rerun()

# 제목
st.title("💬 AI 채팅 인터페이스")

# 이전 채팅 표시
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])


def add_to_chat_history(role, content):
    chat_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": OPENAI_MODEL,
        "role": role,
        "content": content,
    }
    st.session_state.chat_history.append(chat_entry)


# 최근 4개의 대화를 messages 형식으로 변환
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
            for entry in history[-window:]
        ]
    )
    return messages


# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요"):
    try:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
            add_to_chat_history(role="user", content=prompt)

        # 대화 기록 준비
        messages = get_recent_messages(st.session_state.chat_history)
        print(messages, "\n")

        # AI 응답을 위한 채팅 메시지 컨테이너 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # 스트리밍 응답 처리
            stream = OpenAI().chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
            for response in stream:
                if response.choices[0].delta.content is not None:
                    full_response += response.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")  # like cmd prompt

            # 최종 응답 표시
            message_placeholder.markdown(full_response)

        # 대화 기록 저장
        add_to_chat_history(role="assistant", content=full_response)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.info("OpenAI API 키가 올바르게 설정되어 있는지 확인해주세요.")
