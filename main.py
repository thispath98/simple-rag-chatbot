import os

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from utils.chat_util import (
    display_chat_history,
    add_to_chat_history,
    get_recent_messages,
)
from utils.llms import OpenAIStreamResponder


# 페이지 설정
st.set_page_config(page_title="AI 채팅 인터페이스", page_icon="💬", layout="wide")

# 제목
st.title("💬 AI 채팅 인터페이스")

# 환경 변수 로드, 세션 상태 초기화
load_dotenv(override=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
if "OPENAI_MODEL" not in st.session_state:
    st.session_state.OPENAI_MODEL = os.environ["OPENAI_MODEL"]

# 이전 대화 표시
display_chat_history(st.session_state.chat_history)

stream_responder = OpenAIStreamResponder(model=st.session_state.OPENAI_MODEL)

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요"):
    try:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
            add_to_chat_history(role="user", content=prompt)

        # 대화 기록 준비
        messages = get_recent_messages(st.session_state.chat_history)

        # 응답 스트리밍
        response = stream_responder.stream_response(messages)

        # 대화 기록 저장
        add_to_chat_history(role="assistant", content=response)

    except Exception as e:
        st.error(e)
