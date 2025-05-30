import os
import json
from typing import Optional, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from utils.chat_util import (
    display_chat_history,
    add_to_chat_history,
    get_retrieved_docs,
    get_input_message,
    stream_data,
)
from utils.llms import RouterAgent, RerankerAgent, ResponderAgent
from utils.vectorstore_util import get_milvus_client, semantic_search


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state variables.

    This function sets up the initial state for chat history and OpenAI configuration.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    if "OPENAI_MODEL" not in st.session_state:
        st.session_state.OPENAI_MODEL = os.environ["OPENAI_MODEL"]


def process_rag_query(
    client: Any,
    reranker: RerankerAgent,
    responder: ResponderAgent,
    user_query: str,
    input_message: str,
) -> str:
    """
    Process a query that requires RAG (Retrieval Augmented Generation).

    Args:
        client: Milvus client instance
        reranker: Reranker agent instance
        responder: Responder agent instance
        user_query: User's question
        input_message: Formatted chat history

    Returns:
        str: Generated response
    """
    # Step 1: Vector search
    with st.spinner("문서를 검색하고 있습니다..."):
        search_result = semantic_search(client, user_query)

    # Step 2: Rerank documents
    with st.spinner("관련 문서를 정렬 중입니다..."):
        retrieved_query = [e["entity"]["query_text"] for e in search_result[0]]
        rank_indices = reranker.rank_documents(user_query, retrieved_query)

    # Step 3: Format retrieved documents
    with st.spinner("답변에 사용할 문서를 정리 중입니다..."):
        retrieved_docs = get_retrieved_docs(search_result, rank_indices)

    # Step 4: Generate response
    with st.spinner("답변을 생성하고 있습니다..."):
        response = responder.generate_response(
            input_message,
            retrieved_docs,
            user_query,
        )

    return response


# Configure page
st.set_page_config(page_title="네이버 스마트스토어 AI FAQ 챗봇", page_icon="🛍️")
st.title("🛍️ 네이버 스마트스토어 AI FAQ 챗봇")

# Load environment variables and initialize session state
load_dotenv(override=True)
initialize_session_state()

# Display chat history
display_chat_history(st.session_state.chat_history)

# Initialize agents and client
client = get_milvus_client()
router = RouterAgent(model=st.session_state.OPENAI_MODEL)
reranker = RerankerAgent(model=st.session_state.OPENAI_MODEL)
responder = ResponderAgent(model=st.session_state.OPENAI_MODEL)

# Handle user input
if user_query := st.chat_input("메시지를 입력하세요"):
    try:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

            # Add current query to the history
            add_to_chat_history(role="user", content=user_query)

        # get input message
        input_message = get_input_message(st.session_state.chat_history)

        # Route the query
        with st.spinner("질문을 분석하고 적절한 답변 방식을 판단 중입니다..."):
            tool_call = router.route_answer(input_message, user_query)
        name = tool_call.name
        args = json.loads(tool_call.arguments)

        if name == "not_relevant":
            response = "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."

            # Remove current query from the history
            st.session_state.chat_history.pop()

        elif name == "rag_required":
            user_query = args["query"]
            response = process_rag_query(
                client, reranker, responder, user_query, input_message
            )

            # Save to chat history
            add_to_chat_history(role="assistant", content=response)

        # Display assistant response
        with st.chat_message("assistant"):
            st.write_stream(stream_data(response))

    except Exception as e:
        st.error(e)
