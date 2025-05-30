import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
import yaml


class OpenAIStreamResponder:
    def __init__(
        self,
        model: str,
        prompt_type: str = "default_prompt",
        prompt_file_path: str = "configs/prompt_templates.yaml",
    ):
        self.client = OpenAI()
        self.model = model
        self.prompt_type = prompt_type
        self.prompt_file_path = prompt_file_path
        self.prompt_template = self.load_prompt_template()

    def stream_response(self, messages: list[str]) -> str:
        """스트리밍 응답을 받아 실시간 출력하며 최종 응답을 반환"""
        messages = self.prompt_template.format(question=messages[-1])
        print(messages)

        full_response = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                if isinstance(chunk, ChatCompletionChunk):
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_response += delta.content
                        message_placeholder.markdown(full_response)

            message_placeholder.markdown(full_response)
            return full_response

    def load_prompt_template(self) -> dict:
        with open(self.prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = yaml.safe_load(f)[self.prompt_type]
        return prompt_template
