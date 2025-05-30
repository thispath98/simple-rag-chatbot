import json
import yaml
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI


# Define available tools for the router agent
tools = [
    {
        "type": "function",
        "name": "not_relevant",
        "description": "Use this function when the user's question is unrelated to the current 'Naver Smart Store' or context.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "additionalProperties": False,
    },
    {
        "type": "function",
        "name": "rag_required",
        "description": "Use this function when the user's question requires relevant information from documents about 'Naver Smart Store'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The refined version of the user's question for document retrieval.",
                }
            },
            "required": ["query"],
        },
        "additionalProperties": False,
    },
]


class BaseAgent:
    """
    Base class for all LLM-based agents.

    This class provides common functionality for loading prompt templates and initializing
    the OpenAI client.
    """

    def __init__(self, model: str, prompt_type: str, prompt_file_path: str):
        """
        Initialize the base agent.

        Args:
            model (str): Name of the OpenAI model to use
            prompt_type (str): Type of prompt template to load
            prompt_file_path (str): Path to the YAML file containing prompt templates
        """
        self.client = OpenAI()
        self.model = model
        self.prompt_type = prompt_type
        self.prompt_file_path = prompt_file_path
        self.prompt_template = self.load_prompt_template()

    def load_prompt_template(self) -> str:
        """
        Load the prompt template from the YAML file.

        Returns:
            str: The loaded prompt template for the specified type
        """
        with open(self.prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = yaml.safe_load(f)[self.prompt_type]
        return prompt_template


class RouterAgent(BaseAgent):
    """
    Agent responsible for routing user queries to appropriate handlers.

    This agent determines whether a query requires RAG (Retrieval Augmented Generation)
    or can be handled directly.
    """

    def __init__(
        self,
        model: str,
        prompt_file_path: str = "configs/prompt_templates.yaml",
    ):
        """
        Initialize the router agent.

        Args:
            model (str): Name of the OpenAI model to use
            prompt_file_path (str, optional): Path to prompt templates. Defaults to "configs/prompt_templates.yaml"
        """
        super().__init__(
            model,
            prompt_type="router_prompt",
            prompt_file_path=prompt_file_path,
        )

    def route_answer(self, input_message: str, user_query: str) -> str:
        """
        Route the user query to determine if RAG is required.

        Args:
            input_message (str): Formatted chat history
            user_query (str): User's question

        Returns:
            str: The routing decision (function call)
        """
        system_prompt = self.prompt_template.replace("{{history}}", input_message)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        response = self.client.responses.create(
            model=self.model,
            input=messages,
            tools=tools,
        )
        return response.output[0]


class RerankerAgent(BaseAgent):
    """
    Agent responsible for reranking retrieved documents based on relevance.

    This agent evaluates the relevance of each retrieved document to the user's query
    and returns indices of documents that meet the relevance threshold.
    """

    def __init__(
        self,
        model: str,
        prompt_file_path: str = "configs/prompt_templates.yaml",
        threshold: float = 0.5,
    ):
        """
        Initialize the reranker agent.

        Args:
            model (str): Name of the OpenAI model to use
            prompt_file_path (str, optional): Path to prompt templates. Defaults to "configs/prompt_templates.yaml"
            threshold (float, optional): Minimum relevance score. Defaults to 0.5
        """
        super().__init__(
            model,
            prompt_type="reranker_prompt",
            prompt_file_path=prompt_file_path,
        )
        self.threshold = threshold

    def rank_documents(self, user_query: str, retrieved_query: List[str]) -> List[int]:
        """
        Rank retrieved documents based on their relevance to the user query.

        Args:
            user_query (str): User's question
            retrieved_query (List[str]): List of retrieved documents

        Returns:
            List[int]: Indices of documents that meet the relevance threshold, sorted by score
        """
        results = []
        for i, doc in enumerate(retrieved_query):
            messages = [
                {"role": "system", "content": self.prompt_template},
                {"role": "user", "content": f"Query: {user_query}\nDocument: {doc}"},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            try:
                raw = response.choices[0].message.content
                score = json.loads(raw)["score"]
            except Exception:
                score = 0.0

            if score >= self.threshold:
                results.append((i, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in results]


class ResponderAgent(BaseAgent):
    """
    Agent responsible for generating responses to user queries.

    This agent uses retrieved documents and chat history to generate contextually
    appropriate responses.
    """

    def __init__(
        self,
        model: str,
        prompt_file_path: str = "configs/prompt_templates.yaml",
    ):
        """
        Initialize the responder agent.

        Args:
            model (str): Name of the OpenAI model to use
            prompt_file_path (str, optional): Path to prompt templates. Defaults to "configs/prompt_templates.yaml"
        """
        super().__init__(
            model,
            prompt_type="responder_prompt",
            prompt_file_path=prompt_file_path,
        )

    def generate_response(
        self,
        input_message: str,
        retrieved_docs: str,
        user_query: str,
    ) -> str:
        """
        Generate a response to the user query using retrieved documents and chat history.

        Args:
            input_message (str): Formatted chat history
            retrieved_docs (str): Retrieved relevant documents
            user_query (str): User's question

        Returns:
            str: Generated response
        """
        system_prompt = self.prompt_template.replace("{{history}}", input_message)
        system_prompt = system_prompt.replace("{{retrieved_docs}}", retrieved_docs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content
