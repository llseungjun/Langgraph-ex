from typing import Annotated

# from langchain.chat_models import init_chat_model # 유료 API 모델 사용 시
from langchain_groq import ChatGroq # 무료 API 오픈소스 모델 : groq
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from utils import stream_graph_updates, display_graph

from utils import run_chat_loop

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# llm = init_chat_model("anthropic:claude-3-5-sonnet-latest") 유료 API 모델 사용 시
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=300,
    api_key=GROQ_API_KEY
)


def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


if __name__ == "__main__":
    run_chat_loop(graph)