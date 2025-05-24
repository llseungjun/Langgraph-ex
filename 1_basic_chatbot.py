from typing import Annotated

# from langchain.chat_models import init_chat_model # 유료 API 모델 사용 시
from langchain_groq import ChatGroq # 무료 API 오픈소스 모델 : groq
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

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


def chatbot(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


def stream_graph_updates(user_input: str) -> StateGraph:
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
    return graph


def display_graph(graph: StateGraph) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from io import BytesIO

    try:
        # draw_mermaid_png()가 bytes를 반환하는 경우 처리
        img_bytes = graph.get_graph().draw_mermaid_png()
        img = Image.open(BytesIO(img_bytes))  # 바이트 → PIL 이미지

        plt.imshow(np.array(img))
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("그래프를 표시하는 데 실패했습니다:", e)


if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            graph = stream_graph_updates(user_input)
            display_graph(graph)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            graph = stream_graph_updates(user_input)
            display_graph(graph)
            break