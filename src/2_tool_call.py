from typing import Annotated

from langchain_groq import ChatGroq # 무료 API 오픈소스 모델 : groq
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from utils import run_chat_loop

import os
from dotenv import load_dotenv

load_dotenv()
# api key 호출
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# State class 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]

# graph builder 정의
graph_builder = StateGraph(State)

# llm 모델 호출
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=300,
    api_key=GROQ_API_KEY
)

# tools 정의
tool = TavilySearch(max_results=2) # TavilySearch : 웹 검색 엔진
                                   # max_results : 최대 검색 결과
tools = [tool]
llm_with_tools = llm.bind_tools(tools) # llm에 tools 바인딩

# chatbot 함수 정의
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# graph builder에 챗봇 노드 추가
graph_builder.add_node("chatbot", chatbot)

# grpah builder에 tool 노드 추가
tool_node = ToolNode(tools=[tool]) # tools를 ToolNode 객체로 전환, ToolNode는 정의된 tool을 excute하는 역할
graph_builder.add_node("tools", tool_node)

# graph builder에 conditional_edges 추가
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition, # 특정 노드에서 다음 노드로 라우팅해주기 위한 모듈
)
# Any time a tool is called, we return to the chatbot to decide the next step
# graph builder에 각 노드 간 edge 추가
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile() # graph 컴파일

if __name__ == "__main__":
    run_chat_loop(graph)