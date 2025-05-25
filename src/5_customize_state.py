from typing import Annotated

from langchain_groq import ChatGroq # 무료 API 오픈소스 모델 : groq
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

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
    name: str # 이름 필드 추가
    birthday: str # 생일 필드 추가

# human assistance tool 정의
@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


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
tools = [tool, human_assistance] # human_assistance tool 추가
llm_with_tools = llm.bind_tools(tools) # llm에 tools 바인딩

# chatbot 함수 정의
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

# graph builder 정의
graph_builder = StateGraph(State)
# graph builder에 챗봇 노드 추가
graph_builder.add_node("chatbot", chatbot)

# grpah builder에 tool 노드 추가
tool_node = ToolNode(tools=tools) # tools를 ToolNode 객체로 전환, ToolNode는 정의된 tool을 excute하는 역할
graph_builder.add_node("tools", tool_node)

# graph builder에 conditional_edges 추가
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition, # 특정 노드에서 다음 노드로 라우팅해주기 위한 모듈
)

# graph builder에 각 노드 간 edge 추가
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot") # start랑 chatbot 노드랑 연결짓는 대신 chatbot 노드를 entry point로 지정
memory = MemorySaver() # memory saver 지정 : multi turn 대화에서 이전 대화 내역 기억 용도
                    # 실제 서비스 운영 시에는 사용자 대화 내역을 db에 저장 후 SqliteSaver 또는 PostgresSaver로 연결하도록 변경
graph = graph_builder.compile(checkpointer=memory) # graph compile시에 checkpointer에 memory saver 지정
                                                # graph stream시에 config 추가 설정 필요, utils.py/stream_graph_updates 참고

if __name__ == "__main__":
    run_chat_loop(graph)