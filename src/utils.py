from langgraph.graph import StateGraph

def run_chat_loop(graph: StateGraph) -> None:
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(graph, user_input)
            display_graph(graph)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(graph, user_input)
            display_graph(graph)
            break

def stream_graph_updates(graph: StateGraph, user_input: str) -> None:
    config = {"configurable": {"thread_id": "1"}} # 특정 대화에서 핵심 쓰레드로 사용될 id 값 설정
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, 
        config, 
        stream_mode="values" ,
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

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

