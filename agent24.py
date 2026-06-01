# 并行 - 节点并行
# 两个节点同时从 START 出发，同时跑，互不等待
# 通过 time.sleep 模拟耗时，验证并行效果（总时间 ≈ 最慢的那个，而不是两者相加）
import sys
import time
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8')
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END


def node_a(state: MessagesState):
    start_time = datetime.now()
    print(f"[node_a] 进入函数时间: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    time.sleep(2)  # 模拟耗时 2 秒

    end_time = datetime.now()
    print(f"[node_a] 退出函数时间: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    return {'messages': [HumanMessage(
        content=f'节点a运行了{round((end_time - start_time).total_seconds(), 3)}秒'
    )]}


def node_b(state: MessagesState):
    start_time = datetime.now()
    print(f"[node_b] 进入函数时间: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    time.sleep(4)  # 模拟耗时 4 秒

    end_time = datetime.now()
    print(f"[node_b] 退出函数时间: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    return {'messages': [HumanMessage(
        content=f'节点b运行了{round((end_time - start_time).total_seconds(), 3)}秒'
    )]}


builder = StateGraph(MessagesState)

builder.add_node('node_a', node_a)
builder.add_node('node_b', node_b)

# 关键：两个节点都从 START 出发、都连到 END
# LangGraph 检测到这种结构，会自动并行执行 node_a 和 node_b
builder.add_edge(START, 'node_a')
builder.add_edge(START, 'node_b')
builder.add_edge('node_a', END)
builder.add_edge('node_b', END)

my_graph = builder.compile(name='my-graph')

response = my_graph.invoke({
    'messages': [HumanMessage(content='执行 node_a 和 node_b')]
})

for message in response['messages']:
    message.pretty_print()
