# 在 StateGraph 中使用短期记忆（Short-term Memory）
# checkpointer 把对话历史存起来，同一个 thread_id 的多次 invoke 可以共享记忆
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# 助手节点：把系统提示 + 历史消息一起传给 LLM
def assistant(state: MessagesState):
    system_prompt = "You are a helpful assistant."
    all_messages = [SystemMessage(system_prompt)] + state["messages"]
    return {"messages": [llm.invoke(all_messages)]}

# ── 创建短期记忆 ──────────────────────────────────────────

# InMemorySaver 把每次对话状态存在内存里
# 程序关闭后记忆消失，生产环境换 SqliteSaver / PostgresSaver 等持久化方案
checkpointer = InMemorySaver()

# ── 创建图 ────────────────────────────────────────────────

builder = StateGraph(MessagesState)

# 只有一个节点：assistant
builder.add_node("assistant", assistant)

# 边：START → assistant → END（简单的线性流程，不需要条件边）
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# compile 时传入 checkpointer，图才具备记忆能力
# 每次 invoke 执行完，状态会自动存入 checkpointer
graph = builder.compile(checkpointer=checkpointer)

# ── 第一轮：告诉 Agent 我叫 zhangsan ──────────────────────

# thread_id="1" 是这段对话的唯一标识
# 同一个 thread_id 的多次 invoke 共享同一份历史记录
result = graph.invoke(
    {"messages": ["hi! i am zhangsan"]},
    {"configurable": {"thread_id": "1"}},
)

for message in result["messages"]:
    message.pretty_print()

# ── 第二轮：问 Agent 我叫什么名字 ─────────────────────────

# 同一个 thread_id="1"，Agent 能从 checkpointer 里读到上一轮的对话
# 所以它知道你叫 luochang
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    {"configurable": {"thread_id": "1"}},
)

for message in result["messages"]:
    message.pretty_print()
