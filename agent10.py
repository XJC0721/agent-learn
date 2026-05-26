# 在 create_agent 中使用短期记忆
# 和 agent09 的区别：用 create_agent 代替手动搭 StateGraph，更简洁
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# 创建短期记忆
checkpointer = InMemorySaver()

# create_agent 直接支持传入 checkpointer，不需要手动 build 图
# 效果和 agent09 的 StateGraph 完全一样，只是写法更简单
agent = create_agent(
    model=llm,
    checkpointer=checkpointer,
)

# ── 第一轮：告诉 Agent 我叫  zhangsan ─────────────────────

# thread_id="2" 和 agent09 的 "1" 是两个独立的对话，互不干扰
result = agent.invoke(
    {"messages": ["hi! i am zhangsan"]},
    {"configurable": {"thread_id": "2"}},
)

for message in result["messages"]:
    message.pretty_print()

# ── 第二轮：问 Agent 我叫什么名字 ────────────────────────

# 同一个 thread_id="2"，Agent 能记住第一轮说过的名字
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    {"configurable": {"thread_id": "2"}},
)

for message in result["messages"]:
    message.pretty_print()
