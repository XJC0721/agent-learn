# 上下文工程 - 在工具中使用上下文（SqliteStore + ToolRuntime[Context]）
# 和 agent18 的区别：agent18 的 ToolRuntime 只能访问 store
# 这里用 ToolRuntime[Context]，工具还能读到调用者传入的 Context
# Context 里的 key 决定返回用户数据的哪个字段（description / birthplace 等）
import sys
import os
import sqlite3
sys.stdout.reconfigure(encoding='utf-8')
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.sqlite import SqliteStore

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)

if os.path.exists("user-info.db"):
    os.remove("user-info.db")

conn = sqlite3.connect("user-info.db", check_same_thread=False, isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA busy_timeout = 30000;")

store = SqliteStore(conn)

store.put(("user_info",), "柳如烟", {"description": "清冷才女，身怀绝技，为寻身世之谜踏入江湖。", "birthplace": "吴兴县"})
store.put(("user_info",), "苏慕白", {"description": "孤傲剑客，剑法超群，背负家族血仇，隐于市井追寻真相。", "birthplace": "杭县"})


# Context 里的 key 由调用方在 agent.invoke 时传入
# 决定工具返回用户数据里的哪个字段
@dataclass
class Context:
    key: str


# ToolRuntime[Context] = 既能访问 store，又能访问 Context
# 和 agent18 相比，多了 runtime.context.key 这一步
@tool
def fetch_user_data(
    user_id: str,
    runtime: ToolRuntime[Context]
) -> str:
    """Fetch user information from the store."""
    # 从 Context 里拿到调用方指定的字段名（如 "birthplace"）
    key = runtime.context.key

    store = runtime.store
    user_info = store.get(("user_info",), user_id)

    if not user_info:
        return f"数据库中没有找到用户 {user_id} 的信息。"

    # 用 key 动态决定返回哪个字段，而不是写死 "description"
    user_desc = user_info.value.get(key, "")
    return f"{key}: {user_desc}"


agent = create_agent(
    model=llm,
    tools=[fetch_user_data],
    store=store,
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "五分钟之内，我要柳如烟的全部信息"}]},
    # 这里告诉工具：返回 birthplace 字段
    context=Context(key="birthplace"),
)

for message in result["messages"]:
    message.pretty_print()
