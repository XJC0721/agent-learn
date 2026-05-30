# 上下文工程 - 在工具中使用上下文（SqliteStore 持久化存储）
# 和 agent11-13 的区别：之前用 InMemoryStore（内存，程序关了就没），这里用 SqliteStore（持久化到本地文件）
import sys
import os
import sqlite3
sys.stdout.reconfigure(encoding='utf-8')
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

# 删除旧数据库，保证每次运行都是干净的
if os.path.exists("user-info.db"):
    os.remove("user-info.db")

# 创建 SQLite 连接
# check_same_thread=False → 允许多线程访问
# isolation_level=None    → 自动提交，不需要手动 commit
conn = sqlite3.connect("user-info.db", check_same_thread=False, isolation_level=None)

# WAL 模式：写入时不锁表，读写可以同时进行，性能更好
conn.execute("PRAGMA journal_mode=WAL;")
# 等待锁超时时间 30 秒，避免多线程时报错
conn.execute("PRAGMA busy_timeout = 30000;")

# 用 SQLite 连接创建 store，用法和 InMemoryStore 完全一样
store = SqliteStore(conn)

# 预置两条用户信息（存到 user_info 分类下，key 是用户名）
store.put(("user_info",), "柳如烟", {"description": "清冷才女，身怀绝技，为寻身世之谜踏入江湖。", "birthplace": "吴兴县"})
store.put(("user_info",), "苏慕白", {"description": "孤傲剑客，剑法超群，背负家族血仇，隐于市井追寻真相。", "birthplace": "杭县"})


# 工具：从 store 里查用户信息
# user_id 是普通参数，由 Agent 从对话中提取
# runtime 是 ToolRuntime，由框架自动注入，不需要 Agent 传
@tool
def fetch_user_data(
    user_id: str,
    runtime: ToolRuntime
) -> str:
    """
    Fetch user information from the in-memory store.

    :param user_id: The unique identifier of the user.
    :param runtime: The tool runtime context injected by the framework.
    :return: The user's description string if found; an empty string otherwise.
    """
    store = runtime.store
    user_info = store.get(("user_info",), user_id)

    if user_info:
        return user_info.value.get("description", "")

    # 明确告诉 LLM 数据库里没有此人，避免 LLM 自己编造
    return f"数据库中没有找到用户 {user_id} 的信息。"


agent = create_agent(
    model=llm,
    tools=[fetch_user_data],
    store=store,
    # 严格约束：只能根据工具返回的内容回答，不能用自己的知识补充
    system_prompt="你只能根据工具返回的结果回答问题。如果工具说找不到某人的信息，就直接告诉用户找不到，不要用你自己的知识补充任何内容。",
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "五分钟之内，我要柳如烟的全部信息"
    }]
})

for message in result["messages"]:
    message.pretty_print()
