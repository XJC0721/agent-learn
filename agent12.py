# 长期记忆进阶：让 Agent 通过工具读取记忆，而不是直接操作 store
# 和 agent11 的区别：agent11 是代码直接读写 store，这里是 Agent 自己决定什么时候调工具去读
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass

EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024

_ = load_dotenv()

# embedding 用阿里云
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    base_url=os.getenv("DASHSCOPE_BASE_URL", ""),
)

# 对话用 DeepSeek
model = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
    temperature=0.7,
)


def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIM,
    )
    return [item.embedding for item in response.data]


# 创建向量数据库并预存两条用户记忆
store = InMemoryStore(index={"embed": embed, "dims": EMBED_DIM})

store.put(
    ("users",),
    "user_1",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "rule_id": "3",
    },
)

store.put(
    ("users",),
    "user_2",
    {
        "name": "John Smith",
        "language": "English",
    }
)

# ── 使用工具读取长期记忆 ──────────────────────────────────────────

# Context 是传给 Agent 的运行时上下文，定义了每次调用时携带哪些信息
# @dataclass 让这个类自动生成 __init__ 等方法，不用手写
@dataclass
class Context:
    user_id: str  # 当前用户的 ID，Agent 调工具时会用这个去 store 里查


# @tool 把普通函数变成 Agent 可以调用的工具
# ToolRuntime[Context] 让工具能访问 store 和 context，不需要通过参数传进来
@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    # 从 runtime 拿到 store 和当前用户 ID
    store = runtime.store
    user_id = runtime.context.user_id

    # 按 ID 精确查用户记忆，返回 StoreValue 对象（包含 value 和元数据）
    user_info = store.get(("users",), user_id)

    # 有就返回内容，没有就返回 Unknown user
#     Item(
#     namespace=['users'],
#     key='user_1',
#     value={'rules': [...], 'rule_id': '3'},  # ← 这个就是 .value
#     created_at='2026-05-26...',
#     updated_at='2026-05-26...',
#     score=0.408...
# )
    return str(user_info.value) if user_info else "Unknown user"


# 创建 Agent，把 store 和 context_schema 传进去
# store=store        → Agent 调工具时能访问这个向量数据库
# context_schema     → 告诉 Agent 每次 invoke 时会传入什么格式的 context

agent = create_agent(
    model=model,
    tools=[get_user_info],
    store=store,
    context_schema=Context
)

# 运行 Agent，context 指定当前是 user_2 在问
# Agent 会自动判断需要调 get_user_info 工具，然后把结果整合成回答
result = agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_2")
)

for message in result["messages"]:
    message.pretty_print()
