# 长期记忆写入：让 Agent 通过工具把对话中的信息主动存进 store
# 和 agent12 的区别：agent12 是读记忆，这里是写记忆，Agent 自己判断什么时候存
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass
from typing_extensions import TypedDict

_ = load_dotenv()

# 对话用 DeepSeek
model = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
    temperature=0.7,
)

# 这里不需要向量搜索，只需要按 ID 精确存取，所以不传 index
# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production.
store = InMemoryStore()

# 每次调用携带的用户上下文
@dataclass
class Context:
    user_id: str


# TypedDict 定义存入 store 的数据结构，给 LLM 看的 schema
# 和 @dataclass 的区别：TypedDict 是给 LLM 解析用的，定义它应该提取哪些字段
class UserInfo(TypedDict):
    name: str


# 写入工具：Agent 从对话中提取用户信息后调这个工具存进 store
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    # 从 runtime 拿到 store 和当前用户 ID
    store = runtime.store
    user_id = runtime.context.user_id

    # 把用户信息存入 store（namespace, key, data）
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."


# 创建 Agent，工具换成写入工具 save_user_info
agent = create_agent(
    model=model,
    tools=[save_user_info],
    store=store,
    context_schema=Context
)

# 运行 Agent：用户说了自己的名字，Agent 应该提取并存进 store
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    # context 指定当前用户是 user_123，存的时候用这个 ID 作为 key
    context=Context(user_id="user_123")
)

# 直接从 store 按 ID 取出，验证是否存进去了
print(store.get(("users",), "user_123").value)
