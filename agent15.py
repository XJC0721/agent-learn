# 上下文工程 - 使用 Store 管理上下文
# 和 agent14 的区别：agent14 根据 State（消息数量）改 prompt，这里根据 Store（用户偏好）改 prompt
# 1. agent.invoke() 被调用
#    传入消息 + context=Context(user_id="user_1")
#          ↓
# 2. @dynamic_prompt 中间件触发
#    从 request.runtime.store 取 user_1 的偏好
#    store.get(("preferences",), "user_1") → {"communication_style": "Chinese"}
#    拼出 system prompt：
#    "You are a helpful assistant.\nUser prefers Chinese responses."
#          ↓
# 3. 把 system prompt 插到消息列表最前面
#    [
#      {"role": "system", "content": "You are a helpful assistant.\nUser prefers Chinese responses."},
#      {"role": "system", "content": "You are a helpful assistant. Please be extra concise."},
#      {"role": "user",   "content": 'What is a "hold short line"?'}
#    ]
#          ↓
# 4. 完整消息列表发给 DeepSeek
#          ↓
# 5. DeepSeek 回答：
#    "A hold short line is a painted marking..."
#          ↓
# 6. pretty_print() 打印所有消息
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)


# 每次调用携带的用户上下文，用于识别当前是哪个用户
@dataclass
class Context:
    user_id: str


@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    # 通过 request.runtime 拿到 context 和 store（runtime 是总入口）
    user_id = request.runtime.context.user_id
    store = request.runtime.store

    # 从 Store 里读取该用户的偏好设置
    user_prefs = store.get(("preferences",), user_id)

    base = "You are a helpful assistant."

    # 如果有偏好记录，把偏好加进 system prompt
    if user_prefs:
#         字典["key"]          # 找不到会报错
# 字典.get("key", 默认) # 找不到返回默认值，更安全
        style = user_prefs.value.get("communication_style", "balanced")
        # base += f"\n必须用User prefers {style} responses.不管用户用什么语言提问。"
        base += f"\n必须用{style}回复用户，不管用户用什么语言提问。"

    return base


# 不需要向量搜索，只需按 ID 精确取用户偏好，所以不传 index
store = InMemoryStore()

agent = create_agent(
    model=llm,
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=store,
)

# 预置两条用户偏好信息
store.put(("preferences",), "user_1", {"communication_style": "Chinese中文"})
store.put(("preferences",), "user_2", {"communication_style": "Korean"})

# 用户1喜欢中文回复，system prompt 会自动加上 "User prefers Chinese responses"
result = agent.invoke(
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant. Please be extra concise."},
        {"role": "user", "content": 'What is a "hold short line"?'}
    ]},
    context=Context(user_id="user_1"),
)

for message in result["messages"]:
    message.pretty_print()
