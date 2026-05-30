# 上下文工程 - 使用 Runtime 管理上下文
# 和 agent15 的区别：agent15 从 Store 取用户偏好，这里从 Runtime Context 取用户角色和环境
# 用途：根据用户权限决定能不能用某个工具，根据环境决定操作是否需要格外谨慎
import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)


# Context 里存两个运行时变量：用户角色 和 部署环境
@dataclass
class Context:
    user_role: str        # "admin" 或 "user"，决定能不能用工具
    deployment_env: str   # "production" 或 "development"，决定操作是否要格外小心


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    # 从 Runtime Context 读取用户角色和当前环境
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "You are a helpful assistant."

    # 根据角色决定工具权限
    if user_role == "admin":
        base += "\nYou can use the get_weather tool."
    else:
        base += "\nYou are prohibited from using the get_weather tool."

    # 根据环境决定操作谨慎程度
    if env == "production":
        base += "\nBe extra careful with any data modifications."

    return base


# 模拟天气查询工具
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=llm,
    tools=[get_weather],
    middleware=[context_aware_prompt],
    context_schema=Context,
    # 加上 checkpointer 支持多轮对话记忆（短期记忆）
    checkpointer=InMemorySaver(),
)

# 利用 Runtime 中的两个变量，动态控制 System prompt
# 将 user_role 设为 admin，允许使用天气查询工具
# thread_id 用 uuid 随机生成，保证每次是独立的新对话
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "广州今天的天气怎么样？"}]},
    context=Context(user_role="admin", deployment_env="production"),
    config=config,
)

for message in result["messages"]:
    message.pretty_print()
