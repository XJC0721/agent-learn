# 流式输出（Streaming）
import os
from typing import Literal, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

load_dotenv()

# 配置大模型服务
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# 定义权限上下文，authority 只能是 "admin" 或 "user"
class Context(BaseModel):
    authority: Literal["admin", "user"]

# 创建带权限控制的工具
@tool
def math_add(runtime: ToolRuntime[Context, Any], a: int, b: int) -> int:
    """Add two numbers together."""
    authority = runtime.context.authority
    if authority != "admin":
        raise PermissionError("User does not have permission to add numbers")
    return a + b

# 创建 Agent
agent = create_agent(
    model=llm,
    tools=[math_add],
    system_prompt="You are a helpful assistant",
)

# 流式运行 Agent，每执行完一步就推送一个 chunk，不等全部完成
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "请计算 8234783 + 94123832 = ?"}]},
    # updates 模式：每一步有变化就推送一次，而不是推送完整状态
    stream_mode="updates",
    config={"configurable": {"thread_id": "1"}},
    context=Context(authority="admin"),
):
    # chunk 是每一步的数据，chunk.items() 拆出步骤名和步骤数据
    for step, data in chunk.items():
        # step 是当前步骤名：model（LLM思考）或 tools（工具执行）
        print(f"step: {step}")
        # 取这一步最后一条消息的内容块并打印
        print(f"content: {data['messages'][-1].content}")
