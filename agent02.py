# 使用 ToolRuntime 控制工具权限
import os
# 导入类型工具：Literal 限定值只能是固定几个，Any 表示任意类型。
from typing import Literal, Any
from dotenv import load_dotenv
# 导入 Pydantic 的基础类，用来定义数据结构（类似表单模板）。
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
# 定义一个叫 Context 的数据结构，里面有一个字段 authority，值只能是 "admin" 或 "user"，填其他值会报错。
class Context(BaseModel):
    authority: Literal["admin", "user"]

# 创建带权限控制的工具，依赖 ToolRuntime 的内容进行判断
@tool
# runtime: ToolRuntime[Context, Any] 中的 : ToolRuntime[Context, Any] 是类型注解，
# 就是告诉框架："这个参数是 ToolRuntime 类型，请自动注入
def math_add(runtime: ToolRuntime[Context, Any], a: int, b: int) -> int:
# 叫做 docstring（文档字符串），是函数的说明文字。
# LLM 靠这句话判断什么时候该用这个工具。
    """Add two numbers together."""
    # 从 runtime 中取出当前用户的身份
    authority = runtime.context.authority
    # 只有 admin 用户可以访问加法工具
    if authority != "admin":
        raise PermissionError("User does not have permission to add numbers")
    return a + b

# 创建带工具调用的 Agent
agent = create_agent(
    model=llm,
    tools=[math_add],
    system_prompt="You are a helpful assistant",
)

# 在运行 Agent 时注入 context（当前用户身份为 admin）
response = agent.invoke(
    {"messages": [{"role": "user", "content": "请计算 8234783 + 94123832 = ?"}]},
    # 对话会话的唯一标识，用来区分不同的对话。
    config={"configurable": {"thread_id": "1"}},
    context=Context(authority="admin"),
)

# 遍历所有消息并格式化打印
for message in response['messages']:
    # pretty_print() 是谁定义的
    # 是 LangChain 在每种消息类型（HumanMessage、AIMessage、ToolMessage）上内置的方法，
    # 专门用来格式化输出，会自动加上：
    # ====== Ai Message ====== 这种分隔线标题
    message.pretty_print()
