# 多智能体 - 监督者模式（tool-calling 实现）
# Supervisor 把任务拆分交给子 Agent 或工具执行，再汇总结果
# 这里用 tool-calling 方式：把子 Agent 包装成工具，传给 supervisor_agent
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)

# ── 基础工具：三个数学运算函数 ──────────────────────────────────────
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


# ── 子 Agent：每个只负责一种运算 ──────────────────────────────────
# subagent1 专门做加法
subagent1 = create_agent(
    model=llm,
    tools=[add],
    name="subagent-1",
)

# 把 subagent1 包装成工具，Supervisor 调用这个工具就等于调用 subagent1
@tool(
    "subagent-1",
    description="可以准确地计算两数相加"
)
def call_subagent1(query: str) -> str:
    result = subagent1.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    # 只返回最后一条消息（Agent 的最终回答）
    return result["messages"][-1].content


# subagent2 专门做乘法
subagent2 = create_agent(
    model=llm,
    tools=[multiply],
    name="subagent-2",
)

# 和之前学的 @tool 对比：


# # 之前：不传参数，工具名 = 函数名，描述 = docstring
# @tool
# def get_weather(city: str) -> str:
#     """Get weather for a city."""   ← 描述写在这里

# # 现在：直接传参数，名字和描述都写在装饰器括号里
# @tool("subagent-2", description="可以准确地计算两数相乘")
# def call_subagent2(query: str) -> str:


@tool(
    "subagent-2",
    description="可以准确地计算两数相乘"
)
def call_subagent2(query: str) -> str:
    result = subagent2.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


# ── Supervisor：统筹调度，把任务分给合适的子 Agent 或工具 ──────────
# tools 里有：两个子 Agent（加法/乘法） + 一个直接工具（除法）
# system_prompt 提示：没有减法工具，遇到减法用"加负数"来实现
supervisor_agent = create_agent(
    model=llm,
    tools=[call_subagent1, call_subagent2, divide],
    name="supervisor-agent",
    system_prompt="提示：如遇两数相减仍可用两数相加工具实现，只需将一个数加上另一个数的负数",
)

# 测试：复杂表达式，看 Supervisor 能否正确调用所有工具
result = supervisor_agent.invoke({
    "messages": [{"role": "user", "content": "计算 38462 + 378 / 49 * 83723 - 123 的结果"}]
})

for message in result["messages"]:
    message.pretty_print()
