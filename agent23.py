# 多智能体 - 监督者模式（langgraph-supervisor 包实现）
# 和 agent22 的区别：agent22 需要手动把子 Agent 包装成工具
# 这里用 create_supervisor 直接接收 Agent 作为参数，更简洁
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)

# 基础工具
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


# 三个子 Agent，各负责一种运算
subagent1 = create_agent(model=llm, tools=[add],      name="subagent-1")
subagent2 = create_agent(model=llm, tools=[multiply], name="subagent-2")
subagent3 = create_agent(model=llm, tools=[divide],   name="subagent-3")

# create_supervisor 直接接收 Agent 列表，不需要手动包装成工具
# 和 agent22 相比：agent22 要写 @tool + call_subagent 函数，这里直接传 Agent 进去
supervisor_graph = create_supervisor(
    [subagent1, subagent2, subagent3],
    model=llm,
    prompt="提示：如遇两数相减仍可用两数相加工具实现，只需将一个数加上另一个数的负数",
)

# compile() 把 supervisor_graph 编译成可运行的 app
supervisor_app = supervisor_graph.compile()

result = supervisor_app.invoke({
    "messages": [{"role": "user", "content": "计算 38462 + 378 / 49 * 83723 - 123 的结果"}]
})

for message in result["messages"]:
    message.pretty_print()
