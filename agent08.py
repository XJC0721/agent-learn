# 人机交互（Human-in-the-Loop, HITL）
# Agent 执行工具前先暂停，等人类审批，审批通过后再继续
import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
# HumanInTheLoopMiddleware: 内置中间件，在工具调用前触发人工审批流程
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
# InMemorySaver: 内存检查点，暂停时把状态存在这里，恢复时再取出来
from langgraph.checkpoint.memory import InMemorySaver
# Command: 用来向暂停中的 Agent 发送恢复指令（approve/reject/edit）
from langgraph.types import Command

_ = load_dotenv()

# 配置大模型
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# ── 定义三个工具 ──────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters."""
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("height_m and weight_kg must be greater than 0.")
    return weight_kg / (height_m ** 2)

# ── 创建带人机交互中间件的 Agent ──────────────────────────

tool_agent = create_agent(
    model=llm,
    tools=[get_weather, add_numbers, calculate_bmi],
    middleware=[
        HumanInTheLoopMiddleware(
            # interrupt_on: 配置每个工具是否需要人工审批
            interrupt_on={
                # False = 直接执行，不需要人审批
                "get_weather": False,
                # True = 需要审批，支持 approve / edit / reject 三种操作
                "add_numbers": True,
                # 字典格式 = 需要审批，但只允许 approve 和 reject 两种操作
                "calculate_bmi": {"allowed_decisions": ["approve", "reject"]},
            },
            # description_prefix: 审批提示信息的前缀，显示在暂停通知里
            description_prefix="Tool execution pending approval",
        ),
    ],
    # checkpointer: 暂停时把当前状态存到这里，恢复时再读出来继续执行
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant",
)

# ── 第一步：发起请求，Agent 会在 calculate_bmi 前暂停 ─────

# uuid.uuid4() 生成一个随机唯一 ID 作为 thread_id
# 每次对话用同一个 thread_id，Agent 才能认出是"同一个对话"，从暂停点恢复
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = tool_agent.invoke(
    {"messages": [{
        "role": "user",
        # "content": "what is the weather in sf"  # 这条不会触发审批（get_weather=False）
        "content": "我身高180cm，体重180斤，我的BMI是多少",  # 会触发 calculate_bmi 审批
    }]},
    config=config,
)

# __interrupt__ 是 Agent 暂停时放入 result 的特殊字段
# 里面包含：待审批的工具名、参数、允许的审批类型
interrupt_info = result.get("__interrupt__")
print("=== Agent 已暂停，等待审批 ===")
print(interrupt_info)

# ── 第二步：审批通过，恢复执行 ───────────────────────────

print("\n=== 发送审批通过，恢复执行 ===")
result = tool_agent.invoke(
    # Command: 向暂停中的 Agent 发送恢复指令
    # resume: 审批决定列表，type 可以是 "approve" / "reject" / "edit"
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,  # 必须用同一个 config（同一个 thread_id），Agent 才知道恢复哪个对话
)

print(result["messages"][-1].content)
