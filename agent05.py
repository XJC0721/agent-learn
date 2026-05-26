# 消息截断：当对话历史过长时，自动裁剪消息，避免超出上下文长度限制
import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
# RemoveMessage: 用于删除指定消息
from langchain.messages import RemoveMessage
# REMOVE_ALL_MESSAGES: 特殊标记，表示删除所有消息
from langgraph.graph.message import REMOVE_ALL_MESSAGES
# InMemorySaver: 内存检查点，让 Agent 能记住多轮对话历史
from langgraph.checkpoint.memory import InMemorySaver
# AgentState: Agent 的状态类型
# create_agent: 创建 Agent 的函数
from langchain.agents import create_agent, AgentState
# before_model: 装饰器，在每次 LLM 调用之前执行被装饰的函数
from langchain.agents.middleware import before_model
# Runtime: 运行时上下文
from langgraph.runtime import Runtime

load_dotenv()

# 配置大模型
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# ── 截断策略一：保留第一条 + 最近几条 ──────────────────

# @before_model: 在每次调用 LLM 之前先执行这个函数
# state: AgentState → 当前完整状态，包含所有历史消息
# runtime: Runtime  → 运行时上下文
# 返回 dict 表示修改消息，返回 None 表示不修改
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    # 消息数量 <= 3 条时不需要截断，直接返回 None 表示不修改
    if len(messages) <= 3:
        return None

    # 保留第一条消息（通常包含用户的重要信息，比如"我是bob"）
    first_msg = messages[0]

    # 根据消息数量奇偶性决定保留最近几条
    # 目的是保证消息对齐（每次对话是一问一答，成对出现）
    # # 消息总数是偶数 → 取最近3条
    # # 消息总数是奇数 → 取最近4条
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]

    # 拼接：第一条 + 最近几条
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            # 先清空所有消息
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            # 再放入新的消息列表（* 是解包，把列表里的元素一个个放进去）
            *new_messages
        ]
    }

# ── 截断策略二：只保留最近两条（不保留第一条）──────────

@before_model
def trim_without_first_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    return {
        "messages": [
            # 清空所有消息
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            # 只保留最近两条，Agent 会忘记更早的对话内容
            *messages[-2:]
        ]
    }

# ── 创建 Agent ──────────────────────────────────────────

agent = create_agent(
    llm,
    # 使用策略一：保留第一条 + 最近几条
    middleware=[trim_without_first_message],
    # checkpointer: 检查点，让 Agent 能跨多次 invoke 记住对话历史
    # InMemorySaver 把历史存在内存里（程序关闭就消失）
    # 每次 invoke 都是一次独立的对话轮次，因为有 checkpointer + thread_id，Agent 
    # 能把这4次 invoke 的历史串联起来，记住之前说过的内容。
    checkpointer=InMemorySaver(),
)

# config 放在外面，多次 invoke 共用同一个 thread_id，才能记住历史
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# ── 多轮对话测试 ────────────────────────────────────────

def agent_invoke(agent):
    # 第一轮：告诉 Agent 我的名字
    agent.invoke({"messages": "hi, my name is bob"}, config)
    # 第二轮：让 Agent 写一首关于猫的诗
    agent.invoke({"messages": "write a short poem about cats"}, config)
    # 第三轮：让 Agent 写一首关于狗的诗
    agent.invoke({"messages": "now do the same but for dogs"}, config)
    # 第四轮：问 Agent 我叫什么名字
    final_response = agent.invoke({"messages": "what's my name?"}, config)

    # 打印最终回答，测试 Agent 是否还记得名字
    final_response["messages"][-1].pretty_print()

agent_invoke(agent)
