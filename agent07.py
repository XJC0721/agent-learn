# PII（个人隐私信息）检测与处理
# PII = Personally Identifiable Information，如姓名、身份证、电话、邮箱等
import os
from typing import Any
from textwrap import dedent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime
from langchain.agents import create_agent

load_dotenv()

# 主模型：处理用户正常请求
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# 可信模型：专门用来做 PII 检测（生产环境一般用本地模型，这里复用同一个）
trusted_model = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# 定义结构化输出格式：只有一个字段 is_pii
# 让 LLM 检测完之后返回 True 或 False，不返回自然语言
class PiiCheck(BaseModel):
    """Structured output indicating whether text contains PII."""
    is_pii: bool = Field(description="Whether the text contains PII")

# ── 创建 Agent 的工厂函数 ──────────────────────────────
# 接收一个中间件函数，返回一个配置好的 Agent
# 这样方便后面用不同中间件测试
def message_with_pii(pii_middleware):
    agent = create_agent(
        llm,
        middleware=[pii_middleware],
    )
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": dedent("""
                File "/home/zhangsan/proj/agent.py", line 53, in my_agent
                    agent = create_react_agent(
                            ^^^^^^^^^^^^^^^^
                AttributeError: 'RunnableLambda' object has no attribute 'bind_tools'
                ---
                为啥报错
            """).strip()
        }]
    })
    return result

# ── 处理方式一：发现隐私信息，直接拒绝回复 ────────────────

@before_agent(can_jump_to=["end"])
def content_blocker(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Deterministic guardrail: Block requests containing banned keywords."""
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]

    # 只检测用户发的消息
    if last_message.type != "human":
        return None

    content = last_message.content.lower()

    # 构建 PII 检测 prompt，让 trusted_model 判断内容是否包含隐私信息
    prompt = (
        "你是一个隐私保护助手。请识别下面文本中涉及个人可识别信息（PII），"
        "例如：姓名、身份证号、护照号、电话号码、邮箱、住址、银行卡号、社交账号、车牌等。"
        "特别注意，若代码、文件路径中包含用户名，也应被视为敏感信息。"
        "若包含敏感信息，请返回{\"is_pii\": True}，否则返回{\"is_pii\": False}。"
        "请严格以 json 格式返回，并且只输出 json。文本如下：\n\n" + content
    )

    # with_structured_output：让 LLM 按照 PiiCheck 的格式返回结果
    # 返回的不是字符串，而是 PiiCheck 对象，可以直接用 result.is_pii
    # 链式调用两个方法trusted_mode专门做安全检测的模型，和主模型分开，生产环境通常用本地模型避免数据泄露。
    # with_structured_output(PiiCheck)让 LLM 按照 PiiCheck 的格式返回，不返回自然语言，直接返回 True/False：
    pii_agent = trusted_model.with_structured_output(PiiCheck, method="json_mode")
    result = pii_agent.invoke(prompt)

    if result.is_pii is True:
        # 发现隐私信息 → 直接拒绝，跳到 end，不调用主 LLM
        return {
            "messages": [{
                "role": "assistant",
                "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
            }],
            "jump_to": "end"  # 直接结束，主 LLM 不会被调用
        }
    else:
        print("No PII found")

    return None

# ── 处理方式二：发现隐私信息，用 ***** 屏蔽后继续处理 ─────

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Deterministic guardrail: Mask PII before processing."""
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]

    if last_message.type != "human":
        return None

    content = last_message.content.lower()

    # 同样先检测是否有 PII
    prompt = (
        "你是一个隐私保护助手。请识别下面文本中涉及个人可识别信息（PII），"
        "例如：姓名、身份证号、护照号、电话号码、邮箱、住址、银行卡号、社交账号、车牌等。"
        "特别注意，若代码、文件路径中包含用户名，也应被视为敏感信息。"
        "若包含敏感信息，请返回{\"is_pii\": True}，否则返回{\"is_pii\": False}。"
        "请严格以 json 格式返回，并且只输出 json。文本如下：\n\n" + content
    )

    pii_agent = trusted_model.with_structured_output(PiiCheck, method="json_mode")
    result = pii_agent.invoke(prompt)

    if result.is_pii is True:
        # 发现隐私信息 → 不拒绝，而是让 LLM 把隐私信息替换成 * 号
        mask_prompt = (
            "你是一个隐私保护助手。请将下面文本中的所有个人可识别信息（PII）用星号（*）替换。"
            "仅替换敏感片段，其他文本保持不变。"
            "只输出处理后的文本，不要任何解释或额外内容。文本如下：\n\n" + last_message.content
        )
        # 用 trusted_model 把隐私内容打码
        masked_message = trusted_model.invoke(mask_prompt)

        # 把打码后的消息替换回去，继续交给主 LLM 处理
        return {
            "messages": [{
                "role": "assistant",
                "content": masked_message.content
            }]
        }
    else:
        print("No PII found")

    return None

# ── 测试两种处理方式 ──────────────────────────────────────

print("=== 处理方式一：发现 PII 直接拒绝 ===")
result = message_with_pii(pii_middleware=content_blocker)
for message in result["messages"]:
    message.pretty_print()

print("\n=== 处理方式二：发现 PII 打码后继续 ===")
result = message_with_pii(pii_middleware=content_filter)
for message in result["messages"]:
    message.pretty_print()
