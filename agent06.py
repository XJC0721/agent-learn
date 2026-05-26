# 内容过滤（Guardrail）：在请求到达 LLM 之前拦截违禁内容
import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# before_agent: 在 Agent 处理之前触发的装饰器（比 before_model 更早）
# AgentState: Agent 的状态类型
from langchain.agents.middleware import before_agent, AgentState
# Runtime: 运行时上下文
from langgraph.runtime import Runtime
from langchain.agents import create_agent

load_dotenv()

# 配置大模型
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# 违禁词列表
banned_keywords = ["hack", "exploit", "malware"]

# @before_agent: 在 Agent 处理之前触发
# can_jump_to=["end"]: 声明这个中间件有权限直接跳到 "end" 节点，跳过 LLM 调用
@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Deterministic guardrail: Block requests containing banned keywords."""
    # 消息列表为空，不需要过滤
    if not state["messages"]:
        return None

    # 取最后一条消息
    last_message = state["messages"][-1]

    # 只检查用户发的消息，AI 的消息不需要过滤
    if last_message.type != "human":
        return None

    # 转成小写，让检测不区分大小写（Hack 和 hack 都能检测到）
    content = last_message.content.lower()

    # 遍历违禁词列表，检查消息里是否包含违禁词
    for keyword in banned_keywords:
        if keyword in content:
            # 发现违禁词：直接返回拒绝消息，不调用 LLM
            return {
                # 追加一条 AI 拒绝消息
                "messages": [{
                    "role": "assistant",
                    "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                }],
                # 跳到 end，整个流程直接结束，LLM 不会被调用
                "jump_to": "end"
            }

    # 没有违禁词，返回 None 表示不拦截，正常继续
    return None

# 创建带内容过滤中间件的 Agent
agent = create_agent(
    llm,
    middleware=[content_filter],
)

# 发送一条包含违禁词的消息，测试过滤是否生效
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})

# 打印所有消息，验证 LLM 没有被调用，直接返回了拒绝消息
for message in result["messages"]:
    message.pretty_print()
