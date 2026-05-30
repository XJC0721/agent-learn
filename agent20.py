# 压缩上下文 - SummarizationMiddleware
# 对话历史太长时，自动把旧消息压缩成摘要，节省 token，避免上下文腐坏
# 和 agent14 的区别：agent14 是动态修改 system prompt，这里是压缩消息列表本身
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)

# 短期记忆：记住对话历史，压缩摘要需要依赖它才能跨轮保存
checkpointer = InMemorySaver()

# SummarizationMiddleware 参数：
# model                    → 用哪个模型来生成摘要
# max_tokens_before_summary → token 超过这个数才触发压缩（这里设 40 是为了让例子里能触发）
# messages_to_keep          → 压缩后保留最近几条原始消息（其余变成摘要）
# agent20 和 agent14 相比，唯一的区别就是 middleware 里换成了 SummarizationMiddleware，其他结构完全一样
agent = create_agent(
    model=llm,
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=('tokens', 40),    # token 超过 40 触发压缩（新写法）
            keep=('messages', 1),      # 压缩后保留最近 1 条原始消息（新写法）
        ),
    ],
)

# 传入一段多轮对话历史，token 很快超过 40，触发压缩
# checkpointer 在 invoke 时传入，让压缩后的摘要能被保存下来
# 如果没有 checkpointer，这个压缩后的状态只存在于本次 invoke 的内存里，调用结束就消失了。
# 下次再调用还是从原始的6条消息开始，重新压缩一遍，白白浪费。
# 有了 checkpointer，压缩后的状态会被保存下来：
result = agent.invoke(
    {"messages": [
        {"role": "user",      "content": "广州今天的天气怎么样？"},
        {"role": "assistant", "content": "广州天气很好"},
        {"role": "user",      "content": "吃点什么好呢"},
        {"role": "assistant", "content": "要不要吃香茅鳗鱼煲"},
        {"role": "user",      "content": "香茅是什么"},
        {"role": "assistant", "content": "香茅又名柠檬草，常见于泰式冬阴功汤、越南烤肉"},
        {"role": "user",      "content": "auv 那还等什么，咱吃去吧"},
    ]},
    checkpointer=checkpointer,
)

for message in result["messages"]:
    message.pretty_print()

# pretty_print() 输出的顺序就是：


# 第1条 → 摘要（原来6条压缩成的）
# 第2条 → Human: auv 那还等什么...（keep=1 保留的最后1条）
# 第3条 → AI 的回答