# 上下文工程 - 动态修改系统提示词（使用 State 管理上下文）
# 根据对话轮数动态调整 system prompt，对话越长越简洁，节省 token
# @dynamic_prompt 是 LangGraph 专门用来动态修改系统提示词的中间件
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)


# @dynamic_prompt 把这个函数变成动态系统提示词中间件
# 每次调用模型前都会执行，根据当前 State 返回不同的 system prompt
@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    # request.messages 是 request.state["messages"] 的快捷访问方式
    # 统计当前对话里有多少条消息
    message_count = len(request.messages)

    base = "You are a helpful assistant."

    # 消息超过6条说明对话很长，提示模型回答更简洁，避免 token 浪费
    if message_count > 6:
        base += "\nThis is a long conversation - be extra concise."

    # 临时打印看效果
    print(base)
#  @dynamic_prompt 这个装饰器负责把它自动注入成 system prompt 发给模型。
# @dynamic_prompt 做的事就是：
# 把 base 字符串包装成 {"role": "system", "content": base} 插到消息列表最前面，你不用手动写这步。
    return base


agent = create_agent(
    model=llm,
    # 传入动态提示词中间件，每次调用模型前自动执行
    middleware=[state_aware_prompt]
)

result = agent.invoke(
    # 插到最前面 
    # {"role": "system", "content":
    #  "You are a helpful assistant.\nThis is a long conversation - be extra concise."},
    {"messages": [
        {"role": "user", "content": "广州今天的天气怎么样？"},
        {"role": "assistant", "content": "广州天气很好"},
        {"role": "user", "content": "吃点什么好呢"},
        {"role": "assistant", "content": "要不要吃香茅鳗鱼煲"},
        {"role": "user", "content": "香茅是什么"},
        {"role": "assistant", "content": "香茅又名柠檬草，常见于泰式冬阴功汤、越南烤肉"},
        {"role": "user", "content": "auv 那还等什么，咱吃去吧"},
    ]},
)

for message in result["messages"]:
    message.pretty_print()
