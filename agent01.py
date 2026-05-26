# 创建一个简单的agent
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

# 配置大模型服务
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)

# 创建一个简单的Agent
agent = create_react_agent(
    model=llm,
    tools=[],  
    prompt="You are a helpful assistant",
)

# 运行Agent
response = agent.invoke({'messages': '你好'})

print(response['messages'][-1].content)