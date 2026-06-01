import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
)

agent = create_agent(model=llm)

# langgraph-cli 启动时会调用这个函数，拿到 agent 实例
def get_app():
    return agent
