# MCP 客户端 - 使用 MultiServerMCPClient 连接 MCP Server
# 同时连接两个 MCP Server：
#   - get_weather_mcp：HTTP 方式（需要提前启动）
#   - math_mcp：stdio 方式（自动启动子进程，不需要提前启动）

#use_mcp(messages)
#     ↓
# mcp_agent()        → 连接两个 MCP Server，拿到工具，创建 Agent
#     ↓
# agent.ainvoke()    → 把用户问题发给 Agent
#     ↓
# Agent 判断用哪个工具 → 调 MCP Server → 拿结果 → 回答用户
#     ↓
# return response
import sys
import asyncio
import os
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)

# async def连接 MCP Server、获取工具都是网络操作，需要等待。
# Python 用 async/await 处理这种"需要等待"的操作，不会卡住程序。
async def mcp_agent():
    # MultiServerMCPClient 可以同时连接多个 MCP Server
    client = MultiServerMCPClient(
        {
            # math MCP：stdio 方式，Client 直接启动 server.py 子进程，通过 stdin/stdout 通信
            # 不需要提前启动，Client 会自动管理这个子进程
            "math": {
                "command": "python",
                "args": [os.path.abspath("./mcp_server/math_mcp/server.py")],
                "transport": "stdio",
            },
            # weather MCP：HTTP 方式，连接已经在 8000 端口运行的 MCP Server
            # 需要提前用 python -m mcp_server.get_weather_mcp 启动
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    # 从两个 MCP Server 获取所有工具，装进 Agent
    tools = await client.get_tools()
    agent = create_agent(llm, tools=tools)
    return agent
# 同步：一个等一个完成，异步：同时进行

async def use_mcp(messages):
    # await 是因为连接 Server 需要等待（网络操作
    agent = await mcp_agent()
# 把消息发给 Agent 让它处理，返回结果。ainvoke 是异步版本的 invoke
# ——之前用的 agent.invoke() 是同步的，这里因为整个函数是 async，所以要用 ainvoke。
    response = await agent.ainvoke(messages)
    return response


async def main():
    # 调用天气 MCP（需要先启动 weather server）
    messages = {"messages": [{"role": "user", "content": "福州天气怎么样？"}]}
    response = await use_mcp(messages)
    print(response["messages"][-1].content)

    # 调用算数 MCP（stdio，自动启动）
    messages = {"messages": [{"role": "user", "content": "计算 (3 + 5) * 12"}]}
    response = await use_mcp(messages)
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
