import asyncio
import os
from . import server
# __main__.py     ← 入口：用 python -m 启动时执行这里
host = os.getenv('HOST', '127.0.0.1')
port = int(os.getenv('PORT', 8000))

# 用 stdio 方式启动 Server——通过 stdin/stdout 通信，不开 HTTP 端口。给 Agent 用 stdio 连接时调这个。
def stdio():
    asyncio.run(server.mcp.run(transport="stdio"))

# 用 HTTP 方式启动 Server——开一个 HTTP 端口（8000），等待网络请求进来。
# python -m mcp_server.get_weather_mcp 执行的就是这个函数。python -m 启动时默认走 HTTP 模式。
def http():
    """streamable-http entry point for the package."""
    asyncio.run(server.mcp.run(transport="streamable-http"))


if __name__ == "__main__":
    http()
