from mcp.server.fastmcp import FastMCP
# 创建一个 MCP Server 实例，名字叫 "weather"，配置监听本机 8000 端口。
# FastMCP 是 MCP 官方 SDK 提供的工具类，帮你快速搭一个 MCP Server，不用手写网络通信那些底层代码
mcp = FastMCP("weather", host="127.0.0.1", port=8000)

# 把这个函数注册成 MCP 工具。
@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # 模拟天气数据，实际项目里这里调真实天气 API
    weather_data = {
        "广州": "晴天，28°C，湿度70%",
        "北京": "多云，15°C，湿度45%",
        "上海": "小雨，18°C，湿度85%",
        "福州": "晴朗，25°C，湿度60%",
    }
    return weather_data.get(city, f"{city}天气晴朗，温度22°C。")


if __name__ == "__main__":
    mcp.run(transport="stdio")


# python -m mcp_server.get_weather_mcp → 运行的是 __main__.py


# # __main__.py
# if __name__ == "__main__":
#     http()   ← 走 HTTP
# python server.py → 运行的是 server.py


# # server.py
# if __name__ == "__main__":
#     mcp.run(transport="stdio")   ← 走 stdio
