from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math", host="127.0.0.1", port=8001)


@mcp.tool()
def calculate(expression: str) -> str:
    """Calculate a math expression. Example: '(3 + 5) * 12'"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算出错: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
