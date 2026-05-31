import asyncio
import os
from . import server

host = os.getenv('HOST', '127.0.0.1')
port = int(os.getenv('PORT', 8001))


def stdio():
    asyncio.run(server.mcp.run(transport="stdio"))


def http():
    """streamable-http entry point for the package."""
    asyncio.run(server.mcp.run(transport="streamable-http"))


if __name__ == "__main__":
    http()
