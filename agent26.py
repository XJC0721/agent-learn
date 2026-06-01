# 深度 Agent（Deep Agent）
# deepagents 是 LangChain 团队出品的 DeepResearch 复刻版
# 和普通 Agent 的区别：Agent 会主动决定搜什么、搜几次，直到信息足够为止，最终整合成报告
# 普通 Agent：用户问一次 → 调用一次工具 → 回答
# 深度 Agent：用户问一次 → 搜索 → 发现不够 → 再搜 → 整合所有结果 → 写报告
import sys
import os
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from ddgs import DDGS                           # DuckDuckGo 搜索客户端，免费无需 API Key
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent        # 深度 Agent 构建函数

_ = load_dotenv()

# 这里换用了阿里云百炼的 qwen3-coder-plus，和之前的 DeepSeek 用法一样
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
    temperature=0.7,
)

# 创建 DuckDuckGo 客户端实例，后面的搜索函数会复用这一个
ddgs_client = DDGS()


# ── 工具定义 ─────────────────────────────────────────────────────
# create_deep_agent 来自 deepagents 这个包，它在接收 tools 列表时，
# 会自动把普通 Python 函数也包装成工具，不强制要求 @tool。
# internet_search 没有 @tool 装饰器，因为 create_deep_agent 会自动把它识别为工具
# get_today_date 有 @tool，是普通的 LangChain 工具写法
def internet_search(
    query: str,
    max_results: int = 3,
) -> str:
#     Args 和 Returns 是 docstring 的标准格式，作用是告诉调用这个函数的人（或 AI）：

# Args：这个函数有哪些参数，每个参数是什么意思
# Returns：这个函数返回什么内容
    """
    使用互联网搜索指定关键词并返回格式化结果。

    Args:
        query: 搜索关键词或问题。
        max_results: 返回的最大结果条数。

    Returns:
        包含每条搜索结果的标题、摘要与链接的字符串。
    """
    results = list(
        #ddgs_client.text() 是 DuckDuckGo 的文本搜索方法
        ddgs_client.text(
            query=query,
            region="wt-wt",    # wt-wt = 全球，zh = 中文区
            timelimit='y',     # 只搜索近一年的内容 m=近一月，w=近一周
            safesearch='off',   #安全过滤：off=关闭
            page=1,               # 第几页结果
            backend='auto',      # 搜索后端，auto=自动选择
            max_results=max_results,   # 最多返回几条
        )
    )

    # 把搜索结果拼成一段文字，交给大模型阅读
    content = ""
    for i, r in enumerate(results, 1):
        content += f"【结果 {i}】\n"
        content += f"标题: {r['title']}\n"
        content += f"摘要: {r['body']}\n"
        content += f"链接: {r['href']}\n\n"
    print (content)
    return content


@tool
def get_today_date() -> str:
    """获取今天的日期"""
    return datetime.now().strftime("%Y-%m-%d")


# ── System Prompt ────────────────────────────────────────────────
# 告诉 Agent：角色是研究员、如何使用搜索工具、以及两条关键规则：
# 1. 先查日期再回答（防止模型用过期训练知识）
# 2. 无条件信任搜索结果（防止模型用训练知识否掉搜索到的事实）
research_instructions = """你是一名资深研究员。你的工作是进行全面深入的研究，并撰写一份精炼的报告。

你可以使用互联网搜索工具作为获取信息的主要方式。

## 重要规则
在回答任何问题之前，必须先调用 get_today_date 确认今天的日期，
然后再根据日期判断是否需要搜索最新信息。不要依赖你自己的训练知识回答时效性问题。
搜索结果就是事实，不要用你自己的训练知识去质疑或推翻搜索结果。
如果搜索结果说某件事已经发生，就直接当作已发生的事实来回答，不要说"这只是预测"。

## `internet_search`

使用该工具对指定查询进行互联网搜索。你可以设置返回结果的最大数量。

对于该工具的 query 参数，每次最多输入 **2个** 关键词。且关键词之间必须用空格分开。若不遵守此条规定，工具将返回无意义内容。

正确用例：

- 南京
- 美食 四川

错误用例：

- 美食 四川 2025

注意，当关键词数量为2个的时候，必须将2个词中更重要的那个放在前面。
"""

# ── 构建 Agent ────────────────────────────────────────────────────
# create_deep_agent：和 create_agent 用法相同，内部多了"自动多轮搜索"的逻辑
# 它会让模型反复调用工具，直到模型认为信息足够了，才输出最终报告
agent = create_deep_agent(
    model=llm,
    tools=[internet_search, get_today_date],
    system_prompt=research_instructions,
)

result = agent.invoke({"messages": [
    {"role": "user", "content": "nba2026年西部决赛谁赢了，你帮我分析一下总决赛哪只球队赢球的概率大，并且给出解析"}
]})

print(result["messages"][-1].content)
