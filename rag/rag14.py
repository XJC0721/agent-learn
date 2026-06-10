# Function Calling（工具调用）底层原理
# Agent 系列里用 @tool + LangGraph 封装好了，这里看 LangGraph 帮你做了什么
# 核心：两次 API 调用，第一次让模型决定调哪个工具，第二次把结果发回模型生成最终答案
import sys
import os
import json
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from openai import OpenAI

_ = load_dotenv()

# 直接用 OpenAI 客户端调 DeepSeek，不走 LangChain
# 这样能看到最原始的 API 交互
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ── 1. 定义工具（告诉模型有哪些工具可以用）─────────────────────────────
print("=" * 50)
print("1. 定义工具")
print("=" * 50)

# tools 是一个列表，每个工具是一个 JSON Schema
# 模型看到这个列表，才知道有哪些工具、每个工具需要哪些参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",           # 工具名，模型返回 tool_calls 时用这个名字
            "description": "查询指定城市的天气",  # 描述越清楚，模型越能正确选工具
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海",
                    }
                },
                "required": ["city"],        # 必填参数
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "搜索商品信息，返回名称和价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
]

print(f"已定义 {len(tools)} 个工具：get_weather, search_product")


# ── 2. 模拟工具实现（真实场景里这里是真实的 API 调用）────────────────
def get_weather(city: str) -> str:
    weather_data = {
        "北京": "晴天，24℃，东风3级",
        "上海": "多云，28℃，南风2级",
        "杭州": "小雨，22℃，东南风4级",
    }

# 冒号表示暂无数据
    return weather_data.get(city, f"{city}：暂无天气数据")


def search_product(keyword: str) -> str:
    products = {
        "键盘": "机械键盘Pro，青轴，399元",
        "耳机": "降噪耳机X1，蓝牙5.3，799元",
        "鼠标": "游戏鼠标G502，有线，299元",
    }
    for key, val in products.items():
        if key in keyword:
            return val
    return f"未找到与'{keyword}'相关的商品"


# ── 3. Function Calling 完整流程 ──────────────────────────────────────
def ask_with_tools(question: str):
    print(f"\n{'='*50}")
    print(f"用户问题：{question}")

    # ── 第一次调用：用户问题 + 工具列表 → 发给模型 ───────────────────
    # 模型看到工具列表后，如果需要工具才能回答，会返回 tool_calls 而不是直接回答
    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,           # 把工具列表传给模型
        temperature=0,
    )

    message = response.choices[0].message
    print(f"\n[第一次调用] 模型响应：")

    # ── 检查模型是否决定调用工具 ────────────────────────────────────
    if not message.tool_calls:
        # 模型觉得不需要工具，直接回答了
        print(f"  模型直接回答（无需工具）：{message.content}")
        return message.content

    # 模型返回了 tool_calls，说明它要调工具
    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name                        # 模型选了哪个工具
    tool_args = json.loads(tool_call.function.arguments)      # 模型给的参数
    print(f"  模型决定调用工具：{tool_name}")
    print(f"  工具参数：{tool_args}")

    # ── 代码层面实际执行工具 ─────────────────────────────────────────
    # 模型只是给了指令，真正执行是在代码里
    if tool_name == "get_weather":
        tool_result = get_weather(**tool_args)
    elif tool_name == "search_product":
        tool_result = search_product(**tool_args)
    else:
        tool_result = "工具不存在"

    print(f"  工具执行结果：{tool_result}")

    # ── 第二次调用：原始问题 + 工具结果 → 发给模型，获取最终回答 ────
    # 把整个对话历史（用户问题 + 模型的tool_call响应 + 工具执行结果）一起发给模型
    messages.append(message)   # 模型第一次的响应（包含 tool_calls）
    messages.append({
        "role": "tool",                    # role 必须是 "tool"
        "tool_call_id": tool_call.id,      # 对应哪一次工具调用
        "content": tool_result,            # 工具执行的结果
    })

    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        temperature=0,
    )

    final_answer = final_response.choices[0].message.content
    print(f"\n[第二次调用] 最终回答：{final_answer}")
    return final_answer


# ── 4. 测试 ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 测试")
print("=" * 50)

ask_with_tools("北京今天天气怎么样？")
ask_with_tools("帮我搜一下耳机")
ask_with_tools("1加1等于几")   # 不需要工具，模型直接回答


# ── 5. 对比 Agent 系列 ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("总结：和 Agent 系列的关系")
print("=" * 50)
print("agent 系列：@tool 装饰器 + LangGraph 自动完成上面所有步骤")
print("rag14     ：手写两次 API 调用，看清楚底层发生了什么")
print()
print("LangGraph 帮你封装的部分：")
print("  1. 把 @tool 函数自动转成 JSON Schema（tools 列表）")
print("  2. 检查 tool_calls，自动调用对应函数")
print("  3. 把结果打包成 role=tool 消息，自动发第二次请求")
