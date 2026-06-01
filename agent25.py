# 并行 - Map-reduce
# 场景：10种类型的男生同时并行回复女神不回消息，最后选出最佳回复
# Map：把角色列表拆分，每个角色并行发给 generate_response 节点
# Reduce：best_response 节点汇总所有回复，选出最佳

# agent25 完整运行流程：

# 第1步：invoke 传入角色列表


# my_graph.invoke({"roles": ["男神", "暖男", "海王", ...]})
# 第2步：START → continue_to_responses（Map 分发）


# return [Send("generate_response", {"role": "男神"}),
#         Send("generate_response", {"role": "暖男"}),
#         Send("generate_response", {"role": "海王"}),
#         ...]   # 10个 Send，触发10个并行任务
# 第3步：10个 generate_response 并行跑


# [男神节点]    [暖男节点]    [海王节点] ... （同时运行）
#     ↓              ↓             ↓
# 返回各自的 response，自动追加到 Overall.responses 列表里
# （Annotated[list, operator.add] 负责合并，不会覆盖）
# 第4步：全部完成后 → best_response（Reduce 汇总）


# responses = "\n\n".join(state["responses"])  # 把10条回复拼在一起
# # 发给 LLM：这10条哪个最好？返回 ID
# response = llm.with_structured_output(BestResponse).invoke(prompt)
# return {"best_response": state["responses"][response.id]}
# 第5步：END，返回最终结果

import sys
import os
import operator
sys.stdout.reconfigure(encoding='utf-8')
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)

# ── Schema 定义 ──────────────────────────────────────────────────
class Roles(BaseModel):
    roles: list[str]

class Role(BaseModel):          # 单个角色的 state，只给 generate_response 用
    role: str

class Response(BaseModel):      # LLM 返回的单条回复
    response: str

class BestResponse(BaseModel):  # LLM 返回的最佳回复 ID
    id: int

# 全局 state：贯穿整个图
# Annotated[list, operator.add] 表示多个并行节点往 responses 写入时，自动合并（追加）而不是覆盖
# TypedDict 是 Python 的一种特殊字典，让你可以给字典的每个字段定义类型：
# 这是 LangGraph 的核心机制：所有节点共享同一个全局 state（Overall），节点的返回值自动合并进去。
class Overall(TypedDict):
    roles: list[str]
    responses: Annotated[list, operator.add]
    best_response: str


# ── 提示词 ──────────────────────────────────────────────────────
role_prompt = "女神又不回你消息了，作为一个{role}，你应该如何一句话回复女神？请以JSON格式返回，包含response字段"

best_response_prompt = """下面是几种类型的男生，面对女神不回消息的情况，做出的反应。
你觉得以下哪种回复最能挽回女神的心，请返回对应的ID。
注意哦，第一条反应对应的是0号ID。并以JSON格式返回，包含id字段
下面是男生们的反应：\n\n{responses}"""


# ── 节点 ─────────────────────────────────────────────────────────
# Map 节点：把 roles 列表里的每个角色，用 Send 分发给 generate_response 节点并行执行
# Send 是 LangGraph 提供的特殊函数，作用是"派发一个任务到指定节点"：
# Send(
#     "generate_response",   # 派发到哪个节点
#     {"role": r}            # 给那个节点的 state（就是 Role 的数据）
# )
# 每调用一次 Send，就相当于启动一个 generate_response 节点的实例，传入 {"role": r} 作为它的 state。
def continue_to_responses(state: Overall):
    return [Send("generate_response", {"role": r}) for r in state["roles"]]

# 并行节点：每个角色独立运行，state schema 是 Role（不是 Overall）
def generate_response(state: Role):
    filled_prompt = role_prompt.format(role=state["role"])
    # llm_reply 是 LLM 返回的 Response 对象，取 .response 字段拿到回复字符串
    llm_reply = llm.with_structured_output(Response, method="json_mode").invoke(filled_prompt)
    return {"responses": [llm_reply.response]}

# Reduce 节点：收集所有并行结果，选出最佳回复
def best_response(state: Overall):
    all_responses = "\n\n".join(state["responses"])
    filled_prompt = best_response_prompt.format(responses=all_responses)
    # best 是 BestResponse 对象，取 .id 拿到最佳回复的下标
    best = llm.with_structured_output(BestResponse, method="json_mode").invoke(filled_prompt)
    # LangGraph 节点 return： 返回值是"我要更新哪些字段"，没提到的字段保持原样
    return {"best_response": state["responses"][best.id]}


# ── 构建图 ───────────────────────────────────────────────────────
builder = StateGraph(Overall)

builder.add_node("generate_response", generate_response)
builder.add_node("best_response", best_response)

# continue_to_responses 用 Send 动态分发，有几个角色就并行跑几个 generate_response，用这个函数决定去哪里
# continue_to_responses 返回的是动态的 Send 列表，
# LangGraph 不知道它会跳到哪些节点，所以需要你提前声明"这个函数可能会跳到这些节点"：
# ["generate_response"]   # 告诉 LangGraph：目标只可能是 generate_response
builder.add_conditional_edges(START, continue_to_responses, ["generate_response"])
builder.add_edge("generate_response", "best_response")
builder.add_edge("best_response", END)


my_graph = builder.compile(name='best-response')


# ── 调用 ─────────────────────────────────────────────────────────
roles = ["男神", "暖男", "海王", "痴情男", "决绝的男生", "茶茶的男生",
         "理想主义的男生", "喜欢嘶嗫的男生", "大男子主义的男生", "二次元肥宅"]

# invoke() 返回的是整个 Overall state，不是最后节点的返回值。
response = my_graph.invoke({"roles": roles})

# 打印每个角色的回复，zip把两个列表"拉链式"配对，每次循环同时取两个列表对应位置的元素
for role, resp in zip(roles, response["responses"]):
    print(f"【{role}】")
    print(resp)
    print()

# 打印最佳回复
print("=" * 40)
print("最佳回复：")
print(response["best_response"])
