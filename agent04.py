# StateGraph 状态图示例
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# StateGraph: 状态图核心类，用来构建节点和边的流程
# MessagesState: 内置的状态类，自带 messages 字段，存储所有对话消息
# START: 图的起点（虚拟节点）
# END: 图的终点（虚拟节点）
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
# ToolNode: 专门执行工具调用的节点，自动处理工具调用和结果返回
from langgraph.prebuilt import ToolNode
# RunnableConfig: 运行配置，包含 thread_id 等信息，节点函数可以通过它获取配置
from langchain_core.runnables import RunnableConfig

load_dotenv()

# 加载模型
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
    # temperature 控制输出的随机性：0=确定性强，1=更有创意
    temperature=0.7,
)

# 工具函数：对任何城市都返回晴天（简化版，实际项目会调用真实天气 API）
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 把工具放进列表，后面节点和 LLM 都会用到这个列表
tools = [get_weather]

# 创建工具节点：告诉图：工具节点里有这些工具，LLM 说要调用哪个，就真正去执行哪个。
tool_node = ToolNode(tools)

# 创建助手节点函数
# state: MessagesState → 当前图的状态，包含所有历史消息
# config: RunnableConfig → 当前运行配置（thread_id 等）
def assistant(state: MessagesState, config: RunnableConfig):
    system_prompt = 'You are a helpful assistant that can check weather.'
    # 把系统提示 + 历史消息拼在一起传给 LLM
    # SystemMessage 是系统级指令，放在消息列表最前面
    all_messages = [SystemMessage(system_prompt)] + state['messages']
    # bind_tools 告诉 LLM 它有哪些工具可以用
    model = llm.bind_tools(tools)
    # 调用 LLM，返回新消息，追加到 state['messages'] 中
    return {'messages': [model.invoke(all_messages)]}

# 创建条件边函数：决定助手节点执行完后下一步去哪
# 返回字符串，对应 add_conditional_edges 里的 key
def should_continue(state: MessagesState, config: RunnableConfig):
    messages = state['messages']
    last_message = messages[-1]
    # 如果最后一条消息有 tool_calls，说明 LLM 想调用工具，继续执行
    if last_message.tool_calls:
        return 'continue'
    # 否则 LLM 直接给出了最终回答，结束流程
    return 'end'

# ── 构建状态图 ──────────────────────────────────────

# 创建图，指定使用 MessagesState 作为状态类型
builder = StateGraph(MessagesState)

# 添加节点：第一个参数是节点名，第二个是对应的函数
builder.add_node('assistant', assistant)   # 助手节点：LLM 思考
builder.add_node('tool', tool_node)        # 工具节点：执行工具

# 添加普通边：START → assistant（图启动后第一个执行 assistant）
builder.add_edge(START, 'assistant')

# 添加条件边：assistant 执行完后，根据 should_continue 的返回值决定下一步
# 'continue' → 去 tool 节点
# 'end'      → 去 END（结束）
builder.add_conditional_edges(
    'assistant',        # 从哪个节点出发
    should_continue,    # 判断函数
    {
        'continue': 'tool',  # should_continue 返回 'continue' → 去 tool
        'end': END,          # should_continue 返回 'end' → 结束
    },
)

# 添加普通边：tool → assistant（工具执行完后回到 assistant 继续思考）
builder.add_edge('tool', 'assistant')

# 编译图：把所有节点和边组装成可运行的图
my_graph = builder.compile(name='my-graph')

# ── 运行图 ──────────────────────────────────────────

response = my_graph.invoke(
    {'messages': [HumanMessage('What is the weather in Beijing?')]},
    config={'configurable': {'thread_id': '1'}},
)

# 打印所有消息
for message in response['messages']:
    message.pretty_print()
