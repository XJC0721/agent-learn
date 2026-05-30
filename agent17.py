# 上下文工程 - 动态修改消息列表（@wrap_model_call）
# 和 agent14-16 的区别：之前用 @dynamic_prompt 只能改 system prompt
# 这里用 @wrap_model_call 可以对整个消息列表做任意操作
# 演示：把本地文件内容注入消息列表，让模型参考文件内容回答问题
import os
from typing import Callable
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from dataclasses import dataclass

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
)


# Context 里存用户上传的文件列表，每个文件是一个字典 {"path": "文件路径"}，外层是列表，里面每个元素是字典
@dataclass
class FileContext:
    uploaded_files: list[dict]


# @wrap_model_call 比 @dynamic_prompt 更强大，可以修改整个消息列表
# 函数必须接收两个参数：request（当前请求）和 handler（实际调用模型的函数）
# 必须返回 handler(request)，即把处理后的 request 真正发给模型
@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject context about files user has uploaded this session."""
    # 取的是相对路径
    uploaded_files = request.runtime.context.uploaded_files

    # 获取当前脚本所在目录，用于解析相对路径
#     __file__ → 当前 Python 文件的路径，比如 d:/code/trycc/agent17.py
# os.path.abspath() → 转成绝对路径
# os.path.dirname() → 取目录部分，去掉文件名

    base_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = "d:/code/trycc"              # 当前脚本所在目录
# path     = "./docs/rule_horror.md"      # 你传进来的文件路径（相对路径）

    file_sections = []
    for file in uploaded_files:
# file 是列表里的一个字典：
# file = {"path": "./docs/rule_horror.md"}
# file.get("path") 就是取出：
# path = "./docs/rule_horror.md"        
        path = file.get("path")
        if path:
            # os.path.basename() → 只取文件名部分，去掉目录
            # "./docs/rule_horror.md" → "rule_horror.md"
            base_filename = os.path.basename(path)
            # 把文件名拆成名称和扩展名两部分，同时赋值给两个变量
            # "rule_horror.md" → stem="rule_horror", ext=".md"
            stem, ext = os.path.splitext(base_filename)
            # 三元表达式：如果 ext 有值就去掉点 → ".md" 变 "md"，否则返回 None
            # str.lstrip(".") → 去掉字符串左边的点
            name = stem or base_filename
            ftype = ext.lstrip(".") if ext else None

            # 构建文件描述内容
            content_list = [f"名称: {name}"]
            if ftype:
                content_list.append(f"类型: {ftype}")

            # 解析相对路径为绝对路径
            abs_path = path if os.path.isabs(path) else os.path.join(base_dir, path)

            # 读取文件内容
#os.path.exists() → 确认文件真实存在
# open(..., "r") → 只读方式打开文件
# with...as f → 打开后自动关闭，f 是文件对象
# f.read() → 读取全部内容存进 content_block
            if abs_path and os.path.exists(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        content_block = f.read()
                except Exception as e:
                    content_block = f"[读取文件错误 '{abs_path}': {e}]"
            else:
                content_block = "[文件路径缺失或未找到]"

            section = (
                f"---\n"
                f"{chr(10).join(content_list)}\n\n"
                f"{content_block}\n"
                f"---"
            )
            file_sections.append(section)

    file_context = (
        "已加载的会话文件：\n"
        f"{chr(10).join(file_sections)}"
        "\n回答问题时请参考这些文件。"
    )

    # 把文件内容插入消息列表，追加在原有消息后面
    # request.override() 创建一个新的 request，不修改原始 request
    messages = [
        *request.messages,
        {"role": "user", "content": file_context},
    ]
    request = request.override(messages=messages)

    # 必须调用 handler(request) 把处理后的请求真正发给模型
    return handler(request)


agent = create_agent(
    model=llm,
    middleware=[inject_file_context],
    context_schema=FileContext,
)

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "关于上海地铁的无脸乘客，有什么需要注意的？",
        }],
    },
    # 传入要注入的文件路径列表
    context=FileContext(uploaded_files=[{"path": "./docs/rule_horror.md"}]),
)

for message in result["messages"]:
    message.pretty_print()
