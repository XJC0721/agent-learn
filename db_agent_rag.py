# 向量库知识智能体（DB Agent RAG）
# RAG = Retrieval Augmented Generation：先从知识库检索相关内容，再结合检索结果回答问题
# 流程：txt文件 → 按空行切分 → 向量化 → 存入Chroma → Agent检索回答
import sys
import os
import re
from pathlib import Path
from typing import Iterable
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.tools import tool

_ = load_dotenv()

# ── 配置大模型与客户端 ────────────────────────────────────────────
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
    temperature=0,
)

# OpenAI 客户端用于调用 Embeddings 接口（向量化文本）
# 和 llm 用的是同一套 DashScope API，只是接口不同
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)


# ── 自定义 Embeddings 类 ──────────────────────────────────────────
# LangChain 的 Chroma 需要一个实现了 Embeddings 接口的对象
# DashScope 没有现成的 LangChain 封装，所以手动写一个适配类
# Chroma 是 LangChain 生态里的组件，它规定：
# 传进来的 Embeddings 对象必须有 embed_documents 和 embed_query 这两个方法，继承自 Embeddings 基类。
# DashScope 没有官方的 LangChain 封装，所以必须手动写这个适配类，让它"符合 Chroma 的要求"。
# embed_documents：批量向量化（存入数据库时用）
# embed_query：单条向量化（查询时用）
class DashScopeEmbeddings(Embeddings):
    """DashScope 兼容的 Embeddings 封装。"""

    def __init__(self, model: str = "text-embedding-v4", dimensions: int = 1024):
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # DashScope 每次最多接受 10 条，所以分批处理
        for i in range(0, len(texts), 10):
            # texts[start : end]   # 取从 start 到 end（不含end）的元素
            chunk = texts[i: i + 10]
            # 向量化
            response = client.embeddings.create(
                model=self.model,
                input=chunk,
                dimensions=self.dimensions,
            )
# append([1,2,3]) → 把整个列表作为一个元素加进去
# extend([1,2,3]) → 把列表里的每个元素逐个加进去
            vectors.extend([item.embedding for item in response.data])
        return vectors
# Python 类里的方法第一个参数必须是 self
    def embed_query(self, text: str) -> list[float]:
        response = client.embeddings.create(
            model=self.model,
            # API 要求 input 必须是列表格式。
            input=[text],
            dimensions=self.dimensions,
        )
        return response.data[0].embedding


# ── 文档加载逻辑 ──────────────────────────────────────────────────
# 读取 files/ 目录下所有 txt 文件，按空行切分成多个 Document
# 每个 Document 是一个独立的知识片段，单独存入向量库
def load_txt_documents(data_dir: Path) -> list[Document]:
    """读取目录下的 txt 文件并按空行分割为 Document。"""

    # 用空行（含空白字符的行）切分文本，yield 非空片段
    def split_on_blank(text: str) -> Iterable[str]:
        # re.split(pattern, text) 的作用： 按照 pattern 匹配到的位置切分 text，返回切好的列表。
        for block in re.split(r"\n\s*\n", text):
            # block.strip() 去掉首尾空白
            cleaned = block.strip()
            if cleaned:
                yield cleaned

    documents: list[Document] = []

    if not data_dir.exists():
        print(f"Warning: 目录 {data_dir} 不存在")
        return []
# glob("*.txt") 找出 files/ 下所有 .txt 文件，sorted 按文件名排序，path 每次循环是一个文件路径。
    for path in sorted(data_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        for idx, part in enumerate(split_on_blank(content)):
            documents.append(
                Document(
                    page_content=part,
                    metadata={"source": path.name, "chunk_id": idx},
                )
            )

    if not documents:
        print(f"目录 {data_dir} 下未找到 txt 文档")

    return documents


# ── 构建向量数据库 ────────────────────────────────────────────────
# 读取 txt 文件 → 向量化 → 存入本地 Chroma 数据库
# persist_directory="./chroma_db"：数据持久化到本地，程序重启后数据还在
# （原版连接远程 Chroma 服务器，这里改为本地模式，效果相同）
def build_vector_store(data_dir: Path | None = None) -> Chroma:
    """读取 txt 文件并构建本地向量库。"""
    # Path.cwd() 是当前工作目录，/ "files" 是路径拼接（不是除法）
    target_dir = data_dir or (Path.cwd() / "files")
    documents = load_txt_documents(target_dir)

    print(f"成功加载 {len(documents)} 个文档到向量库")

# 创建我们自己写的 Embeddings 对象，它封装了调用 DashScope 向量化接口的逻辑
    embeddings = DashScopeEmbeddings()

    vector_store = Chroma(
        collection_name="knowledge_base",# 集合名，相当于数据库里的表名
        embedding_function=embeddings,# 用哪个 Embeddings 来向量化
        persist_directory="./chroma_db",   # 本地持久化路径
    )

    # 清空旧数据，防止重复运行时数据堆积
#     vector_store.get() 返回数据库里所有数据的详细信息，是个字典，["ids"] 就是取出所有 ID 的列表：


# {
#     "ids": ["id1", "id2", "id3", ...],
#     "documents": ["文本1", "文本2", ...],
#     "embeddings": [[...], [...], ...],
#     ...
# }
    try:
        existing_ids = vector_store.get()["ids"]
        if existing_ids:
            vector_store.delete(ids=existing_ids)
    except Exception as e:
        print(f"清理数据库时提示: {e}")

    # 把所有文档写入向量库
    if documents:
        # chroma自带方法，存 原文 + 向量 + metadata 
        _ = vector_store.add_documents(documents)

    return vector_store


# ── 创建 ReAct Agent ──────────────────────────────────────────────
# 把向量库封装成一个检索工具，传给 Agent
# Agent 收到问题后，先调用 retrieve_context 搜索相关片段，再结合内容回答
def create_react_agent_wrapper(vector_store: Chroma):
    """基于给定向量库创建带检索工具的 ReAct Agent。"""

    # response_format="content_and_artifact"：工具同时返回文字内容和原始对象，不加只能返回一个值
    # serialized 是给大模型看的文字，retrieved 是原始 Document 对象（供后续处理）
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """基于向量库检索与问题最相关的文本片段。"""
        retrieved = vector_store.similarity_search(query, k=3)
        serialized = "\n\n".join(
            f"[{doc.metadata['source']}#{doc.metadata['chunk_id']}] {doc.page_content}"
            for doc in retrieved
        )
        return serialized, retrieved

    return create_agent(
        llm,
        tools=[retrieve_context],
        system_prompt=(
            "你可以使用检索工具获得参考资料。回答时结合检索到的内容，"
            "如有必要可以在答案中简单引用来源标识。"
        ),
    )


# ── 运行演示 ──────────────────────────────────────────────────────
# 1. 构建向量库（读取 files/ 目录下的 txt 文件）
vector_store = build_vector_store()
print("嵌入完成\n")

# 2. 创建 Agent
agent = create_react_agent_wrapper(vector_store)

# 3. 提问
query = "公司的考勤方式是什么？"
print(f"Query: {query}\n" + "-" * 20)

input_payload = {"messages": [{"role": "user", "content": query}]}
# 用 stream 代替 invoke，区别是：

# invoke：等全部执行完再返回结果
# stream：每执行一步就返回一次，可以实时看到中间过程
for event in agent.stream(input_payload, stream_mode="values"):
    event["messages"][-1].pretty_print()
