# 向量数据库：FAISS
# FAISS 是 Meta 开发的向量计算库，把索引保存为本地文件（.faiss + .pkl）
# 与 Chroma 不同，它没有数据库服务，纯文件存取，轻量适合原型开发
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_ = load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)


class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-v4", dimensions: int = 1024):
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 10):
            chunk = texts[i: i + 10]
            response = client.embeddings.create(model=self.model, input=chunk, dimensions=self.dimensions)
            vectors.extend([item.embedding for item in response.data])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = client.embeddings.create(model=self.model, input=[text], dimensions=self.dimensions)
        return response.data[0].embedding


embeddings = DashScopeEmbeddings()

# ── 1. 创建向量存储并保存到本地 ───────────────────────────────────────
print("=" * 50)
print("1. 创建 FAISS 向量存储并保存到本地")
print("=" * 50)

texts = [
    "RAG 是一种将外部知识库与大语言模型结合的技术，在生成答案前先检索相关文档。",
    "FAISS 是 Meta 开发的高性能向量计算库，专用于高效相似性搜索和密集向量聚类。",
    "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。",
    "向量数据库专门用于存储和检索高维向量，支持语义相似度搜索。",
    "Chroma 是一款轻量级开源向量数据库，本地优先设计，零配置安装。",
    "Milvus 是开源的分布式向量数据库，支持 GPU 加速，适合亿级向量检索。",
]
# Document(
#     page_content="文本内容",   # 必填，存文字
#     metadata={"source": "xx"}  # 选填，存额外信息
# )
# docs 就是一个列表，里面每个元素都是 Document 对象。（把原来那个字符串变成docs）
docs = [Document(page_content=t) for t in texts]
# 会把 docs 里每个 Document 的 page_content 文本，交给 embeddings 模型转成向量
vectorstore = FAISS.from_documents(docs, embeddings)

# 保存到本地文件（生成 .faiss 索引文件 + .pkl 映射文件）
# 索引文件（.faiss）：存的是向量本身，就是一堆浮点数。查询时在这里做相似度计算，找最近邻。

# 映射文件（.pkl）：存的是向量和原始文本的对应关系。因为 FAISS 内部只认向量编号（0, 1, 2...），
# 不知道编号 0 对应的文字是什么，.pkl 就是这个翻译表。
faiss_path = Path(__file__).parent / "files" / "faiss_index"
vectorstore.save_local(str(faiss_path))
print(f"索引已保存到：{faiss_path}")


# ── 2. 从本地加载并查询 ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 从本地加载 FAISS 索引并执行相似度搜索")
print("=" * 50)

# allow_dangerous_deserialization=True：加载 .pkl 文件需要此参数
loaded_store = FAISS.load_local(
    str(faiss_path),
    embeddings,
    allow_dangerous_deserialization=True,
)

queries = ["什么是 RAG？", "有哪些向量数据库？"]
for query in queries:
    results = loaded_store.similarity_search(query, k=2)#返回最相似的前 2 条结果
    print(f"\n查询：'{query}'")
    for i, doc in enumerate(results):
        print(f"  Top{i+1}: {doc.page_content}")


# ── 3. 增量添加文档 ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. 增量添加文档（FAISS 支持动态追加）")
print("=" * 50)

new_docs = [Document(page_content="Qdrant 是用 Rust 开发的高性能向量数据库，支持二进制量化，适合性能敏感场景。")]
loaded_store.add_documents(new_docs)
print(f"追加后文档总数：{loaded_store.index.ntotal}")

# 重新保存（覆盖原文件）
loaded_store.save_local(str(faiss_path))
print("更新后的索引已保存")
