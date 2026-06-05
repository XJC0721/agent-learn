# 结构化索引：元数据过滤 + 向量搜索
# 问题：知识库很大时，全库向量搜索慢且容易被不相关内容干扰
# 解决：存文档时打上元数据标签（部门、年份、类型），检索时先过滤再搜索
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

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

# ── 1. 构建带元数据的知识库 ──────────────────────────────────────────
# 模拟一个企业知识库，包含多个部门、多个年份的文档
# metadata 就是给每个文档贴的"标签"，用于后续过滤
print("=" * 50)
print("1. 构建带元数据的知识库")
print("=" * 50)

docs = [
    # 技术部 2023
    Document(page_content="2023年技术部完成了RAG系统的研发，采用Chroma向量数据库，检索准确率提升40%。",
             metadata={"department": "技术部", "year": 2023, "type": "年报"}),
    Document(page_content="2023年技术部引入LangChain框架，统一了Agent开发规范，开发效率提升30%。",
             metadata={"department": "技术部", "year": 2023, "type": "年报"}),
    Document(page_content="2023年技术部服务器成本同比增加15%，主要用于GPU算力扩容。",
             metadata={"department": "技术部", "year": 2023, "type": "财报"}),

    # 技术部 2024
    Document(page_content="2024年技术部完成多模态模型的接入，支持图文混合检索。",
             metadata={"department": "技术部", "year": 2024, "type": "年报"}),
    Document(page_content="2024年技术部基础设施成本下降20%，云服务采购优化效果显著。",
             metadata={"department": "技术部", "year": 2024, "type": "财报"}),

    # 市场部 2023
    Document(page_content="2023年市场部AI营销活动覆盖用户500万，转化率较传统方式提升25%。",
             metadata={"department": "市场部", "year": 2023, "type": "年报"}),
    Document(page_content="2023年市场部广告投放费用1200万，其中数字营销占比达70%。",
             metadata={"department": "市场部", "year": 2023, "type": "财报"}),

    # 市场部 2024
    Document(page_content="2024年市场部推出AI个性化推荐系统，用户留存率提升18%。",
             metadata={"department": "市场部", "year": 2024, "type": "年报"}),
    Document(page_content="2024年市场部营销费用同比下降10%，AI工具替代了部分人工运营工作。",
             metadata={"department": "市场部", "year": 2024, "type": "财报"}),

    # 人事部 2023
    Document(page_content="2023年人事部引入AI简历筛选系统，招聘周期从30天缩短至15天。",
             metadata={"department": "人事部", "year": 2023, "type": "年报"}),
    Document(page_content="2023年人事部全年招聘120人，技术岗位占比60%，人均培训费用8000元。",
             metadata={"department": "人事部", "year": 2023, "type": "财报"}),
]

# 使用 Chroma 内存模式（不持久化，演示用）
vectorstore = Chroma.from_documents(docs, embeddings)
print(f"共存入 {len(docs)} 条文档，每条都带有 department/year/type 元数据")


# ── 2. 普通向量搜索（全库检索，不过滤）───────────────────────────────
print("\n" + "=" * 50)
print("2. 普通向量搜索（全库，不过滤）")
print("=" * 50)

query = "AI技术的应用成果"
results = vectorstore.similarity_search(query, k=3)
print(f"查询：'{query}'")
for i, doc in enumerate(results):
    meta = doc.metadata
    print(f"  结果{i+1} [{meta['department']} {meta['year']} {meta['type']}]：{doc.page_content[:40]}...")


# ── 3. 元数据过滤搜索 ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. 元数据过滤搜索（先过滤，再搜索）")
print("=" * 50)

# 场景1：只看技术部2023年的内容
print("\n场景1：技术部 + 2023年")
results = vectorstore.similarity_search(
    query,
    k=2,
    filter={"$and": [{"department": "技术部"}, {"year": 2023}]}
)
for i, doc in enumerate(results):
    meta = doc.metadata
    print(f"  结果{i+1} [{meta['department']} {meta['year']}]：{doc.page_content[:50]}...")

# 场景2：只看财报类文档
print("\n场景2：只看财报")
results = vectorstore.similarity_search(
    "费用和成本",
    k=3,
    filter={"type": "财报"}
)
for i, doc in enumerate(results):
    meta = doc.metadata
    print(f"  结果{i+1} [{meta['department']} {meta['year']}]：{doc.page_content[:50]}...")

# 场景3：2024年所有部门
print("\n场景3：2024年所有部门")
results = vectorstore.similarity_search(
    "AI应用",
    k=3,
    filter={"year": 2024}
)
for i, doc in enumerate(results):
    meta = doc.metadata
    print(f"  结果{i+1} [{meta['department']} {meta['year']}]：{doc.page_content[:50]}...")


# ── 4. 对比总结 ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("4. 对比总结")
print("=" * 50)
print("普通搜索：11条文档全部参与向量计算，结果可能混入不同部门/年份")
print("元数据过滤：先缩小候选范围，只在目标子集内搜索，更精准更高效")
print("知识库越大，元数据过滤的优势越明显")
