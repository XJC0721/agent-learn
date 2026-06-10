# 查询构建（Query Construction）- 自查询检索
# rag07 手动写 filter，这里让 LLM 自动从自然语言里提取 filter
# 核心：用 LLM 把用户问题拆成 "语义查询" + "结构化过滤条件" 两部分
import sys
import os
import json
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,  # 设为0：让输出稳定可复现，不要随机性，filter必须精确
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

# ── 1. 构建知识库（和 rag07 相同的企业文档）────────────────────────────
print("=" * 50)
print("1. 构建知识库")
print("=" * 50)

docs = [
    Document(page_content="2023年技术部完成了RAG系统的研发，采用Chroma向量数据库，检索准确率提升40%。",
             metadata={"department": "技术部", "year": 2023, "type": "年报"}),
    Document(page_content="2023年技术部引入LangChain框架，统一了Agent开发规范，开发效率提升30%。",
             metadata={"department": "技术部", "year": 2023, "type": "年报"}),
    Document(page_content="2023年技术部服务器成本同比增加15%，主要用于GPU算力扩容。",
             metadata={"department": "技术部", "year": 2023, "type": "财报"}),
    Document(page_content="2024年技术部完成多模态模型的接入，支持图文混合检索。",
             metadata={"department": "技术部", "year": 2024, "type": "年报"}),
    Document(page_content="2024年技术部基础设施成本下降20%，云服务采购优化效果显著。",
             metadata={"department": "技术部", "year": 2024, "type": "财报"}),
    Document(page_content="2023年市场部AI营销活动覆盖用户500万，转化率较传统方式提升25%。",
             metadata={"department": "市场部", "year": 2023, "type": "年报"}),
    Document(page_content="2023年市场部广告投放费用1200万，其中数字营销占比达70%。",
             metadata={"department": "市场部", "year": 2023, "type": "财报"}),
    Document(page_content="2024年市场部推出AI个性化推荐系统，用户留存率提升18%。",
             metadata={"department": "市场部", "year": 2024, "type": "年报"}),
    Document(page_content="2024年市场部营销费用同比下降10%，AI工具替代了部分人工运营工作。",
             metadata={"department": "市场部", "year": 2024, "type": "财报"}),
    Document(page_content="2023年人事部引入AI简历筛选系统，招聘周期从30天缩短至15天。",
             metadata={"department": "人事部", "year": 2023, "type": "年报"}),
    Document(page_content="2023年人事部全年招聘120人，技术岗位占比60%，人均培训费用8000元。",
             metadata={"department": "人事部", "year": 2023, "type": "财报"}),
]

vectorstore = Chroma.from_documents(docs, embeddings)
print(f"共存入 {len(docs)} 条文档，每条都带有 department/year/type 元数据")


# ── 2. 让 LLM 自动生成 filter ─────────────────────────────────────────
print("\n" + "=" * 50)
print("2. LLM 自动解析查询条件")
print("=" * 50)

# 告诉 LLM 有哪些字段可以过滤，以及字段的含义
METADATA_SCHEMA = """
可用的元数据字段：
- department (字符串): 部门名称，可选值："技术部"、"市场部"、"人事部"
- year (整数): 年份，可选值：2023、2024
- type (字符串): 文档类型，可选值："年报"、"财报"
"""

# 提示词：让 LLM 把用户问题拆成 query（语义搜索部分）+ filter（精确过滤部分）
extract_prompt = ChatPromptTemplate.from_template(
    """你是一个查询解析助手。根据用户问题和可用的元数据字段，将问题拆分为：
1. query：用于语义搜索的核心内容（去掉过滤条件后的部分）
2. filter：从问题中提取的结构化过滤条件，格式为 Chroma 的 filter 字典

{schema}

只输出 JSON，不要任何解释。格式：
{{"query": "语义搜索内容", "filter": {{过滤条件}}}}

如果没有过滤条件，filter 为 {{}}。
多个条件用 {{"$and": [条件1, 条件2]}}。

用户问题：{question}"""
)


def self_query(question: str, k: int = 3) -> list[Document]:
    # 第一步：LLM 解析问题，生成 query + filter
    response = llm.invoke(extract_prompt.format(
        schema=METADATA_SCHEMA,
        question=question
    ))

    # 解析 LLM 返回的 JSON
    raw = response.content.strip()
    # 去掉 LLM 可能包的 markdown 代码块
    # startswith表示以xx开头就..
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw.strip())

    query = parsed.get("query", question)
    filter_dict = parsed.get("filter", {})

    print(f"  原始问题：{question}")
    print(f"  LLM生成 query：{query}")
    print(f"  LLM生成 filter：{filter_dict}")

    # 第二步：用解析出的 query + filter 执行检索
    if filter_dict:
        results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
    else:
        results = vectorstore.similarity_search(query, k=k)
    return results


# ── 3. 对比：手动 filter vs LLM 自动生成 ─────────────────────────────
print("\n" + "=" * 50)
print("3. 自然语言 → 自动生成 filter → 检索")
print("=" * 50)

queries = [
    "2023年技术部做了哪些技术创新",
    "所有部门的财报",
    "2024年的年报",
]

for q in queries:
    print(f"\n{'─'*40}")
    results = self_query(q, k=2)
    print(f"  检索结果（{len(results)}条）：")
    for doc in results:
        meta = doc.metadata
        print(f"    [{meta['department']} {meta['year']} {meta['type']}] {doc.page_content[:40]}...")


# ── 4. 对比总结 ───────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("4. 对比总结")
print("=" * 50)
print("rag07 手动 filter：开发者写死过滤条件，用户看不见")
print("rag09 自动 filter：用户说自然语言，LLM 自动翻译成 filter")
print("优势：用户不需要知道字段名，系统自动理解意图并过滤")
