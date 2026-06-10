# 重排序（Re-ranking）- RankLLM 方式
# 问题：向量检索召回的 Top-N 排序不一定准，最相关的可能不在第一位
# 解决：两阶段流程 → 第一阶段向量检索召回 Top-6，第二阶段 LLM 精排取 Top-3
import sys
import os
import json
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
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
    temperature=0,
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

# ── 1. 构建知识库 ─────────────────────────────────────────────────────
print("=" * 50)
print("1. 构建知识库")
print("=" * 50)

docs = [
    Document(page_content="RAG是检索增强生成技术，将外部知识库与大语言模型结合，先检索相关文档再生成回答。"),
    Document(page_content="FAISS是Meta开发的向量计算库，支持高效的近似最近邻搜索，适合本地原型开发。"),
    Document(page_content="BM25是经典的关键词检索算法，基于词频TF和逆文档频率IDF计算相关性分数。"),
    Document(page_content="混合检索结合稀疏向量和密集向量，既能精确匹配关键词，又能理解语义近义词。"),
    Document(page_content="向量数据库专门存储高维向量，支持相似度搜索，常见有Chroma、Milvus、Pinecone。"),
    Document(page_content="RRF算法融合多路检索结果，不依赖原始分数只看排名，最终分数=各路1/(rank+60)之和。"),
    Document(page_content="重排序（Re-ranking）对初步检索结果进行二次精排，Cross-Encoder精度最高但速度慢。"),
    Document(page_content="句子窗口检索将文档切成单句索引，检索精准后再扩展为前后窗口传给LLM。"),
    Document(page_content="Chroma是轻量级向量数据库，支持元数据过滤，本地零配置安装。"),
    Document(page_content="文本嵌入模型将自然语言转化为高维向量，相似语义的文本在向量空间中距离接近。"),
    Document(page_content="LangChain是构建LLM应用的框架，提供Agent、Chain、Memory等组件。"),
    Document(page_content="查询重写通过改写用户问题来提升检索召回率，常见方法有HyDE和Step-back。"),
]

print("正在构建向量库，请稍候...")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # 第一阶段多召回一些
print(f"知识库构建完成，共 {len(docs)} 条文档")


# ── 2. RankLLM 重排函数 ───────────────────────────────────────────────
rerank_prompt = ChatPromptTemplate.from_template(
    """你是一个文档相关性评估助手。给定用户问题和一组候选文档，按与问题的相关性从高到低排序。

用户问题：{question}

候选文档：
{docs_text}

只输出 JSON 数组，格式为文档编号列表，最相关的排最前，只返回最相关的 {top_k} 个：
["1", "3", "5"]

不要任何解释，只输出 JSON。"""
)


def rerank(question: str, candidates: list[Document], top_k: int = 3) -> list[Document]:
    # 把候选文档编号后拼成文本给 LLM 看
    docs_text = "\n".join(
        f"文档 {i+1}: {doc.page_content}" for i, doc in enumerate(candidates)
    )

    response = llm.invoke(rerank_prompt.format(
        question=question,
        docs_text=docs_text,
        top_k=top_k,
    ))

    raw = str(response.content).strip()
    # 去掉可能的 markdown 代码块
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    ranked_indices = json.loads(raw.strip())
    # LLM 返回的是字符串编号（"1","2"...），转成 0-based 索引取文档
    result = []
    for idx_str in ranked_indices[:top_k]:
        idx = int(idx_str) - 1
        if 0 <= idx < len(candidates):
            result.append(candidates[idx])
    return result


# ── 3. 两阶段检索对比 ─────────────────────────────────────────────────
def search_with_rerank(question: str):
    print(f"\n{'='*50}")
    print(f"问题：「{question}」")

    # 第一阶段：向量检索召回 Top-6
    candidates = retriever.invoke(question)
    print(f"\n[第一阶段] 向量检索 Top-6：")
    for i, doc in enumerate(candidates):
        print(f"  {i+1}. {doc.page_content[:45]}...")

    # 第二阶段：LLM 精排取 Top-3
    reranked = rerank(question, candidates, top_k=3)
    print(f"\n[第二阶段] LLM 重排 Top-3：")
    for i, doc in enumerate(reranked):
        print(f"  {i+1}. {doc.page_content[:45]}...")

    return reranked


# ── 4. 测试 ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 两阶段检索测试")
print("=" * 50)

questions = [
    "如何提升检索结果的准确性",
    "向量数据库有哪些选择",
]

for q in questions:
    search_with_rerank(q)


print("\n" + "=" * 50)
print("总结")
print("=" * 50)
print("第一阶段（向量检索）：召回多，排序粗，速度快")
print("第二阶段（LLM重排） ：从候选中精选，排序准，但多一次 LLM 调用")
print("适用场景：对精度要求高、top-1 结果必须准确的问答系统")
