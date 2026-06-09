# 混合检索（Hybrid Search）
# 单纯向量检索对专有名词不敏感，单纯关键词不懂语义
# 混合 = BM25（稀疏，关键词精准）+ FAISS（密集，语义理解）→ 手动 RRF 融合
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from openai import OpenAI
import jieba
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

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

# ── 知识库文档 ─────────────────────────────────────────────────────────
docs = [
    Document(page_content="RAG是检索增强生成技术，将外部知识库与大语言模型结合，先检索相关文档再生成回答。"),
    Document(page_content="FAISS是Meta开发的向量计算库，支持高效的近似最近邻搜索，适合本地原型开发。"),
    Document(page_content="BM25是经典的关键词检索算法，基于词频TF和逆文档频率IDF计算相关性分数。"),
    Document(page_content="混合检索结合稀疏向量和密集向量，既能精确匹配关键词，又能理解语义近义词。"),
    Document(page_content="向量数据库专门存储高维向量，支持相似度搜索，常见有Chroma、Milvus、Pinecone。"),
    Document(page_content="RRF算法融合多路检索结果，不依赖原始分数只看排名，最终分数=各路1/(rank+60)之和。"),
    Document(page_content="句子窗口检索将文档切成单句索引，检索精准后再扩展为前后窗口传给LLM。"),
    Document(page_content="Chroma是轻量级向量数据库，支持元数据过滤，本地零配置安装。"),
    Document(page_content="文本嵌入模型将自然语言转化为高维向量，相似语义的文本在向量空间中距离接近。"),
    Document(page_content="GPT-4o是OpenAI发布的多模态大语言模型，支持文本、图像、音频多种输入格式。"),
    Document(page_content="Reranker重排序模型对检索结果做二次精排，比向量相似度更精准但计算开销更大。"),
    Document(page_content="查询重写通过改写用户问题来提升检索召回率，常见方法有HyDE和Step-back。"),
]


# ── 1. 稀疏检索：BM25 ─────────────────────────────────────────────────
print("=" * 50)
print("构建检索器")
print("=" * 50)

# jieba 分词：把中文句子切成词列表，BM25 按词统计词频
# "混合检索结合稀疏向量" → ["混合", "检索", "结合", "稀疏", "向量"]
# 内置了一个词典（几十万个中文词），加上一个基于隐马尔可夫模型（HMM） 训练的算法，用来判断哪里是词的边界。
def jieba_tokenizer(text: str) -> list[str]:
    # jieba.cut(text) — 把一段中文文本切成词，返回一个生成器
    return list(jieba.cut(text))

# 构建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_tokenizer)
# 检索时最多返回 4 条结果
bm25_retriever.k = 4
print("稀疏检索器（BM25）构建完成，纯本地统计，无需 API 调用")

# ── 2. 密集检索：FAISS 向量 ───────────────────────────────────────────
print("正在调用 DashScope 生成向量，请稍候...")
vectorstore = FAISS.from_documents(docs, embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("密集检索器（FAISS）构建完成\n")


# ── 3. 手动实现 RRF 融合 ──────────────────────────────────────────────
def rrf_merge(
    list1: list[Document],
    list2: list[Document],
    weight1: float = 0.5,
    weight2: float = 0.5,
    c: int = 60,
) -> list[Document]:
    """
    RRF 倒数排名融合：不看原始分数，只看每个文档在两个列表里的排名。
    分数 = weight1 * 1/(rank_in_list1 + c) + weight2 * 1/(rank_in_list2 + c)
    c=60 是平滑系数，防止第1名权重远超第2名，让结果更稳定。
    """

    # 字典，键值对，看{}
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(list1):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + weight1 * (1 / (rank + c))
        doc_map[key] = doc

    for rank, doc in enumerate(list2):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + weight2 * (1 / (rank + c))
        doc_map[key] = doc

    # 按分数降序排列，返回前3条
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys[:3]]


# ── 4. 对比三种检索方式 ──────────────────────────────────────────────
def compare(query: str, w1: float = 0.5, w2: float = 0.5):
    print(f"\n{'='*50}")
    print(f"查询：「{query}」  (BM25权重={w1}, 向量权重={w2})")

    bm25_results  = bm25_retriever.invoke(query)
    dense_results = dense_retriever.invoke(query)
    hybrid_results = rrf_merge(bm25_results, dense_results, weight1=w1, weight2=w2)

    print("\n[稀疏 BM25] ——关键词匹配：")
    for i, doc in enumerate(bm25_results[:3]):
        print(f"  Top{i+1}: {doc.page_content[:50]}...")

    print("\n[密集 FAISS] ——语义向量：")
    for i, doc in enumerate(dense_results[:3]):
        print(f"  Top{i+1}: {doc.page_content[:50]}...")

    print("\n[混合 RRF] ——融合结果：")
    for i, doc in enumerate(hybrid_results):
        print(f"  Top{i+1}: {doc.page_content[:50]}...")


# 场景1：精确专有名词——稀疏占优
compare("BM25算法")

# 场景2：语义模糊查询——密集占优
compare("怎么让检索结果更准确")

# 场景3：电商偏关键词 weights 调整
compare("向量相似度搜索", w1=0.7, w2=0.3)

# 场景4：问答偏语义 weights 调整
compare("向量相似度搜索", w1=0.3, w2=0.7)


print("\n" + "=" * 50)
print("总结")
print("=" * 50)
print("BM25       ——纯关键词，快，不懂语义，对专有名词精准")
print("FAISS 向量 ——语义检索，能找近义词，对新词/缩写不敏感")
print("混合 RRF   ——两路并行，按排名融合，召回率和准确率都高")
print("权重调整   ——w1>w2 偏关键词（电商），w2>w1 偏语义（问答）")
