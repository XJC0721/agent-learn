# 向量相似度路由（Embedding Similarity Routing）
# rag10 用 LLM 分类做路由，要调一次 LLM，慢且贵
# 这里用向量相似度做路由：把路由描述提前向量化，用户问题向量化后和描述算距离，选最近的
# 不调 LLM，只用嵌入模型，速度快、成本低
import sys
import os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
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

tech_docs = [
    Document(page_content="RAG是检索增强生成技术，将外部知识库与大语言模型结合使用。"),
    Document(page_content="LangChain是构建LLM应用的框架，提供Agent、Chain、Memory等组件。"),
    Document(page_content="向量数据库用于存储高维向量，支持语义相似度检索。"),
    Document(page_content="FAISS是Meta开发的向量计算库，适合本地快速原型开发。"),
    Document(page_content="混合检索结合BM25关键词检索和向量语义检索，召回率更高。"),
]

biz_docs = [
    Document(page_content="公司退款政策：7天内无理由退款，超过7天需提供质量问题证明。"),
    Document(page_content="会员等级分为普通、银牌、金牌三档，消费满1000元升银牌。"),
    Document(page_content="客服工作时间：周一至周五9:00-18:00，节假日不提供服务。"),
    Document(page_content="订单发货时间：付款后24小时内发货，偏远地区3-5个工作日到达。"),
    Document(page_content="促销活动：每月最后一周全场8折，会员额外享受9折叠加优惠。"),
]

print("正在构建向量库，请稍候...")
tech_store = Chroma.from_documents(tech_docs, embeddings, collection_name="tech")
biz_store  = Chroma.from_documents(biz_docs,  embeddings, collection_name="biz")
print("技术知识库和业务知识库构建完成")


# ── 2. 定义路由描述并向量化 ──────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 路由描述向量化（提前算好，不用每次调 LLM）")
print("=" * 50)

# 每个路由写一段描述，越具体越准确
route_descriptions = [
    "技术问题，涉及AI、RAG、LangChain、向量数据库、嵌入模型、编程开发等",
    "业务问题，涉及退款、订单、会员、客服、促销、发货、购买等",
]
route_names = ["tech", "biz"]

# 把路由描述向量化，存起来备用
# 这一步只在启动时做一次，之后每次路由只需要算用户问题的向量
route_vectors = embeddings.embed_documents(route_descriptions)
route_vectors_np = np.array(route_vectors)  # 转成 numpy 数组方便计算
print(f"已向量化 {len(route_descriptions)} 条路由描述")
print(f"每条描述的向量维度：{len(route_vectors[0])}")


# ── 3. 向量相似度路由函数 ────────────────────────────────────────────
def classify_by_embedding(question: str) -> str:
    # 第一步：把用户问题向量化
    query_vector = embeddings.embed_query(question)
    query_vector_np = np.array([query_vector])  # 变成二维数组才能算相似度

    # 第二步：计算和每条路由描述的余弦相似度
    # cosine_similarity 返回一个矩阵，[0] 取第一行，得到每条路由的分数
    scores = cosine_similarity(query_vector_np, route_vectors_np)[0]

    # 第三步：找最高分对应的路由
    # np.argmax 是"找最大值的位置"，不是最大值本身。
    best_index = int(np.argmax(scores))

    print(f"  相似度分数：", end="")#不换行
    for i, (name, score) in enumerate(zip(route_names, scores)):
        print(f"{name}={score:.4f}", end="  ")
    print()
    return route_names[best_index]


# ── 4. RAG 处理函数（和 rag10 相同）─────────────────────────────────
rag_prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题，回答简洁准确。\n\n上下文：{context}\n\n问题：{question}"
)


def ask(store: Chroma, store_name: str, question: str) -> str:
    docs = store.similarity_search(question, k=2)
    context = "\n".join(doc.page_content for doc in docs)
    print(f"  [路由到{store_name}] 检索到 {len(docs)} 条文档")
    response = llm.invoke(rag_prompt.format(context=context, question=question))
    return str(response.content)


def route(question: str) -> str:
    topic = classify_by_embedding(question)
    if topic == "tech":
        return ask(tech_store, "技术知识库", question)
    else:
        return ask(biz_store, "业务知识库", question)


# ── 5. 测试 ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. 路由测试")
print("=" * 50)

questions = [
    "RAG和向量数据库是什么关系",
    "我买的商品能退款吗",
    "LangChain有哪些核心组件",
    "会员折扣怎么计算",
]

for q in questions:
    print(f"\n问题：「{q}」")
    answer = route(q)
    print(f"  回答：{answer[:80]}...")


# ── 6. 对比两种路由方式 ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("对比总结")
print("=" * 50)
print("LLM分类路由（rag10）：调一次LLM判断意图，灵活但慢且贵")
print("向量相似度路由（rag11）：只调嵌入模型算距离，快且便宜")
print("选择建议：路由类别明确 → 向量路由；路由逻辑复杂 → LLM路由")
