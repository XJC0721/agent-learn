# 查询路由（Query Routing）
# 根据用户问题的意图，自动分发到不同的知识库或处理链
# 核心：LLM 先分类 → 代码根据分类结果路由到对应处理器
import sys
import os
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

# ── 1. 构建两个不同的知识库（模拟多数据源场景）────────────────────────
print("=" * 50)
print("1. 构建知识库")
print("=" * 50)

# 技术知识库
tech_docs = [
    Document(page_content="RAG是检索增强生成技术，将外部知识库与大语言模型结合使用。"),
    Document(page_content="LangChain是构建LLM应用的框架，提供Agent、Chain、Memory等组件。"),
    Document(page_content="向量数据库用于存储高维向量，支持语义相似度检索。"),
    Document(page_content="FAISS是Meta开发的向量计算库，适合本地快速原型开发。"),
    Document(page_content="混合检索结合BM25关键词检索和向量语义检索，召回率更高。"),
]

# 业务知识库
biz_docs = [
    Document(page_content="公司退款政策：7天内无理由退款，超过7天需提供质量问题证明。"),
    Document(page_content="会员等级分为普通、银牌、金牌三档，消费满1000元升银牌。"),
    Document(page_content="客服工作时间：周一至周五9:00-18:00，节假日不提供服务。"),
    Document(page_content="订单发货时间：付款后24小时内发货，偏远地区3-5个工作日到达。"),
    Document(page_content="促销活动：每月最后一周全场8折，会员额外享受9折叠加优惠。"),
]

print("正在构建向量库，请稍候...")
tech_store = Chroma.from_documents(tech_docs, embeddings, collection_name="tech")
biz_store = Chroma.from_documents(biz_docs, embeddings, collection_name="biz")
print("技术知识库和业务知识库构建完成")


# ── 2. 定义分类器 ─────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 定义路由分类器")
print("=" * 50)

classifier_prompt = ChatPromptTemplate.from_template(
    """将用户问题分类为以下两类之一：
- tech：技术问题（涉及AI、RAG、LangChain、向量数据库、编程等）
- biz：业务问题（涉及退款、订单、会员、客服、促销等）

只输出一个单词：tech 或 biz，不要任何解释。

问题：{question}"""
)


def classify(question: str) -> str:
    # 填入 prompt，发给 LLM，取出文字结果
    response = llm.invoke(classifier_prompt.format(question=question))
    return str(response.content).strip()


print("分类器构建完成")


# ── 3. 定义两条 RAG 处理函数 ──────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题，回答简洁准确。\n\n上下文：{context}\n\n问题：{question}"
)


def ask_tech(question: str) -> str:
    docs = tech_store.similarity_search(question, k=2)
    context = "\n".join(doc.page_content for doc in docs)
    print(f"  [路由到技术知识库] 检索到 {len(docs)} 条文档")
    response = llm.invoke(rag_prompt.format(context=context, question=question))
    return str(response.content)


def ask_biz(question: str) -> str:
    docs = biz_store.similarity_search(question, k=2)
    context = "\n".join(doc.page_content for doc in docs)
    print(f"  [路由到业务知识库] 检索到 {len(docs)} 条文档")
    response = llm.invoke(rag_prompt.format(context=context, question=question))
    return str(response.content)


# ── 4. 路由函数：分类 → 选择处理函数 ────────────────────────────────
def route(question: str) -> str:
    # 第一步：LLM 分类
    topic = classify(question)
    print(f"  分类结果：{topic}")

    # 第二步：根据分类结果路由到对应知识库
    if "tech" in topic:
        return ask_tech(question)
    elif "biz" in topic:
        return ask_biz(question)
    else:
        return ask_biz(question)  # 默认走业务库


# ── 5. 测试路由效果 ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. 路由测试")
print("=" * 50)

questions = [
    "RAG和向量数据库是什么关系",   # 技术问题
    "我买的商品能退款吗",           # 业务问题
    "LangChain有哪些核心组件",     # 技术问题
    "会员折扣怎么计算",             # 业务问题
]

for q in questions:
    print(f"\n问题：「{q}」")
    answer = route(q)
    print(f"  回答：{answer[:80]}...")


print("\n" + "=" * 50)
print("总结")
print("=" * 50)
print("classify()  ：LLM 判断问题类型，返回 tech 或 biz")
print("ask_tech()  ：去技术知识库检索，调 LLM 生成答案")
print("ask_biz()   ：去业务知识库检索，调 LLM 生成答案")
print("route()     ：先分类，再根据结果选择对应处理函数")
print("应用场景：多知识库路由、用户身份路由（Adaptive-RAG）、组件分流")
