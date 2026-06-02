# 四步构建 RAG（基础版）
# RAG = Retrieval Augmented Generation：检索增强生成
# 流程：加载文档 → 切片 → 向量化存入内存向量库 → 检索 → LLM 生成答案
import sys
import os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader

_ = load_dotenv()

# ── 配置模型 ──────────────────────────────────────────────────────
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
    temperature=0,
)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)


# ── 自定义 Embeddings ─────────────────────────────────────────────
# 文档用的是 HuggingFace 本地模型，这里改用 DashScope API，功能相同
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


# ── 第1步：数据准备 ───────────────────────────────────────────────
# TextLoader 读取 markdown 文件，RecursiveCharacterTextSplitter 按段落/句子递归切片
# chunk_size：每块最大字符数；chunk_overlap：相邻块重叠字符数，防止切断语义
md_path = Path(__file__).parent / "files" / "rag_knowledge.md"
loader = TextLoader(str(md_path), encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"文档切分为 {len(chunks)} 个片段")


# ── 第2步：建索引 ─────────────────────────────────────────────────
# InMemoryVectorStore：内存向量库，程序结束数据消失（对比 Chroma 的持久化）
# add_documents 内部自动调用 embed_documents 向量化再存入
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)
print("向量库构建完成\n")


# ── 第3步：检索 ───────────────────────────────────────────────────
# 用户提问 → 问题向量化 → 和库里所有向量做相似度计算 → 返回最相关的 k 个片段
question = "RAG有哪些检索优化技术？"
retrieved_docs = vectorstore.similarity_search(question, k=3)

# 把检索到的多个片段拼接成一段上下文
context = "\n\n".join(doc.page_content for doc in retrieved_docs)


# ── 第4步：生成答案 ───────────────────────────────────────────────
# 把上下文 + 问题填进 prompt 模板，让 LLM 基于上下文回答，不许乱编
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:""")

answer = llm.invoke(prompt.format(question=question, context=context))

print(f"问题：{question}")
print("-" * 30)
print(answer.content)
