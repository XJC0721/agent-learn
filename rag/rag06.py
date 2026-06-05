# 句子窗口检索（Sentence Window Retrieval）
# 核心思想：为检索精确性而索引小块（单句），为生成质量而扩展大块（窗口）
# 检索时：按单句向量化，找到最精准的匹配句
# 生成时：把该句前后 N 句一起传给 LLM，保证上下文完整
import sys
import os
import re
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

files_dir = Path(__file__).parent / "files"
text = (files_dir / "rag_knowledge.md").read_text(encoding="utf-8")


# ── 工具函数 ─────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    # 按中文句号/问号/感叹号/换行切分，过滤空行，返回列表
    # (?<=...) 是"向后看"，意思是在 。！？\n 之后切，保留标点在前面那句里。
    parts = re.split(r'(?<=[。！？\n])', text)
    return [p.strip() for p in parts if p.strip()]
# 等价于
# result = []
#for p in parts:
 #   if p.strip():          # 如果这句话去掉空格后不是空字符串
  #      result.append(p.strip())  # 才加进去，同时去掉首尾空格


def build_window_docs(sentences: list[str], window_size: int = 3) -> list[Document]:
    """每个句子单独存为 Document，metadata 中保存前后窗口文本"""
    docs = []
    for i, sent in enumerate(sentences):
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        # [start:end] 是左闭右开，用 "" 作为分隔符（即不加任何分隔）
        window_text = "".join(sentences[start:end])
        docs.append(Document(
            page_content=sent,          # 只用这一句做向量化，保证检索精准
            metadata={
                "window": window_text,  # 完整窗口留给 LLM 使用
                "idx": i,
            }
        ))
    return docs


def expand_to_window(docs: list[Document]) -> list[Document]:
    """检索后把 page_content 替换为完整窗口"""
    return [
        Document(page_content=doc.metadata["window"], metadata=doc.metadata)
        for doc in docs
    ]


prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题，回答要简洁准确。\n\n上下文：{context}\n\n问题：{question}"
)


def ask(vectorstore: InMemoryVectorStore, question: str, use_window: bool = False) -> str:

    retrieved = vectorstore.similarity_search(question, k=2)
    if use_window:
        retrieved = expand_to_window(retrieved)
    context = "\n\n".join(doc.page_content for doc in retrieved)
    answer = llm.invoke(prompt.format(question=question, context=context))
    return answer.content


# ── 1. 构建句子窗口向量库 ─────────────────────────────────────────────
print("=" * 50)
print("构建索引")
print("=" * 50)

sentences = split_sentences(text)
window_docs = build_window_docs(sentences, window_size=3)
window_store = InMemoryVectorStore(embeddings)
window_store.add_documents(window_docs)
print(f"句子总数：{len(sentences)}")
print(f"每个 Document 存的是单句，metadata 里存前后 3 句的完整窗口")

# ── 2. 构建常规分块向量库（用于对比）────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# 大字符串 text 切成 Document 列表。
base_chunks = splitter.create_documents([text])
base_store = InMemoryVectorStore(embeddings)
# 把所有 chunk 向量化后存进向量库，这一步会调用 DashScope API。
base_store.add_documents(base_chunks)
print(f"常规分块数：{len(base_chunks)}")


# ── 3. 对比两种检索方式 ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("对比：句子窗口 vs 常规分块")
print("=" * 50)

question = "RAG 的检索优化有哪些技术？"
print(f"\n问题：{question}")

print("\n--- 句子窗口检索（检索单句，扩展后传给 LLM）---")
# 先看检索到的原始单句
raw = window_store.similarity_search(question, k=2)
for i, doc in enumerate(raw):
    print(f"  检索到句{i+1}：{doc.page_content[:60]}...")
# 扩展为窗口后的效果
answer_window = ask(window_store, question, use_window=True)
print(f"  LLM 回答：{answer_window}")

print("\n--- 常规分块检索 ---")
answer_base = ask(base_store, question, use_window=False)
print(f"  LLM 回答：{answer_base}")


# ── 4. 展示窗口扩展效果 ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("窗口扩展效果展示")
print("=" * 50)

sample = window_store.similarity_search(question, k=1)[0]
print(f"检索到的单句（向量化内容）：\n  {sample.page_content}")
print(f"\n扩展后传给 LLM 的完整窗口：\n  {sample.metadata['window']}")


# 扩展的是把content切换成windowtext，然后给llm