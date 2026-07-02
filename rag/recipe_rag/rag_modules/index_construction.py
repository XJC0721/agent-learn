# 索引构建：向量化子文档 → 建 FAISS + BM25 双索引，支持持久化
#
# 和 rag08 的区别：
#   - rag08：直接对完整文档建索引
#   - 这里：对子文档切片建索引，但 parent_map 保留完整菜谱
#   检索命中子块 → 通过 parent_id 找回完整菜谱 → 传给 LLM
import sys
import os
import pickle
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    EMBED_MODEL, EMBED_DIMS, INDEX_DIR,
)


# ── DashScope 向量化（与之前 rag05-12 相同） ──────────────────────────────
class DashScopeEmbeddings(Embeddings):
    #  创建连接，存到 self.client
    def __init__(self, model: str = EMBED_MODEL, dimensions: int = EMBED_DIMS):
        self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # DashScope 单次最多 10 条，分批处理，range(起始, 结束, 步长)
        for i in range(0, len(texts), 10):
            batch = texts[i: i + 10]
            # 拿出来用，用这个连接调 API
            resp = self.client.embeddings.create(model=self.model, input=batch, dimensions=self.dimensions)
            # append加整个列表，extend把列表里的元素逐个加进去
            vectors.extend([item.embedding for item in resp.data])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=[text], dimensions=self.dimensions)
        return resp.data[0].embedding

# 创建上述实例
embeddings = DashScopeEmbeddings()


# ── FAISS 索引：建立 / 加载 ────────────────────────────────────────────────
# 定义文件存储路径
FAISS_PATH = INDEX_DIR / "faiss_index"
# .pkl 是 Python 的序列化格式（pickle），
# 可以把任何 Python 对象（字典、列表等）直接存成文件，下次读出来还是原来的对象。
BM25_PATH = INDEX_DIR / "bm25.pkl"
PARENT_MAP_PATH = INDEX_DIR / "parent_map.pkl"


def build_faiss(child_docs: list[Document]) -> FAISS:
    """向量化子文档，建 FAISS 索引并保存到本地"""
    print("正在向量化并构建 FAISS 索引（首次约需 2-3 分钟）...")
    # 内部会自动调用 embeddings.embed_documents()
    vectorstore = FAISS.from_documents(child_docs, embeddings)
    vectorstore.save_local(str(FAISS_PATH))
    print(f"FAISS 索引已保存：{FAISS_PATH}")
    return vectorstore


def load_faiss() -> FAISS:
    # 从哪里读，用什么模型，是否允许读pkl文件
    return FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)


def build_bm25(child_docs: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """对子文档建 BM25 索引，按字符切分（适合中文）"""
    # list("水煮鱼的做法")    # 变成单字符列表：["水","煮","鱼","的","做","法"]
    # list() 作用在字符串上，就是把字符串拆成一个个字符组成列表。
    tokenized = [list(doc.page_content) for doc in child_docs]
    # bm25 是一个对象，是 BM25Okapi 类的实例。
    # 把所有子文档的字符列表传进去，BM25Okapi 内部做统计计算，建好索引存在 bm25 对象里。
    bm25 = BM25Okapi(tokenized)
    # 写入文件 wb表示二进制 把 open() 返回的文件对象赋值给 f
    with open(BM25_PATH, "wb") as f:
        # pickle.dump(data, f) 就是把 data 写进 f 这个文件里
        pickle.dump({"bm25": bm25, "docs": child_docs}, f)
    print(f"BM25 索引已保存：{BM25_PATH}")
    return bm25, child_docs


def load_bm25() -> tuple:
    with open(BM25_PATH, "rb") as f:# "rb" 二进制读取（和存时的 "wb" 对应）
        data = pickle.load(f) # 把文件里的内容还原成 Python 对象
    return data["bm25"], data["docs"]# 取出字典里的两个值返回


def save_parent_map(parent_map: dict[str, Document]):#存父文档
    with open(PARENT_MAP_PATH, "wb") as f:
        pickle.dump(parent_map, f)
    print(f"父文档映射已保存：{PARENT_MAP_PATH}")


def load_parent_map() -> dict[str, Document]:
    with open(PARENT_MAP_PATH, "rb") as f:
        return pickle.load(f)
# 三个函数的规律：存用 "wb" + dump，读用 "rb" + load，固定搭配。

def index_exists() -> bool:
    return FAISS_PATH.exists() and BM25_PATH.exists() and PARENT_MAP_PATH.exists()


def build_all_indexes(child_docs: list[Document], parent_map: dict[str, Document]):
    """一次性构建所有索引并保存，返回 (vectorstore, bm25, child_docs, parent_map)"""
    vectorstore = build_faiss(child_docs)
    bm25, docs = build_bm25(child_docs)
    save_parent_map(parent_map)
    return vectorstore, bm25, docs, parent_map


def load_all_indexes():
    """加载已有索引"""
    print("加载已有索引...")
    vectorstore = load_faiss()
    bm25, child_docs = load_bm25()
    parent_map = load_parent_map()
    print(f"索引加载完成（{len(child_docs)} 个子文档，{len(parent_map)} 个菜谱）")
    return vectorstore, bm25, child_docs, parent_map


if __name__ == "__main__":
    from data_preparation import load_recipes, split_recipes

    docs = load_recipes()
    child_docs, parent_map = split_recipes(docs)

    if index_exists():
        print("索引已存在，跳过构建")
        vectorstore, bm25, child_docs, parent_map = load_all_indexes()
    else:
        vectorstore, bm25, child_docs = build_all_indexes(child_docs, parent_map)

    print(f"\nFAISS 向量库：{vectorstore.index.ntotal} 个向量")
    print(f"BM25 文档数：{len(child_docs)}")
    print(f"父文档数量：{len(parent_map)}")




# vectorstore 是一个 FAISS 类的实例对象。

# 里面主要包含：


# vectorstore.index          # FAISS 的向量索引（C++底层，存所有1848个向量）
# vectorstore.docstore       # 文档存储（存每个向量对应的 Document 内容）
# vectorstore.index_to_docstore_id  # 向量编号 → 文档ID 的映射
# 但实际使用时不需要碰这些内部属性，只用两个方法：


# vectorstore.similarity_search(query, k=6)  # 检索最相似的k个文档
# vectorstore.save_local(path)               # 存到本地
#  resp 是 API 返回的一个对象，不是普通列表，结构长这样：


# resp.data = [
#     EmbeddingObject(embedding=[0.123, -0.456, 0.789, ...]),  # 第1条文本的向量
#     EmbeddingObject(embedding=[0.234, -0.567, 0.890, ...]),  # 第2条文本的向量
#     ...  # 共10个（一批10条）
# ]


# tokenized = [
#     ["水","煮","鱼","是",...],   # 第0个子文档的所有字
#     ["必","备","原","料",...],   # 第1个子文档的所有字
#     ["操","作","步","骤",...],   # 第2个子文档的所有字
#     ...  # 共1848个
# ]
# BM25Okapi 拿到 tokenized（1848个字符列表）之后，内部做统计计算：

# 每个字在整个语料里出现多少次
# 每个字在哪些文档里出现过
# 每个文档有多长

# bm25 是一个对象，是 BM25Okapi 类的实例。

# 里面存的是建索引时统计好的数据，主要有：


# bm25.corpus_size    # 总文档数（1848）
# bm25.avgdl          # 平均文档长度（所有子文档的平均字数）
# bm25.doc_freqs      # 每个字出现在几个文档里
# bm25.idf            # 每个字的 IDF 值（越稀有的字 IDF 越高）
# bm25.doc_len        # 每个文档的长度
# 以及最重要的一个方法：


# bm25.get_scores(query_tokens)  # 传入查询词，返回每个文档的相关分数


#  extend 和 append 的区别：


# vectors = [向量A, 向量B]

# # append：把整个列表作为一个元素加进去
# vectors.append([向量C, 向量D])
# # → [向量A, 向量B, [向量C, 向量D]]  ← 嵌套了

# # extend：把列表里每个元素逐个加进去
# vectors.extend([向量C, 向量D])
# # → [向量A, 向量B, 向量C, 向量D]   ← 平铺，正确


# # parent_map：一个字典，里面存了 365 个键值对（365道菜）
# {"水煮鱼": Doc, "红烧肉": Doc, ...}

# # bm25.pkl：一个字典，里面存了 2 个键值对（bm25对象 和 child_docs）
# {"bm25": bm25对象, "docs": child_docs}
# parent_map 读出来直接就能用，bm25.pkl 读出来还要再用 data["bm25"] 和 data["docs"] 把这两个东西分别取出来。