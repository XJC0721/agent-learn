# 检索优化：混合检索（BM25 + FAISS）+ RRF 融合 + 查询路由
#
# 路由逻辑（来自 rag10/rag11）：
#   list   → "推荐几道菜" "有哪些汤" → 不做全文检索，直接从 parent_map 按分类筛选
#   detail → "红烧肉怎么做" "水煮鱼步骤" → 混合检索 + 父子文档 → 返回完整菜谱
#   general → "做饭需要准备什么" → 混合检索 → 返回相关片段
import sys
import os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, FAISS_TOP_K, BM25_TOP_K, FINAL_TOP_K


llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0,
)


# ── 1. 混合检索 + RRF（来自 rag08） ─────────────────────────────────────────
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (rank + k)


def hybrid_search(
    query: str,
    vectorstore: FAISS,
    bm25: BM25Okapi,
    child_docs: list[Document],
    parent_map: dict[str, Document],
    top_k: int = FINAL_TOP_K,
) -> list[Document]:
    """
    混合检索：BM25 召回 + FAISS 召回 → RRF 融合排名 → 换回父文档

    子文档用于精准定位，父文档（完整菜谱）用于传给 LLM
    """
    # FAISS 检索（语义）
    faiss_results = vectorstore.similarity_search(query, k=FAISS_TOP_K)
    # 第二个参数 "" 是找不到 parent_id 时的默认值
    faiss_ids = [doc.metadata.get("parent_id", "") for doc in faiss_results]

    # BM25 检索（关键词），中文按字符分词
    tokenized_query = list(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    #  把分数从小到大排序，返回的是索引而不是分数，[::-1] → 翻转，变成从大到小，取前6个索引
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
    # 用这6个索引去 child_docs 里取出对应子文档，再拿它的 parent_id（菜名）
    bm25_ids = [child_docs[i].metadata.get("parent_id", "") for i in bm25_top_indices]

    # RRF 融合：对同一 parent_id 的分数累加，分别算出faiss的分数和bm25的分数相加
    score_map: dict[str, float] = {}
    for rank, pid in enumerate(faiss_ids):
        if pid:
            score_map[pid] = score_map.get(pid, 0) + rrf_score(rank)
    for rank, pid in enumerate(bm25_ids):
        if pid:
            score_map[pid] = score_map.get(pid, 0) + rrf_score(rank)

    # 按 RRF 分数排序，取 top_k 个父文档
    sorted_ids = sorted(score_map, key=lambda x: score_map[x], reverse=True)[:top_k]
    return [parent_map[pid] for pid in sorted_ids if pid in parent_map]


# ── 2. 查询路由 ─────────────────────────────────────────────────────────────
# 多行字符串
ROUTE_PROMPT = """判断用户问题属于哪种类型，只输出一个词：

- list：用户想知道有哪些菜、推荐菜单、某类菜有哪些（例：推荐几道素菜、有什么汤）
- detail：用户想知道某道具体菜的做法或配料（例：红烧肉怎么做、番茄炒蛋的步骤）
- general：其他问题（例：做饭需要什么工具、怎么腌肉）

用户问题：{question}

只输出 list、detail 或 general 三个词之一："""


def route_query(question: str) -> str:
    """路由：返回 'list' / 'detail' / 'general'"""
    # 把question填入返回发给llm
    response = llm.invoke(ROUTE_PROMPT.format(question=question))
    # 把返回的去掉空格换行 编程小写
    intent = response.content.strip().lower()
    # 容错：如果模型多输出了内容，只取第一个词
    for keyword in ["list", "detail", "general"]:
        if keyword in intent:
            return keyword
    return "general"


# ── 3. 按路由执行检索 ────────────────────────────────────────────────────────
def retrieve(
    question: str,
    vectorstore: FAISS,
    bm25: BM25Okapi,
    child_docs: list[Document],
    parent_map: dict[str, Document],
) -> tuple[str, list[Document]]:
    """
    根据意图选检索策略：
    - list：按 category 筛父文档，返回菜名列表
    - detail/general：混合检索，返回完整菜谱
    """
    intent = route_query(question)
    print(f"  [路由] 意图识别：{intent}")

    if intent == "list":
        # 判断用户想要哪类菜，右边是根据data的中文，左边是猜用户会说什么
        category_keywords = {
            "素菜": "素菜", "蔬菜": "素菜",
            "肉": "肉菜", "荤菜": "肉菜",
            "汤": "汤", "靓汤": "汤",
            "早餐": "早餐",
            "甜品": "甜品", "甜点": "甜品",
            "主食": "主食", "饭": "主食", "面": "主食",
            "水产": "水产", "鱼": "水产", "海鲜": "水产",
        }
        matched_category = None
        # items() 是字典的方法，同时遍历 key 和 value。
        for kw, cat in category_keywords.items():
            if kw in question:
                matched_category = cat
                break

        if matched_category:
            # 有明确分类：按分类过滤，再用混合检索排序
            category_docs = [doc for doc in parent_map.values() if doc.metadata.get("category") == matched_category]
            # 从分类内做检索，取相关度最高的
            results = hybrid_search(question, vectorstore, bm25, child_docs, parent_map, top_k=8)
            results = [doc for doc in results if doc.metadata.get("category") == matched_category] or category_docs[:8]
        else:
            # 没有明确分类：用混合检索找最相关的菜，避免随机抓到不相关的
            results = hybrid_search(question, vectorstore, bm25, child_docs, parent_map, top_k=8)

        return intent, results[:8]

    else:
        # detail / general：混合检索
        results = hybrid_search(question, vectorstore, bm25, child_docs, parent_map)
        return intent, results


if __name__ == "__main__":
    from index_construction import load_all_indexes

    vectorstore, bm25, child_docs, parent_map = load_all_indexes()

    test_questions = [
        "有哪些素菜推荐",
        "红烧肉怎么做",
        "做饭一般需要准备什么工具",
    ]

    for q in test_questions:
        print(f"\n问题：{q}")
        intent, docs = retrieve(q, vectorstore, bm25, child_docs, parent_map)
        print(f"检索到 {len(docs)} 个结果：")
        for doc in docs:
            print(f"  - {doc.metadata['name']}（{doc.metadata['category']}）")


# dict[key] 取的是对应的 value。字典只能用 key 取 value，反过来不行。



# sorted(score_map, ...)
# # score_map 传进去的是字典，sorted 拿到的是所有 key（菜名）
# # ["水煮鱼", "红烧肉", "番茄炒蛋", "清蒸生蚝"]

# key=lambda x: score_map[x]
# # 排序依据：拿菜名 x，用 score_map[x] 取出它的分数来比大小

# reverse=True
# # 分数从高到低排

# [:top_k]
# # 取前4个


# sorted 内部会遍历传进来的东西，拿到每个元素，再按 key 指定的标准排序。

# 你可以理解成：


# sorted(score_map, key=lambda x: score_map[x], reverse=True)

# # 内部大概做了这些事：
# elements = []
# for x in score_map:          # 遍历字典，x 是菜名
#     elements.append((score_map[x], x))  # 记录 (分数, 菜名)

# elements.sort(reverse=True)  # 按分数从高到低排
# result = [x for score, x in elements]  # 只取菜名
# 我们不用写这些，sorted + key 一行搞定。


# # 这是 docstring（放在函数第一行）
# def route_query():
#     """路由：返回 'list' / 'detail' / 'general'"""
#     ...

# # 这是多行字符串（赋值给变量）
# ROUTE_PROMPT = """判断用户问题属于哪种类型..."""


# 有明确分类：先缩小范围，再检索


# "推荐几道汤"
#     ↓ 先定位到汤类
#     ↓ 在全库混合检索8个
#     ↓ 过滤掉非汤类的
#     → 返回汤类菜谱
# 没有明确分类：直接全库检索


# "推荐几道减脂期吃的"
#     ↓ 没有分类关键词，不定位
#     ↓ 直接全库混合检索8个
#     → 返回语义相关的菜谱