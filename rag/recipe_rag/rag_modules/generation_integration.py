# 生成集成：根据意图和检索结果，用不同 prompt 生成最终回答
#
# list   → 列菜名 + 简短介绍，不需要完整菜谱内容
# detail → 把完整菜谱传给 LLM，回答"怎么做"
# general → 片段拼接，回答通用知识问题
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL


llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,   # 生成回答时稍高一点，更自然
)


# ── 不同意图对应不同 prompt ───────────────────────────────────────────────
LIST_PROMPT = ChatPromptTemplate.from_template(
    """你是一个烹饪助手。根据以下菜谱列表，回答用户的问题。
只需列出菜名和一句话简介，不需要完整做法。

菜谱列表：
{context}

用户问题：{question}

请用简洁的列表格式回答："""
)

DETAIL_PROMPT = ChatPromptTemplate.from_template(
    """你是一个烹饪助手。根据以下菜谱详情，回答用户的问题。
如果菜谱中有完整的原料和步骤，请整理后呈现给用户。

菜谱详情：
{context}

用户问题：{question}

请根据菜谱内容回答："""
)

GENERAL_PROMPT = ChatPromptTemplate.from_template(
    """你是一个烹饪助手。根据以下参考内容，回答用户的烹饪问题。
如果参考内容不足以回答，可以结合你的知识补充。

参考内容：
{context}

用户问题：{question}

请回答："""
)

PROMPT_MAP = {
    "list": LIST_PROMPT,
    "detail": DETAIL_PROMPT,
    "general": GENERAL_PROMPT,
}


def build_context(intent: str, docs: list[Document], question: str = "") -> str:
    """根据意图决定把文档内容怎么组织成 context"""
    if intent == "list":
        # 只传菜名和分类，避免 prompt 太长。（格式化字符串）变成"- 水煮鱼（水产）"
        lines = [f"- {doc.metadata['name']}（{doc.metadata['category']}）" for doc in docs]
        return "\n".join(lines)
    elif intent == "detail":
        if not docs:
            return "未找到相关菜谱"
        # 优先选菜名出现在问题里的文档，避免排名靠前但不相关的文档覆盖精确匹配
        for doc in docs:
            if doc.metadata["name"] in question:
                return doc.page_content
        return docs[0].page_content
    else:
        # general：把检索片段拼接，每个片段截取前 500 字
        parts = []
        for doc in docs:
            name = doc.metadata.get("name", "")
            content = doc.page_content[:500]
            parts.append(f"【{name}】\n{content}")
        return "\n\n".join(parts)


def generate_stream(question: str, intent: str, docs: list[Document]):
    """流式生成：用 chain.stream() 逐块 yield，调用方负责打印"""
    context = build_context(intent, docs, question)
    prompt = PROMPT_MAP.get(intent, GENERAL_PROMPT)
# | 是 LangChain 的管道符，把 prompt 和 llm 串起来，prompt 的输出自动作为 llm 的输入。
    chain = prompt | llm
    for chunk in chain.stream({"question": question, "context": context}):
        # yield 是生成器关键字，和 return 类似，但不是一次性返回全部，而是每次产出一个值
        # 这种一小块的方式实现了流式输出
        yield chunk.content


def ask(
    question: str,
    vectorstore,
    bm25,
    child_docs: list[Document],
    parent_map: dict[str, Document],
) -> str:
    """完整的 RAG 流程：路由 → 检索 → 流式生成"""
    from rag_modules.retrieval_optimization import retrieve

    print(f"\n{'='*50}")
    print(f"问题：{question}")
# 路由和文档
    intent, docs = retrieve(question, vectorstore, bm25, child_docs, parent_map)
    print(f"  [检索] 召回 {len(docs)} 个文档")
    for doc in docs:
        print(f"    - {doc.metadata['name']}（{doc.metadata['category']}）")

    print("\n回答：")
    full_answer = ""
    # yield只是产出 然后chunk接到再打印
    for chunk in generate_stream(question, intent, docs):
        print(chunk, end="", flush=True)   # flush=True 确保每块立即输出，不缓冲
        full_answer += chunk
    print()   # 流式结束后换行
    return full_answer


if __name__ == "__main__":
    from index_construction import load_all_indexes

    vectorstore, bm25, child_docs, parent_map = load_all_indexes()

    test_questions = [
        "推荐几道汤",
        "红烧肉怎么做",
        "腌肉的时候需要注意什么",
    ]

    for q in test_questions:
        ask(q, vectorstore, bm25, child_docs, parent_map)
        print()


# 1
# 菜谱列表：
# - 水煮鱼（水产）
# - 番茄蛋花汤（汤）
# - 红烧肉（肉菜）

# 最终传给 LLM 的 context 长这样：

# 3
# 【水煮鱼】
# 水煮鱼是一道麻辣鲜香...（前500字）

# 【腌制技巧】
# 腌制时需要注意...（前500字）

# 【红烧肉】
# 红烧肉的做法...（前500字）


# 因为三种意图的需求不同，传的内容也不一样：

# list（推荐列表）
# 用户只想知道有哪些菜，不需要做法，传菜名就够了。传完整菜谱反而让 prompt 太长，浪费 token。

# detail（具体做法）
# 用户想知道一道菜怎么做，必须传完整菜谱（原料+步骤），只传菜名 LLM 没法回答做法。而且只传一道菜，传多了 LLM 容易混乱。

# general（通用问题）
# 用户问的是跨菜谱的知识（比如"怎么腌肉"），多个菜谱里都可能有相关内容，所以把几个相关片段都拼进去，让 LLM 综合参考。但每个截500字防止太长。

# chain.stream()
# 和 chain.invoke() 类似，区别是：


# chain.invoke(...)   # 等 LLM 全部生成完，一次性返回
# chain.stream(...)   # LLM 每生成一小块就立刻返回，像打字机一样
# 返回的是一个迭代器，每次循环拿到一小块文字（chunk）。