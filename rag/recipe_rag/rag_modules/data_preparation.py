# 数据准备：加载 365 个菜谱 md 文件 → 父子文档切分
#
# 父子文档策略（来自 rag06 句子窗口检索的进阶版）：
#   - 子文档（child）：切成 300 字小块，用于 FAISS/BM25 精准检索
#   - 父文档（parent）：完整菜谱，检索命中子块后换回父文档传给 LLM
#   好处：检索精准（小块），上下文完整（大块），两全其美
import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
import os
# 插入到搜索列表第0位，保证我们写的config能被识别
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, CATEGORY_MAP


def load_recipes() -> list[Document]:
    """加载所有菜谱文件，每个文件 = 一个 Document，带 name/category metadata"""
    docs = []
    # 递归搜索所有md文件rglob递归所有子文件夹
    for md_file in DATA_DIR.rglob("*.md"):
        category_en = md_file.parent.name
        if category_en == "template":   # 跳过模板目录
            continue
        # 第二个参数也传 category_en 本身，找不到就保留英文原名
        category = CATEGORY_MAP.get(category_en, category_en)

#md_file = Path("d:/code/trycc/rag/recipe_rag/data/dishes/aquatic/水煮鱼.md")
# md_file.stem    # → "水煮鱼"      只要文件名，去掉 .md
        dish_name = md_file.stem

        content = md_file.read_text(encoding="utf-8")
        # 临时去掉首尾空白，判断是不是空文件
        if not content.strip():
            continue

        docs.append(Document(
            page_content=content,
            metadata={
                "name": dish_name,
                "category": category,
                "source": str(md_file),
            },
        ))

    print(f"已加载 {len(docs)} 个菜谱")
    return docs


def split_recipes(docs: list[Document]) -> tuple[list[Document], dict[str, Document]]:
    """
    父子文档切分：
    - 返回 child_docs：用于建索引，每块带 parent_id，存的是切分开的一段一段
    - 返回 parent_map：{dish_name: parent_doc}，检索后用 parent_id 取完整菜谱，存名字和原文
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        # 若用第一级切分还是超出300 则在使用第二级切分
        separators=["\n## ", "\n### ", "\n\n", "\n", "。"],
    )

    child_docs: list[Document] = []
    parent_map: dict[str, Document] = {}

    for doc in docs:
        name = doc.metadata["name"]
        parent_map[name] = doc   # 父文档：完整内容存入字典
        # 按 separators 里规定的顺序找切割点
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            child_docs.append(Document(
                page_content=chunk,
                # 就是把旧 metadata 里的所有字段复制过来，再追加两个新字段，合并成一个新字典。
                metadata={**doc.metadata, "parent_id": name, "chunk_index": i},
            ))

    print(f"子文档切片：{len(child_docs)} 个（平均每菜谱 {len(child_docs)//len(docs)} 块）")
    return child_docs, parent_map

# 单独测试这个文件
if __name__ == "__main__":
    docs = load_recipes()
    child_docs, parent_map = split_recipes(docs)

    print(f"\n示例子文档（水煮鱼 第0块）：")
    sample = next((d for d in child_docs if d.metadata["name"] == "水煮鱼" and d.metadata["chunk_index"] == 0), None)
    if sample:
        print(f"  内容：{sample.page_content[:100]}...")
        print(f"  metadata：{sample.metadata}")


# parent_map（字典）：


# {
#     "水煮鱼": Document(
#         page_content="# 水煮鱼的做法\n\n水煮鱼是一道...",  # 完整菜谱原文
#         metadata={
#             "name": "水煮鱼",
#             "category": "水产",
#             "source": "d:/.../.../水煮鱼.md",
#         }
#     ),
#     "红烧肉": Document(...),
#     ...  # 共 365 个
# }

# child_docs（列表）：


# [
#     Document(
#         page_content="# 水煮鱼的做法\n\n水煮鱼是一道...",  # 一小块文本（≤300字）
#         metadata={
#             "name": "水煮鱼",
#             "category": "水产",
#             "source": "d:/.../.../水煮鱼.md",
#             "parent_id": "水煮鱼",    # ← 比parent多这两个
#             "chunk_index": 0,         # ← 第几块
#         }
#     ),
#     Document(page_content="## 必备原料...", metadata={..., "parent_id": "水煮鱼", "chunk_index": 1}),
#     Document(page_content="## 操作...",    metadata={..., "parent_id": "水煮鱼", "chunk_index": 2}),
#     ...  # 共 1848 个
# ]
# 差异就一点：child 比 parent 的 metadata 多了 parent_id 和 chunk_index 两个字段。