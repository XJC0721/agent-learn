# 菜谱 RAG 系统 - 主入口
# 首次运行：加载 365 个菜谱 → 向量化 → 建索引（约 2-3 分钟）
# 之后运行：直接加载已有索引（秒级）
import sys
sys.stdout.reconfigure(encoding='utf-8')

from rag_modules.data_preparation import load_recipes, split_recipes
from rag_modules.index_construction import index_exists, build_all_indexes, load_all_indexes
from rag_modules.generation_integration import ask


def build_index_if_needed():
    """首次运行时构建索引，已有则直接加载"""
    if index_exists():
        return load_all_indexes()

    print("首次运行，开始构建索引...")
    docs = load_recipes()
    child_docs, parent_map = split_recipes(docs)
    return build_all_indexes(child_docs, parent_map)


def main():
    print("=" * 50)
    print("菜谱 RAG 系统")
    print("=" * 50)

    vectorstore, bm25, child_docs, parent_map = build_index_if_needed()
    print(f"\n系统就绪，共收录 {len(parent_map)} 道菜谱")
    print("输入问题开始查询，输入 q 退出\n")

    while True:
        question = input("你：").strip()
        if not question:
            continue
        if question.lower() in ("q", "quit", "exit", "退出"):
            print("再见！")
            break
        ask(question, vectorstore, bm25, child_docs, parent_map)
        print()


if __name__ == "__main__":
    main()
