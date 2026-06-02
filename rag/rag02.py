# 文档加载（Data Loading）
# RAG 的第一步：把各种格式的文档加载成 LangChain 的 Document 对象
# Document 对象统一格式：page_content（正文）+ metadata（来源、页码等）
# 本文件演示三种常用加载方式：TextLoader、DirectoryLoader、PyPDFLoader
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader

files_dir = Path(__file__).parent / "files"


# ── 1. TextLoader：加载单个文本文件 ───────────────────────────────
# 适合：txt、markdown 等纯文本格式，最轻量
print("=" * 50)
print("1. TextLoader（单文件加载）")
print("=" * 50)

loader = TextLoader(str(files_dir / "rag_knowledge.md"), encoding="utf-8")
docs = loader.load()

# 每个文件加载后是一个 Document，page_content 是完整文件内容
print(f"加载文档数: {len(docs)}")
print(f"来源: {docs[0].metadata['source']}")
print(f"内容前100字: {docs[0].page_content[:100]}")


# ── 2. DirectoryLoader：批量加载整个目录 ──────────────────────────
# 适合：知识库有多个文件，一次性全部加载
# glob 参数指定加载哪种格式，"**/*.txt" 表示递归匹配所有 txt 文件
print("\n" + "=" * 50)
print("2. DirectoryLoader（目录批量加载）")
print("=" * 50)

loader = DirectoryLoader(
    str(files_dir),
    glob="**/*.md",                         # 只加载 md 文件
    loader_cls=TextLoader,                  # 用 TextLoader 来读每个文件
    loader_kwargs={"encoding": "utf-8"},    # 传给 TextLoader 的参数
    show_progress=False,
)
docs = loader.load()

print(f"加载文档数: {len(docs)}")
for doc in docs:
    print(f"  - {Path(doc.metadata['source']).name}，{len(doc.page_content)} 字符")


# ── 3. PyPDFLoader：加载 PDF 文件 ─────────────────────────────────
# 适合：PDF 格式文档，按页切分，每页是一个 Document
# metadata 里会自动记录页码（page），方便后续引用来源
print("\n" + "=" * 50)
print("3. PyPDFLoader（PDF加载）")
print("=" * 50)

loader = PyPDFLoader(str(files_dir / "rag_tech.pdf"))
docs = loader.load()

print(f"加载页数: {len(docs)}")
for doc in docs:
    print(f"  第 {doc.metadata.get('page', 0) + 1} 页，{len(doc.page_content)} 字符")
    print(f"  内容前80字: {doc.page_content[:80]}")


# ── 小结 ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("三种加载器对比：")
print("  TextLoader      → 单个文本/markdown 文件")
print("  DirectoryLoader → 批量加载整个目录（可指定格式）")
print("  PyPDFLoader     → PDF 文件，按页分割，带页码 metadata")
print("=" * 50)
