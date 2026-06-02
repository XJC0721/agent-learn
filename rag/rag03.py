# 文本切片策略（Text Splitting）
# 切片是 RAG 效果的关键：块太大检索不精准，块太小上下文不完整
# 本文件演示三种切分器：CharacterTextSplitter、RecursiveCharacterTextSplitter（含中文优化）、代码专用
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language

files_dir = Path(__file__).parent / "files"
loader = TextLoader(str(files_dir / "rag_knowledge.md"), encoding="utf-8")
docs = loader.load()


# ── 1. CharacterTextSplitter：固定大小分块 ────────────────────────
# 先按 separator（默认 \n\n）分段落，再智能合并：
# 累积长度快超过 chunk_size 时才切断，不是硬切
# chunk_overlap：相邻块重叠字符数，防止语义在边界处断裂
print("=" * 50)
print("1. CharacterTextSplitter（固定大小分块）")
print("=" * 50)

splitter = CharacterTextSplitter(
    chunk_size=300,     # 每块目标大小（字符数）
    chunk_overlap=30,   # 相邻块重叠字符数
    separator="\n\n",   # 优先按空行切分
)
chunks = splitter.split_documents(docs)

print(f"切分为 {len(chunks)} 个块")
print("--- 前3块预览 ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n块{i+1}（{len(chunk.page_content)}字）:")
    print(chunk.page_content[:100], "...")


# ── 2. RecursiveCharacterTextSplitter：递归分块 ───────────────────
# 按优先级递归尝试切分符：\n\n → \n → 句号 → 空格 → 单字符
# 优先保证语义完整，实在太长才往下一级切
# 是 LangChain 推荐的默认切分器，适合大多数场景
print("\n" + "=" * 50)
print("2. RecursiveCharacterTextSplitter（递归分块）")
print("=" * 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
)
chunks = splitter.split_documents(docs)

print(f"切分为 {len(chunks)} 个块")
print("--- 前3块预览 ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n块{i+1}（{len(chunk.page_content)}字）:")
    print(chunk.page_content[:100], "...")


# ── 3. 对比 chunk_size 参数的影响 ─────────────────────────────────
# chunk_size 越小：块越多，检索越精准，但上下文可能不完整
# chunk_size 越大：块越少，上下文完整，但检索精度下降
print("\n" + "=" * 50)
print("3. chunk_size 对切片数量的影响")
print("=" * 50)

for size in [100, 300, 500]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    print(f"  chunk_size={size}：切分为 {len(chunks)} 个块，平均 {sum(len(c.page_content) for c in chunks) // len(chunks)} 字/块")


# ── 4. 中文优化分隔符 ─────────────────────────────────────────────
# 默认分隔符是按英文设计的，中文没有空格分词
# 处理中文文档时额外加入中文标点，确保在句子边界切分
print("\n" + "=" * 50)
print("4. RecursiveCharacterTextSplitter（中文优化）")
print("=" * 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=15,
    separators=[
        "\n\n",     # 段落（最高优先级）
        "\n",       # 换行
        "。",       # 中文句号
        "，",       # 中文逗号
        "，",   # 全角逗号
        "。",   # 表意句号
        " ",        # 空格
        "",         # 单字符（最后手段）
    ],
)
chunks = splitter.split_documents(docs)
print(f"切分为 {len(chunks)} 个块")
print("--- 前2块预览 ---")
for i, chunk in enumerate(chunks[:2]):
    print(f"\n块{i+1}（{len(chunk.page_content)}字）: {chunk.page_content[:80]}...")


# ── 5. 代码专用模式 ───────────────────────────────────────────────
# from_language() 针对编程语言预设分隔符：类定义 → 函数定义 → 控制流
# 保证不会把一个函数切成两半，适合代码库的 RAG
print("\n" + "=" * 50)
print("5. 代码专用分块（from_language）")
print("=" * 50)

code_sample = """
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def clean(self):
        return [x.strip() for x in self.data if x.strip()]

    def process(self):
        cleaned = self.clean()
        result = []
        for item in cleaned:
            if len(item) > 10:
                result.append(item.upper())
        return result

def run_pipeline(raw_data):
    processor = DataProcessor(raw_data)
    return processor.process()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,   # 按 Python 语法结构切分
    chunk_size=100,
    chunk_overlap=10,
)
chunks = splitter.split_text(code_sample)
print(f"切分为 {len(chunks)} 个块")
for i, chunk in enumerate(chunks):
    print(f"\n块{i+1}:\n{chunk}")
