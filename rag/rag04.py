# 语义分块（Semantic Chunking）
# 不依赖固定字符数或分隔符，而是根据语义变化决定在哪切
# 核心：计算相邻句子的向量距离，距离突变的地方就是切分点
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader

_ = load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)


# ── s自定义 Embedding ─────────────────────────────────────────────
# 文档原版用 HuggingFace 本地模型，这里换成 DashScope API，效果相同
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

# SemanticChunker 默认的句子分割正则只支持英文标点
# 处理中文文本时必须自定义正则，让它能识别中文句号/问号/感叹号
CHINESE_SENTENCE_REGEX = r'(?<=[。！？\n])'

files_dir = Path(__file__).parent / "files"
loader = TextLoader(str(files_dir / "rag_knowledge.md"), encoding="utf-8")
docs = loader.load()

# 语义分块需要文本覆盖多个不同主题，才能检测到明显的语义跳跃
# rag_knowledge.md 全是 RAG 内容，主题单一，语义差距太小，切不动
# 这里用一段跨越多个主题的文本来演示效果
text = """
RAG是一种将外部知识库与大语言模型结合的技术。在生成答案前先检索相关文档，有效减少幻觉问题。
向量数据库存储文本的向量表示，支持高效的语义相似度搜索。常见的向量数据库包括Chroma、Milvus和Faiss。
文本切片策略决定了RAG的检索质量。chunk_size越小检索越精准，但可能丢失上下文完整性。
混合检索结合了向量检索和BM25关键词检索，兼顾语义理解和精确匹配，效果优于单一检索方式。

今天的晚饭做了红烧肉，先把五花肉切块焯水，然后加入生抽老抽和冰糖慢火炖煮两小时。
炒青菜要大火快炒，放少量盐和蒜末，出锅前淋几滴香油，口感更清脆。
西红柿鸡蛋汤是最简单的家常菜，西红柿切块，鸡蛋打散，先炒鸡蛋再加西红柿，加水煮开即可。
煮米饭时水和米的比例是1.2:1，焖15分钟后不要立刻开盖，再等5分钟口感更好。

NBA季后赛进入白热化阶段，湖人队在关键时刻展现出强大的团队配合。
足球欧冠决赛即将举行，皇马和曼城的对决备受关注，球迷们都期待一场精彩的比赛。
网球大满贯赛事温布尔顿开幕，草地赛场上的比拼对球员的体能和技术都是极大考验。
游泳世锦赛上，选手们打破了多项世界纪录，中国队表现出色，摘得多枚金牌。
"""


# ── 1. percentile（百分位法，默认）────────────────────────────────
# 计算所有相邻句子的语义差异值，取第95百分位作为阈值
# 超过阈值的地方就是断点，只有最显著的5%跳跃才会切分
print("=" * 50)
print("1. percentile（百分位法，默认）")
print("=" * 50)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
    sentence_split_regex=CHINESE_SENTENCE_REGEX,
)
chunks = splitter.split_text(text)
print(f"切分为 {len(chunks)} 个块")
for i, chunk in enumerate(chunks):
    print(f"\n块{i+1}（{len(chunk)}字）: {chunk[:80]}...")


# ── 2. standard_deviation（标准差法）─────────────────────────────
# 差异值超过"均值 + N倍标准差"才算断点
# 适合话题切换明显的文章
print("\n" + "=" * 50)
print("2. standard_deviation（标准差法）")
print("=" * 50)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5,
    sentence_split_regex=CHINESE_SENTENCE_REGEX,
)
chunks = splitter.split_text(text)
print(f"切分为 {len(chunks)} 个块")
for i, chunk in enumerate(chunks):
    print(f"  块{i+1}（{len(chunk)}字）: {chunk[:60]}...")


# ── 3. 对比三种方法的切块数量 ─────────────────────────────────────
print("\n" + "=" * 50)
print("3. 三种断点方法对比")
print("=" * 50)

configs = [
    ("percentile",         70,  "差异值前30%才切"),
    ("standard_deviation", 0.5, "超过均值+0.5σ才切"),
    ("interquartile",      0.5, "超过Q3+0.5*IQR才切"),
]
for method, amount, desc in configs:
    sp = SemanticChunker(embeddings, breakpoint_threshold_type=method, breakpoint_threshold_amount=amount, sentence_split_regex=CHINESE_SENTENCE_REGEX)
    c = sp.split_text(text)
    print(f"  {method:<22} → {len(c)} 块  （{desc}）")
