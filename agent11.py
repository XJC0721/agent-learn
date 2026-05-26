# 长期记忆：用 text embedding 把记忆存成向量，可跨 thread 持久化检索
# 和 agent10 的区别：短期记忆只活在单次 thread，长期记忆通过向量相似度跨会话召回
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass

EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024

# 加载模型配置
_ = load_dotenv()

# 用于获取 text embedding 的接口
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    base_url=os.getenv("DASHSCOPE_BASE_URL", ""),
)

# 加载模型（对话用 DeepSeek）
model = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
    model="deepseek-chat",
    temperature=0.7,
)


# embedding 生成函数
# client.embeddings.create() 是 OpenAI SDK 自带的方法，专门用来调向量化接口。   
#  ├─ model=EMBED_MODEL      → 告诉阿里云用哪个向量模型
    # ├─ input=texts            → 要转换的文字列表
    # └─ dimensions=EMBED_DIM   → 要输出多少维的向量
def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIM,
    )

    # print(response)
    return [item.embedding for item in response.data]


# 测试能否正常生成 text embedding
texts = [
    "LangGraph的中间件非常强大",
    "LangGraph的MCP也很好用",
]
vectors = embed(texts)

print(len(vectors), len(vectors[0]))
# len(vectors)	外层列表有几个元素 = 输入了几句话	2
# len(vectors[0])	第一句话的向量有几个数字 = 维度	1024

# ── 直接读写长期记忆 ──────────────────────────────────────────

# 创建向量数据库，用 embed() 做向量化
store = InMemoryStore(index={"embed": embed, "dims": EMBED_DIM})

# 添加两条用户数据namespace	
# 分类/文件夹	图书馆的"用户"书架
# key	这条记忆的唯一ID	书的编号
# data	实际存的内容	书里的内容

# 括号是什么： 这是 Python 的元组（tuple），和列表 [] 类似，也是装多个元素的容器，但用圆括号。

# 为什么要有逗号： 这是 Python 的特殊规定，元组只有一个元素时，必须加逗号，否则括号只是"括号"而不是元组：

# ("users",)   # ✅ 元组，一个元素
# ("users")    # ❌ 只是个括号，等于 "users" 字符串
# 为什么 namespace 要用元组而不是字符串：

# 元组可以多层嵌套，比如：

# ("users", "admin")    # users 下面的 admin 子分类
# ("users", "normal")   # users 下面的 normal 子分类
# 设计成元组是为了支持多级分类，现在只用一级所以只有一个元素。
namespace = ("users", )
key = "user_1"
store.put(
    namespace,
    key,
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "rule_id": "3",
    },
)

store.put(
    ("users",),  # Namespace to group related data together (users namespace for user data)
    "user_2",    # Key within the namespace (user ID as key)
    {
        "name": "John Smith",
        "language": "English",
    }  # Data to store for the given user
)

# get the "memory" by ID
item = store.get(namespace, "a-memory")

# search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
items = store.search(
    namespace, filter={"rule_id": "3"}, query="language preferences"
)

print(items)
