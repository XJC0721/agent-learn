import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

# 路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "dishes"
INDEX_DIR = BASE_DIR / "vector_index"
INDEX_DIR.mkdir(exist_ok=True)  #创建文件夹，如果文件夹已经存在，不报错，直接跳过。

# 模型
EMBED_MODEL = "text-embedding-v4"
EMBED_DIMS = 1024
LLM_MODEL = "deepseek-chat"

# 检索参数
FAISS_TOP_K = 6
BM25_TOP_K = 6
FINAL_TOP_K = 4  #融合排名

# 菜品分类中英文映射
CATEGORY_MAP = {
    "aquatic": "水产",
    "breakfast": "早餐",
    "condiment": "调料",
    "dessert": "甜品",
    "drink": "饮品",
    "meat_dish": "肉菜",
    "semi-finished": "半成品",
    "soup": "汤",
    "staple": "主食",
    "vegetable_dish": "素菜",
}
