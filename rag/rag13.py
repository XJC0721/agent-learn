# 格式化生成（Structured Output）- PydanticOutputParser
# rag09/12 里手写 prompt 说"只输出JSON"，再手动 json.loads() 解析
# 这里用 PydanticOutputParser：自动生成格式指令 + 自动解析 + 类型验证
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

_ = load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


# ── 1. 定义期望的数据结构 ──────────────────────────────────────────────
print("=" * 50)
print("1. 定义数据结构")
print("=" * 50)

# BaseModel 是 Pydantic 的基类，用来定义数据结构和类型约束
# Field 的 description 会直接被 parser 提取出来，作为格式指令发给 LLM
# Field 不是赋值，是给这个字段加描述和约束，后面给llm
class ProductInfo(BaseModel):
    name: str = Field(description="商品名称")
    price: float = Field(description="商品价格，纯数字")
    features: list[str] = Field(description="商品特点列表，3条以内")
    suitable_for: str = Field(description="适合人群")

print("数据结构定义完成：ProductInfo")
print(f"  字段：name(str), price(float), features(list), suitable_for(str)")


# ── 2. 创建解析器，查看自动生成的格式指令 ────────────────────────────
print("\n" + "=" * 50)
print("2. 解析器自动生成的格式指令")
print("=" * 50)

parser = PydanticOutputParser(pydantic_object=ProductInfo)

# get_format_instructions() 自动根据 ProductInfo 的字段和描述生成一段指令
# 这段指令会被注入到 prompt 里，告诉 LLM 应该输出什么格式
format_instructions = parser.get_format_instructions()
print(format_instructions)


# ── 3. 构建 prompt，注入格式指令 ─────────────────────────────────────
print("\n" + "=" * 50)
print("3. 提取商品信息")
print("=" * 50)

extract_prompt = ChatPromptTemplate.from_template(
    """从以下商品描述中提取结构化信息。

{format_instructions}

商品描述：{text}"""
)


def extract_product(text: str) -> ProductInfo:
    # 把格式指令和商品描述填入 prompt
    prompt_value = extract_prompt.format(
        format_instructions=format_instructions,
        text=text,
    )
    response = llm.invoke(prompt_value)

    # parser.parse() 自动完成：去代码块 + json.loads() + Pydantic 类型验证
    # 比 rag09 里手动处理更简洁，且如果类型不对会直接报错
    result = parser.parse(response.content)
    return result


# ── 4. 测试 ───────────────────────────────────────────────────────────
products = [
    "机械键盘Pro，采用青轴设计，手感清脆，RGB背光，售价399元，适合游戏玩家和程序员使用，支持全键无冲。",
    "降噪耳机X1，主动降噪效果出色，续航30小时，支持蓝牙5.3，价格799元，适合经常通勤或在嘈杂环境工作的人。",
]

for text in products:
    print(f"\n原始描述：{text[:30]}...")
    result = extract_product(text)
    print(f"  商品名称：{result.name}")
    print(f"  价格：    {result.price} 元")
    print(f"  特点：    {result.features}")
    print(f"  适合人群：{result.suitable_for}")


# ── 5. 对比：手动解析 vs PydanticOutputParser ─────────────────────────
print("\n" + "=" * 50)
print("对比总结")
print("=" * 50)
print("手动方式（rag09）：")
print("  1. prompt 里手写 '只输出JSON：{...}' ")
print("  2. 手动去掉 ``` 代码块")
print("  3. json.loads() 解析字符串")
print("  4. 无类型验证，age='三十' 不会报错")
print()
print("PydanticOutputParser（rag13）：")
print("  1. 定义 BaseModel，parser 自动生成格式指令")
print("  2. parser.parse() 一步完成解析+类型验证")
print("  3. 字段缺失或类型错误直接抛异常，更安全")
