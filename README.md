# agent-learn

LangChain / LangGraph 学习笔记，记录从零开始学习 AI Agent 开发的过程。

## 文件说明

| 文件 | 内容 |
|------|------|
| agent01.py | 基础 ReAct Agent，无工具，最简单的 LLM 对话 |
| agent02.py | ToolRuntime 权限控制，用 Context 限制工具调用权限 |
| agent03.py | 流式输出（Streaming），stream_mode="updates" 逐步推送结果 |
| agent04.py | StateGraph 状态图，手动构建节点和边，实现工具调用循环 |
| agent05.py | 消息截断，防止对话历史超出上下文限制，配合 InMemorySaver 实现多轮记忆 |
| agent06.py | 内容过滤（关键词 Guardrail），before_agent 拦截违禁词 |
| agent07.py | PII 隐私检测，用 LLM 语义识别个人信息，支持拒绝或打码两种处理方式 |

## 技术栈

- [LangChain](https://python.langchain.com/) — Agent 创建、工具定义、中间件
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 状态图、检查点、Runtime
- [DeepSeek API](https://platform.deepseek.com/) — 大模型服务

## 运行环境

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install langchain langgraph langchain-openai python-dotenv

# 配置 API Key（新建 .env 文件）
DEEPSEEK_API_KEY=你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 运行任意示例
python agent01.py
```
