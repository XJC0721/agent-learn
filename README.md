# agent-learn

LangChain / LangGraph / RAG 学习笔记，记录从零开始学习 AI Agent 与 RAG 应用开发的过程。

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
| agent08.py | 短期记忆基础，InMemorySaver checkpointer，同一 thread 内多轮对话记忆 |
| agent09.py | 短期记忆进阶，StateGraph 手动构建带记忆的对话图 |
| agent10.py | create_agent 简洁写法，用 checkpointer 实现短期记忆，替代手动 StateGraph |
| agent11.py | 长期记忆，text-embedding-v4 向量化文本，InMemoryStore 存储和语义检索跨会话记忆 |
| agent12.py | 长期记忆进阶，用 @tool + ToolRuntime 让 Agent 通过工具读取 store |
| agent13.py | 长期记忆写入，用工具让 Agent 从对话中提取信息主动存入 store |
| agent14.py | 上下文工程-State，@dynamic_prompt 根据消息数量动态修改 system prompt |
| agent15.py | 上下文工程-Store，从长期记忆读取用户偏好动态修改 system prompt |
| agent16.py | 上下文工程-Runtime，根据用户角色和环境动态控制工具权限 |
| agent17.py | 上下文工程-消息列表，@wrap_model_call 将本地文件内容注入消息列表 |
| agent18.py | 上下文工程-工具中使用上下文，SqliteStore 持久化存储，ToolRuntime 读取 store |
| agent19.py | 上下文工程-工具中使用上下文进阶，ToolRuntime[Context] 动态控制工具返回哪个字段 |
| agent20.py | 压缩上下文，SummarizationMiddleware 自动将旧消息压缩成摘要，避免上下文腐坏 |
| agent21.py | MCP 客户端，MultiServerMCPClient 同时连接天气和算数两个 MCP Server |
| mcp_server/get_weather_mcp/ | MCP Server - 天气工具，HTTP 方式运行在 8000 端口 |
| mcp_server/math_mcp/ | MCP Server - 算数工具，stdio 方式按需启动 |
| agent22.py | 多智能体-监督者模式（tool-calling），子 Agent 包装成工具传给 supervisor |
| agent23.py | 多智能体-监督者模式（langgraph-supervisor），create_supervisor 直接接收 Agent 列表 |
| agent24.py | 并行-节点并行，两个节点同时从 START 出发，验证并行执行效果 |
| agent25.py | 并行-Map-reduce，Send 动态分发角色并行生成回复，汇总选出最佳 |
| agent26.py | 深度Agent，deepagents + DuckDuckGo 实现多轮自主搜索，整合网络信息生成报告 |
| simple_agent.py | LangGraph CLI 调试用 Agent，配合 langgraph.json 启动本地可视化调试页面 |
| langgraph.json | LangGraph CLI 配置文件，指定 Agent 入口和环境变量路径 |
| db_agent_rag.py | 向量库知识智能体，txt文件按空行切分后向量化存入Chroma，Agent检索回答 |
| files/ | 知识库目录，存放待向量化的 txt 文件 |

### RAG 系列

| 文件 | 内容 |
|------|------|
| rag/rag01.py | 四步基础 RAG：TextLoader 加载 → 切片 → InMemoryVectorStore → 检索 → LLM 回答 |
| rag/rag02.py | 文档加载器对比：TextLoader、DirectoryLoader、PyPDFLoader 三种加载方式 |
| rag/rag03.py | 切片策略：CharacterTextSplitter、RecursiveCharacterTextSplitter、代码专用切分 |
| rag/rag04.py | 语义分块：SemanticChunker 按向量距离判断切分点，支持中文优化正则 |
| rag/rag05.py | 向量数据库 FAISS：from_documents 建库、save_local/load_local 持久化、增量追加 |
| rag/rag06.py | 句子窗口检索：单句索引保证检索精准，metadata 存窗口文本扩展上下文给 LLM |
| rag/rag07.py | 结构化索引：存文档时打 metadata 标签，检索时用 filter 先过滤再向量搜索 |
| rag/rag08.py | 混合检索：BM25 稀疏（关键词精准）+ FAISS 密集（语义理解），手动 RRF 融合排名 |
| rag/rag09.py | 自查询检索：LLM 从自然语言中自动提取 metadata filter，拆分为语义查询+结构化过滤条件 |
| rag/rag10.py | LLM 路由：LLM 分类问题意图（tech/biz），路由到对应知识库检索回答 |
| rag/rag11.py | 向量路由：路由描述提前向量化，用余弦相似度匹配最近路由，不调 LLM，速度快成本低 |
| rag/rag12.py | 重排序（RankLLM）：两阶段检索，FAISS 召回 Top-6 后 LLM 精排取 Top-3 |
| rag/rag13.py | 格式化生成（PydanticOutputParser）：BaseModel 定义结构，自动生成格式指令，自动解析+类型验证 |
| rag/rag14.py | Function Calling 底层原理：两次 API 调用，模型决策调哪个工具，代码执行后结果发回模型生成最终答案 |
| rag/recipe_rag/ | 菜谱问答系统（完整 RAG 应用）：365 道菜谱，父子文档策略（子文档索引 + 完整菜谱回传 LLM），混合检索（BM25 关键词 + FAISS 语义）+ RRF 融合排名，LLM 查询路由（列表/详情/通用三种策略），流式输出 |
| rag/files/ | RAG 知识库目录，存放 md/pdf/faiss 等文件 |

## 技术栈

- [LangChain](https://python.langchain.com/) — Agent 创建、工具定义、中间件
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 状态图、检查点、Runtime
- [DeepSeek API](https://platform.deepseek.com/) — 大模型服务
- [阿里云百炼 DashScope](https://dashscope.aliyuncs.com/) — 向量化模型（text-embedding-v4）

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
