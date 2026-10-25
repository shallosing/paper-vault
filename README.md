# Paper Vault API

一个面向实验室论文与课程资料的轻量检索问答助手。项目用 FastAPI、SQLite 和 Python 标准库实现，不需要数据库服务。

## 已实现功能

- 上传、查看、读取和删除 `.txt` / `.md` 资料
- UTF-8 与 UTF-16 文本兼容
- SQLite 持久化资料元数据与文本块索引
- 关键词搜索：`GET /search?q=关键词`
- 文本切分与零依赖检索：`GET /retrieve?q=问题`
- 两阶段关键词检索与启发式重排：`GET /retrieve?q=问题&top_k=3`
- 离线检索评估：`POST /evaluate`
- 带可追溯引用的问答：`GET /ask?q=问题`
- 浏览器问答页面：`GET /`；交互式 API：`GET /docs`
- 请求 ID、耗时与错误日志：`logs/paper_vault.log`
- 可选 OpenAI 兼容模型生成；未配置 API 时明确返回检索证据，不伪造模型回答

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
fastapi dev app\main.py
```

打开 `http://127.0.0.1:8000/docs` 测试接口。

## 检索评估

评估需要人工标注“每个问题对应哪些相关文档”。先调用 `GET /documents`，复制真实文档 `id`，再在 `/docs` 的 `POST /evaluate` 中提交：

```json
{
  "k": 3,
  "use_reranker": true,
  "cases": [
    {
      "question": "RAG 检索有哪些关键步骤？",
      "relevant_document_ids": ["替换成 GET /documents 返回的文档 ID"]
    }
  ]
}
```

响应包含 Hit@K、Recall@K、Precision@K 和 MRR，以及逐题结果。把 `use_reranker` 改为 `false`，用同一批问题再评一次，就能比较重排前后的指标。建议先准备 10–30 个真实问题及相关文档标注；样本少时，指标只用于发现明显退化，不代表模型的普遍效果。

## 重排与引用怎么读

检索先用关键词/中文二元词对所有文本块打分，再保留候选集；默认重排综合词覆盖率、词频和整句命中。它是无需模型和额外依赖的教学基线，不是语义向量检索，也不是神经 cross-encoder，可能无法理解同义表达。

`/ask` 的每个引用会返回文档 ID、文件名、文本块序号、原文字符起止位置和 `source_ref`（例如 `doc:<文档ID>#chunk=2`）。字符位置是 Python 字符索引，便于从原始文本定位；回答里的 `[1]` 对应响应 `citations` 中 `citation_id: 1`。当前仍应人工抽查模型引用是否确实支持结论。

## 日志和简单前端

- 浏览器打开 `http://127.0.0.1:8000/` 可直接提问并查看证据片段。HTML、CSS、JavaScript 分别放在 `app/static/` 的独立文件中。
- 每个响应带 `X-Request-ID`；服务记录请求路径、状态码和耗时。未处理异常会记录 traceback，可用响应头中的 ID 对照日志。
- 日志按大小轮转，保留在 `logs/`；此目录已加入 `.gitignore`，不提交运行日志。
- `/docs` 保留完整 API 调试界面，网页首页是用于演示的极简单页，不是独立前端工程。

## 问答模式

默认不配置模型时，`/ask` 返回 `mode: "retrieval_only"`，其中 `citations` 是支持答案的资料片段。

若要接入 OpenAI 兼容接口，在 PowerShell 当前窗口设置：

```powershell
$env:LLM_API_KEY = "你的密钥"
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:LLM_MODEL = "deepseek-chat"
```

重启服务后，`/ask` 将返回 `mode: "llm"`。不要把密钥写入代码、README 或 Git 仓库。

## RAG v1 架构

```text
上传文本 -> SQLite 元数据 + 文本切分 -> chunks 索引
问题 -> 关键词/中文二元词召回 -> 候选片段
候选片段 -> 启发式重排 -> Top-K + 原文位置引用
Top-K -> 可选大模型 -> 带 [1]、[2] 来源编号的回答
标注问题集 -> Hit@K / Recall@K / Precision@K / MRR
```

下一阶段可给评估集增加更多问题，再将 `retrieve()` 中的零依赖词频召回替换为 embedding + 向量数据库，并使用独立的 cross-encoder 重排器对比同一评估集。
