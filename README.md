# Paper Vault API

一个面向实验室论文与课程资料的轻量检索问答助手。项目用 FastAPI、SQLite 和 Python 标准库实现，不需要数据库服务。

## 已实现功能

- 上传、查看、读取和删除 `.txt` / `.md` 资料
- UTF-8 与 UTF-16 文本兼容
- SQLite 持久化资料元数据与文本块索引
- 关键词搜索：`GET /search?q=关键词`
- 文本切分与零依赖检索：`GET /retrieve?q=问题`
- 带来源编号的问答：`GET /ask?q=问题`
- 可选 OpenAI 兼容模型生成；未配置 API 时明确返回检索证据，不伪造模型回答

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
fastapi dev app\main.py
```

打开 `http://127.0.0.1:8000/docs` 测试接口。

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
问题 -> 关键词/中文二元词检索 -> Top-K 片段
Top-K 片段 -> 可选大模型 -> 带 [1]、[2] 来源编号的回答
```

下一阶段可将 `retrieve()` 中的零依赖词频评分替换为 embedding + 向量数据库检索。
