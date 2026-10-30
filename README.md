# 青蔬：菜价与采购知识助手

源码已统一放在 **[src](src/)**，测试在 `src/tests/`；按顺序阅读完整代码见 **[全部代码.md](全部代码.md)**。

一个适合实习项目展示的小型 Python RAG 应用。加载 TXT / Markdown / 文字型 PDF，读取页码和章节，切分、调用真实 Embedding，比较 BM25、向量和 RRF 混合检索，依据证据回答；通过真正的 MCP stdio 工具读取按日期保存的教学菜价。

**内置采购规则、市场名称和价格都是作者自编教学示例，不是行业标准或实时行情。**不提供现实价格预测。项目定位为单机、小语料的学习作品，没有验证生产并发、独立业务数据上的泛化效果或完整的语义引用校验。

## 先看这些文件

- [实习面试宝典](实习面试宝典.md)：介绍话术、流程、算法、实际问题与追问。
- [原项目改写对照](docs/原项目改写对照.md)：如何从基金助手收敛成当前项目。
- [实测报告](reports/评测结果说明.md)：指标定义、结果、逐题失败分析及限制。
- [验收记录](reports/验收记录.json)：区分真实接口、离线测试和未验证范围。

## 能力与对应代码

| 能力 | 入口 |
|---|---|
| 发现与加载文件 | `documents.document_paths()` |
| 读取 Markdown 标题、TXT、PDF 物理页码 | `documents.read_document()` |
| 章节内字符切分、重叠、来源标识 | `documents.split_documents()` |
| 真实向量生成、归一化、索引落盘 | `retrieval.Embedder`、`create_index()`、`build_index()` |
| BM25 / 向量 / RRF 混合检索 | `retrieval.Retriever.search()` |
| 证据编号、模型回答、引用编号检查 | `qa.answer_question()`、`generate()` |
| 真正 MCP 客户端/服务端 | `src/mcp_client.py`、`src/mcp_server.py` |
| 三种方案共享语料与评测题 | `src/evaluate.py` |
| 上传、问答、比较、菜价查询、原文预览 | `src/app.py`，Streamlit WebUI |

采用固定问答流水线。小规模索引使用 NumPy 精确扫描，避免额外部署数据库和服务；也不引入通用 Agent、审批流、个人画像或分布式任务系统。

```text
离线：文件 → 读取章节/页面 → 切分 → 真实Embedding → index.npz
问答：问题 → BM25 / 向量 / RRF混合 → Top-3原文 → 模型 → 来源编号检查
菜价：WebUI → MCP客户端 → stdio JSON-RPC → 独立工具进程 → prices.csv
评测：同一语料、题目、Top-k → 三方法检索 → Hit / Recall / MRR / 延迟
```

## 本机启动

这里指当前已配置的工作目录。可分享源码包不包含虚拟环境、私有配置和运行索引；从源码包安装请先执行“新电脑安装”。

本目录已准备独立 `.venv`。双击 `启动.cmd`，或执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m streamlit run src/app.py
```

打开 `http://127.0.0.1:8512`。关闭终端或按 Ctrl+C 停止服务。本版与原基金助手使用不同端口。

当前本机已为新版独立配置 `.env`，只复用已有配置中与模型相关的字段。打包文件不包含这份私有配置；在新电脑或从源码包安装时，按下文单独填写 `.env`。

## 新电脑安装

建议 Python 3.12，已验证版本为 3.12.2。在项目根目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

如果希望复现本次具体依赖，使用 `requirements.lock.txt`。锁文件是 Windows / Python 3.12 环境参考；其他平台仍需验证。

填写聊天服务的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`，以及真实文本向量模型的 `EMBEDDING_MODEL`。Embedding 需要不同服务时，再填写 `EMBEDDING_API_KEY` 与 `EMBEDDING_BASE_URL`。BASE_URL 应是服务根路径，例如以 `/v1` 结尾，代码会追加 `/embeddings` 和 `/chat/completions`。

已有配置也可显式指定 `MODEL_ENV_FILE=已有配置的绝对路径`。本项目自己的非空字段优先，操作系统环境变量优先级最高。不要把实际密钥提交到 Git 或发到聊天中。

本次真实验收使用 `qwen3.7-text-embedding-flash`，得到 1024 维向量；它需要相应服务支持，**不是通用服务都可用的模型名**。更换模型或服务后必须重建索引，维度相同也不代表向量空间相同。聊天服务若需要额外参数，可以使用 `LLM_EXTRA_JSON`，不要用它覆盖消息或模型名。

随后启动页面，到“资料与索引”点击“建立 / 更新索引”，或命令行执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/retrieval.py
```

Embedding 会把文档片段发送给配置的服务。模型回答会把 Top-3 证据发送给聊天服务，可能产生相应 API 用量。不要上传不适合发送给该服务的资料。

## 四个页面怎么演示

1. **资料与索引**：选一份 MD 或示例 PDF，展开读取与切分结果；上传自己的资料后重建索引。
2. **知识问答**：提问“周转筐的重量要计入净重吗？”，点击来源核对原文。关闭生成开关时只展示检索原文，界面明确标记未调用生成模型。
3. **方法对比**：对同一道问题比较三种结果；勾选生成选项才额外调用聊天模型三次。数字分数不是答案正确概率。
4. **菜价工具**：查询“番茄、2026-10-26”，展示两个演示市场的记录、日期与 MCP 元数据；查询 2026-10-27 则无记录，不自动换日。

MCP 工具有 `get_prices(vegetable, date)` 和 `search_documents(question, method)`；没有城市筛选或真实行情接口。`stdio` 通过子进程标准输入输出通信，普通业务代码仍保持在 `src/prices.py` 和 `src/retrieval.py`。

```powershell
# 发现工具
.\.venv\Scripts\python.exe -X utf8 src/mcp_client.py
# 真实MCP查询
.\.venv\Scripts\python.exe -X utf8 src/mcp_client.py --vegetable 番茄 --date 2026-10-26
```

外部 MCP 客户端的配置模板见 `mcp_config.example.json`，需替换为实际项目与虚拟环境绝对路径。

## 比较方法与评测

```powershell
.\.venv\Scripts\python.exe -X utf8 src/evaluate.py
# 只评预先分组的test部分
.\.venv\Scripts\python.exe -X utf8 src/evaluate.py --split test
# 可选：多次重复测延迟，会增加Embedding调用
.\.venv\Scripts\python.exe -X utf8 src/evaluate.py --repeats 3
# 可选：额外生成回答，供人工按评分表核对
.\.venv\Scripts\python.exe -X utf8 src/evaluate.py --generate
```

默认报告写入 `reports/retrieval_report.json`。`--output -` 仅将 JSON 输出到终端，方便在只读环境保存验收结果。默认只计算检索指标，不会给生成回答自动打“正确率”。

题集为作者自编的 30 道小样本：dev 6 道；test 24 道中 20 道有答案、4 道无答案。它不是独立盲测集。证据按文件与章节标注，重叠块不会重复增加证据召回数量。无答案题不计算检索 Hit/Recall/MRR；其拒答表现需另外核对。

## 测试与限制

完整测试需先完成建库；只想运行不调用外部服务的核心回归，可以加上 `-p test_core.py`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s src/tests -v
```

单元测试使用假向量隔离外部依赖；真实评测报告使用实际 Embedding 接口，两者不可混称。MCP 测试实际启动工具进程。WebUI 测试使用 Streamlit AppTest，关闭生成模型后检查真实页面交互。

单文件最多 10 MB、PDF 最多 200 页。扫描 PDF、空白 PDF 页会明确提示，暂未接入 OCR 或复杂表格解析。单块默认 450 字符、重叠 70 字符，属于字符预算，不宣称已精确计算模型 token。建库前后检查资料哈希；更新、删除、切分参数或 Embedding 服务/模型变化后需要重建。索引单文件先写临时文件，再替换旧文件；首版没有并发建库锁，因此应由一位本机用户顺序操作。

引用检查只校验本次来源编号是否存在，不能证明每句话被证据支持。未完成全量生成质量评分、拒答率统计、多人认证、生产压测、知识图谱或微调。欢迎在已有失败案例上逐项扩展。

参考实现：用户提供的《美股指数基金助手_最终版》的流程、BM25 公式与资料追溯方式。技术文档：[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)、[NumPy](https://numpy.org/doc/)、[Streamlit](https://docs.streamlit.io/)、[pypdf](https://pypdf.readthedocs.io/)。
