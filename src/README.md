# 源码目录

所有业务和测试 Python 文件统一放在本目录。项目外层保留启动文件、配置、资料、评测结果和面试文档。

建议顺序：settings → documents → retrieval → qa → prices → mcp_server → mcp_client → evaluate → app。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m streamlit run src/app.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s src/tests -v
```

“全部代码.md”是本次整理生成的阅读快照。修改程序应编辑本目录的实际 Python 文件；快照不作为运行入口。
