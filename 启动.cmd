@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先按照 README 安装 Python 虚拟环境和依赖。
  pause
  exit /b 1
)
if not exist "storage" mkdir "storage"
if not exist "reports" mkdir "reports"
if not exist "data\uploads" mkdir "data\uploads"
echo 浏览器打开 http://127.0.0.1:8512 ，按 Ctrl+C 停止。
".venv\Scripts\python.exe" -X utf8 -m streamlit run src/app.py
pause
