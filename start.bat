@echo off
rem 本地考古学文献梳理 AI-Agent 一键启动
rem 自动检测 Python：优先用当前虚拟环境，否则用系统 python
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON=python
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 Python，请安装 Python 3.12+ 并加入 PATH
    pause
    exit /b 1
)

echo ============================================
echo   考古学文献图谱 · 3D 关系图服务
echo   http://127.0.0.1:8000
echo   按 Ctrl+C 停止服务
echo ============================================
echo.

"%PYTHON%" -m src.cli serve

pause
