@echo off
chcp 65001 >nul
title 多模态RAG服务启动器
echo ============================================
echo    多模态RAG系统 - 服务启动脚本
echo ============================================
echo.

REM 设置工作目录
cd /d "D:\Multimodal_RAG\src"

echo [1/3] 正在检查 conda 环境...
conda --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 conda，请确保 Anaconda/Miniconda 已安装
    pause
    exit /b 1
)
echo [OK] conda 已找到
echo.

echo [2/3] 正在激活 rag 虚拟环境...
call conda activate rag
if errorlevel 1 (
    echo [错误] 无法激活 rag 环境，请检查环境是否存在
    echo 可用环境列表：
    conda env list
    pause
    exit /b 1
)
echo [OK] rag 环境已激活
echo.

echo [3/3] 正在启动 FastAPI 服务...
echo.
echo ============================================
echo    服务启动成功！
echo    API地址: http://localhost:8000
echo    文档地址: http://localhost:8000/docs
echo ============================================
echo.
echo 按 Ctrl+C 停止服务
echo.

uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
