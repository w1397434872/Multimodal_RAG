@echo off
chcp 65001 >nul
title 多模态RAG系统 - 一键启动
echo ============================================
echo    多模态RAG系统 - 一键启动脚本
echo ============================================
echo.

cd /d "D:\Multimodal_RAG"

echo 启动步骤：
echo   1. 启动后端服务 (rag环境)
echo   2. 启动前端页面
echo.

REM 启动后端服务（在新窗口中）
echo [1/2] 正在启动后端服务...
start "多模态RAG后端服务" cmd /k "cd /d D:\Multimodal_RAG\src && call conda activate rag && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo [OK] 后端服务窗口已打开
echo.

timeout /t 3 /nobreak >nul

REM 启动前端服务（在新窗口中）
echo [2/2] 正在启动前端服务...
start "多模态RAG前端服务" cmd /k "cd /d D:\Multimodal_RAG\web && call conda activate rag && python -m http.server 8080"

echo [OK] 前端服务窗口已打开
echo.
echo ============================================
echo    所有服务已启动！
echo ============================================
echo.
echo 访问地址：
echo   - 前端页面: http://localhost:8080
echo   - API文档:  http://localhost:8000/docs
echo.
echo 按任意键关闭此窗口（服务继续运行）
pause >nul
