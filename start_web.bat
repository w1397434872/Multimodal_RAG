@echo off
chcp 65001 >nul
title 多模态RAG前端启动器
echo ============================================
echo    多模态RAG系统 - 前端启动脚本
echo ============================================
echo.

cd /d "D:\Multimodal_RAG\web"

echo 正在启动本地服务器...
echo.
echo ============================================
echo    前端页面地址: http://localhost:8080
echo ============================================
echo.
echo 提示: 请确保后端服务已启动 (运行 start_server.bat)
echo.

call conda activate rag && python -m http.server 8080

pause
