"""
多模态RAG FastAPI服务
提供三个RAG方案的统一接口

启动命令:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API文档:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
import uvicorn
import os
import shutil

from schemas import (
    QueryRequest, QueryResponse, IndexRequest, IndexResponse,
    HealthResponse, RAGMethod, DocumentSource, IndexStatus, IndexStatusResponse
)
from rag_service import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("🚀 多模态RAG服务启动中...")
    yield
    # 关闭时执行
    print("🛑 多模态RAG服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="多模态RAG API",
    description="基于FastAPI的多模态检索增强生成服务，支持三种RAG方案",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 返回服务状态"""
    return HealthResponse(
        status="running",
        version="1.0.0",
        available_methods=[
            RAGMethod.MULTIMODAL_VECTOR,
            RAGMethod.MULTIMODAL_LLM,
            RAGMethod.BALANCED
        ]
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        available_methods=[
            RAGMethod.MULTIMODAL_VECTOR,
            RAGMethod.MULTIMODAL_LLM,
            RAGMethod.BALANCED
        ]
    )


@app.get("/index/status", response_model=IndexStatusResponse)
async def get_index_status():
    """
    获取所有索引的状态

    返回每个方法的索引是否存在以及统计信息
    """
    indices = []

    for method in RAGMethod:
        retriever = rag_service.get_retriever(method)
        vectorstore = rag_service.get_vectorstore(method)

        exists = retriever is not None and vectorstore is not None

        # 尝试获取统计信息
        stats = {}
        if exists and vectorstore:
            try:
                # 获取集合中的文档数量
                collection = vectorstore._collection
                stats = {"documents": collection.count()}
            except Exception:
                stats = {"documents": 0}

        indices.append(IndexStatus(
            method=method.value,
            exists=exists,
            stats=stats
        ))

    return IndexStatusResponse(indices=indices)


@app.post("/index", response_model=IndexResponse)
async def build_index(request: IndexRequest):
    """
    构建索引

    - **method**: 选择RAG方法
        - `multimodal_vector`: 方案一，基于多模态向量模型
        - `multimodal_llm`: 方案二，基于多模态大模型（摘要索引）
        - `balanced`: 方案三，成本与效果平衡（默认）
    - **pdf_path**: 可选，指定PDF文件路径
    - **clear_existing**: 是否清空现有索引
    """
    try:
        stats = rag_service.build_index(
            method=request.method,
            pdf_path=request.pdf_path
        )

        return IndexResponse(
            success=True,
            message=f"索引构建成功，使用方法: {request.method.value}",
            method=request.method.value,
            stats=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引构建失败: {str(e)}")


@app.post("/index/upload", response_model=IndexResponse)
async def upload_and_build_index(
    file: UploadFile = File(..., description="要上传的PDF文件"),
    method: RAGMethod = Form(default=RAGMethod.BALANCED, description="使用的RAG方法"),
    clear_existing: bool = Form(default=True, description="是否清空现有索引")
):
    """
    上传PDF文件并构建索引

    - **file**: PDF文件
    - **method**: 选择RAG方法
        - `multimodal_vector`: 方案一，基于多模态向量模型
        - `multimodal_llm`: 方案二，基于多模态大模型（摘要索引）
        - `balanced`: 方案三，成本与效果平衡（默认）
    - **clear_existing**: 是否清空现有索引
    """
    # 验证文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    # 保存上传的文件
    upload_dir = os.path.join(os.path.dirname(__file__), "resources")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")
    finally:
        file.file.close()

    # 构建索引
    try:
        stats = rag_service.build_index(
            method=method,
            pdf_path=file_path
        )

        return IndexResponse(
            success=True,
            message=f"文件上传并索引成功，使用方法: {method.value}",
            method=method.value,
            stats=stats
        )
    except Exception as e:
        import traceback
        error_detail = f"索引构建失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # 打印到服务器日志
        # 清理上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/index/all", response_model=List[IndexResponse])
async def build_all_indices(pdf_path: str = None):
    """
    为所有方法构建索引

    - **pdf_path**: 可选，指定PDF文件路径
    """
    responses = []

    for method in RAGMethod:
        try:
            stats = rag_service.build_index(method=method, pdf_path=pdf_path)
            responses.append(IndexResponse(
                success=True,
                message=f"索引构建成功",
                method=method.value,
                stats=stats
            ))
        except Exception as e:
            responses.append(IndexResponse(
                success=False,
                message=f"索引构建失败: {str(e)}",
                method=method.value,
                stats={}
            ))

    return responses


@app.delete("/index/{method}", response_model=IndexResponse)
async def clear_index(method: RAGMethod):
    """
    清空指定方法的索引

    - **method**: 要清空的RAG方法
    """
    try:
        success = rag_service.clear_index(method)
        if success:
            return IndexResponse(
                success=True,
                message=f"{method.value} 索引已清空",
                method=method.value,
                stats={}
            )
        else:
            return IndexResponse(
                success=False,
                message=f"{method.value} 索引不存在或清空失败",
                method=method.value,
                stats={}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空索引失败: {str(e)}")


@app.delete("/index", response_model=IndexResponse)
async def clear_all_indices():
    """
    清空所有索引
    """
    try:
        cleared_count = 0
        for method in RAGMethod:
            if rag_service.clear_index(method):
                cleared_count += 1

        return IndexResponse(
            success=True,
            message=f"已清空 {cleared_count}/3 个索引",
            method="all",
            stats={"cleared": cleared_count}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空所有索引失败: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    查询接口

    - **query**: 用户查询问题
    - **method**: 使用的RAG方法（默认balanced）
    - **top_k**: 返回的最相关文档数量（默认5，最大20）

    返回:
    - **query**: 原始查询
    - **answer**: 生成的答案
    - **sources**: 参考的文档来源
    - **method**: 使用的RAG方法
    """
    try:
        # 检查索引是否存在
        retriever = rag_service.get_retriever(request.method)
        if not retriever:
            raise HTTPException(
                status_code=400,
                detail=f"方法 {request.method.value} 的索引尚未构建，请先调用 /index 接口"
            )

        # 执行查询
        answer, docs = rag_service.query(
            method=request.method,
            query=request.query,
            top_k=request.top_k
        )

        # 转换文档为来源信息
        sources = rag_service.convert_docs_to_sources(docs)

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            method=request.method.value
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/query/all", response_model=List[QueryResponse])
async def query_all_methods(query: str, top_k: int = 5):
    """
    使用所有方法进行查询对比

    - **query**: 用户查询问题
    - **top_k**: 返回的最相关文档数量
    """
    responses = []

    for method in RAGMethod:
        try:
            retriever = rag_service.get_retriever(method)
            if not retriever:
                responses.append(QueryResponse(
                    query=query,
                    answer=f"方法 {method.value} 的索引尚未构建",
                    sources=[],
                    method=method.value
                ))
                continue

            answer, docs = rag_service.query(method=method, query=query, top_k=top_k)
            sources = rag_service.convert_docs_to_sources(docs)

            responses.append(QueryResponse(
                query=query,
                answer=answer,
                sources=sources,
                method=method.value
            ))
        except Exception as e:
            responses.append(QueryResponse(
                query=query,
                answer=f"查询失败: {str(e)}",
                sources=[],
                method=method.value
            ))

    return responses


@app.get("/methods")
async def get_methods():
    """获取所有可用的RAG方法"""
    return {
        "methods": [
            {
                "id": RAGMethod.MULTIMODAL_VECTOR,
                "name": "多模态向量模型",
                "description": "基于多模态向量模型同时处理文本和图片，适合图片能单独理解的场景"
            },
            {
                "id": RAGMethod.MULTIMODAL_LLM,
                "name": "多模态大模型（摘要索引）",
                "description": "使用多模态大模型生成图片摘要，检索效果更精准，成本较高"
            },
            {
                "id": RAGMethod.BALANCED,
                "name": "成本与效果平衡",
                "description": "图片摘要融入文档上下文，使用文本大模型回答，平衡成本和效果（推荐）"
            }
        ]
    }


@app.delete("/index/{method}")
async def clear_index(method: RAGMethod):
    """清空指定方法的索引"""
    try:
        # 这里可以实现清空索引的逻辑
        return {"success": True, "message": f"方法 {method.value} 的索引已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空索引失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
