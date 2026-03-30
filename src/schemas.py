"""
Pydantic模型定义
用于FastAPI请求和响应的数据验证
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class RAGMethod(str, Enum):
    """RAG方法枚举"""
    MULTIMODAL_VECTOR = "multimodal_vector"      # 方案一：基于多模态向量模型
    MULTIMODAL_LLM = "multimodal_llm"            # 方案二：基于多模态大模型（摘要索引）
    BALANCED = "balanced"                        # 方案三：成本与效果平衡


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询问题", example="什么是RAG")
    method: RAGMethod = Field(default=RAGMethod.BALANCED, description="使用的RAG方法")
    top_k: int = Field(default=5, ge=1, le=20, description="返回的最相关文档数量")


class DocumentSource(BaseModel):
    """文档来源信息"""
    type: str = Field(..., description="文档类型：text/table/image")
    content: Optional[str] = Field(None, description="文本内容或图片路径")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")


class QueryResponse(BaseModel):
    """查询响应模型"""
    query: str = Field(..., description="原始查询")
    answer: str = Field(..., description="生成的答案")
    sources: List[DocumentSource] = Field(default=[], description="参考的文档来源")
    method: str = Field(..., description="使用的RAG方法")


class IndexRequest(BaseModel):
    """索引构建请求模型"""
    pdf_path: Optional[str] = Field(None, description="PDF文件路径，不填则使用配置中的默认路径")
    method: RAGMethod = Field(default=RAGMethod.BALANCED, description="构建索引使用的RAG方法")
    clear_existing: bool = Field(default=True, description="是否清空现有索引")


class IndexResponse(BaseModel):
    """索引构建响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态信息")
    method: str = Field(..., description="使用的RAG方法")
    stats: Dict[str, int] = Field(default={}, description="索引统计信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(default="1.0.0", description="API版本")
    available_methods: List[str] = Field(..., description="可用的RAG方法")


class IndexStatus(BaseModel):
    """单个索引状态"""
    method: str = Field(..., description="RAG方法")
    exists: bool = Field(..., description="索引是否存在")
    stats: Dict[str, int] = Field(default={}, description="索引统计信息")


class IndexStatusResponse(BaseModel):
    """索引状态响应"""
    indices: List[IndexStatus] = Field(..., description="所有索引的状态")
