"""
多模态RAG系统公共模块
包含配置加载、公共函数和工具类
"""

import os
import shutil
import base64
import io
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from dotenv import load_dotenv

# ============================================
# 配置加载
# ============================================

def load_config():
    """加载环境变量配置"""
    load_dotenv()

    # 基础路径 - 获取当前文件所在目录（项目根目录）
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir_from_env = os.getenv("BASE_DIR", ".")

    # 如果 BASE_DIR 是相对路径，则相对于当前文件目录解析
    if base_dir_from_env == ".":
        base_dir = current_file_dir
    elif not os.path.isabs(base_dir_from_env):
        base_dir = os.path.abspath(os.path.join(current_file_dir, base_dir_from_env))
    else:
        base_dir = base_dir_from_env

    resource_dir = os.path.join(base_dir, "resources")
    tools_dir = os.path.join(base_dir, "tools")

    # 子目录路径
    image_out_dir = os.path.join(resource_dir, "images")
    resize_image_dir = os.path.join(resource_dir, "temp")
    pdf_name = os.getenv("PDF_NAME", "5566778899.pdf")
    pdf_path = os.path.join(resource_dir, pdf_name)

    # 模型下载路径
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # 设置模型缓存环境变量
    os.environ['TRANSFORMERS_CACHE'] = model_dir
    os.environ['HF_HOME'] = model_dir
    os.environ['HF_HUB_CACHE'] = model_dir

    # 设置外部工具路径
    poppler_path_from_env = os.getenv("POPPLER_PATH", "tools/poppler/bin")
    tesseract_path_from_env = os.getenv("TESSERACT_PATH", "tools/tesseract")

    # 解析工具路径（支持相对路径）
    if not os.path.isabs(poppler_path_from_env):
        poppler_path = os.path.abspath(os.path.join(base_dir, poppler_path_from_env))
    else:
        poppler_path = poppler_path_from_env

    if not os.path.isabs(tesseract_path_from_env):
        tesseract_path = os.path.abspath(os.path.join(base_dir, tesseract_path_from_env))
    else:
        tesseract_path = tesseract_path_from_env

    # 将工具路径添加到系统PATH
    if os.path.exists(poppler_path):
        os.environ['path'] += os.pathsep + poppler_path
    if os.path.exists(tesseract_path):
        os.environ['path'] += os.pathsep + tesseract_path

    # API密钥
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key

    # HuggingFace配置
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        os.environ['HF_ENDPOINT'] = hf_endpoint
    hf_hub_offline = os.getenv("HF_HUB_OFFLINE")
    if hf_hub_offline:
        os.environ['HF_HUB_OFFLINE'] = hf_hub_offline

    # 模型名称配置
    config = {
        "BASE_DIR": base_dir,
        "RESOURCE_DIR": resource_dir,
        "TOOLS_DIR": tools_dir,
        "IMAGE_OUT_DIR": image_out_dir,
        "RESIZE_IMAGE_DIR": resize_image_dir,
        "PDF_PATH": pdf_path,
        "MODEL_DIR": model_dir,
        "POPPLER_PATH": poppler_path,
        "TESSERACT_PATH": tesseract_path,
        "DASHSCOPE_API_KEY": dashscope_api_key,
        "LLM_MODEL": os.getenv("LLM_MODEL", "qwen-max"),
        "VL_MODEL": os.getenv("VL_MODEL", "qwen-vl-max"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-v1"),
        "MULTIMODAL_EMBEDDING_MODEL": os.getenv("MULTIMODAL_EMBEDDING_MODEL", "multimodal-embedding-v1"),
    }

    return config


# ============================================
# 目录管理
# ============================================

def ensure_dir(directory: str, clean: bool = False) -> str:
    """
    确保目录存在，可选择是否清空

    Args:
        directory: 目录路径
        clean: 是否清空已有目录

    Returns:
        目录路径
    """
    if clean and os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    return directory


def init_directories(config: Dict[str, str], clean_images: bool = True) -> None:
    """
    初始化所有必要的目录

    Args:
        config: 配置字典
        clean_images: 是否清空图片输出目录
    """
    ensure_dir(config["MODEL_DIR"])
    ensure_dir(config["IMAGE_OUT_DIR"], clean=clean_images)
    ensure_dir(config["RESIZE_IMAGE_DIR"], clean=True)


# ============================================
# PDF解析配置
# ============================================

def get_pdf_partition_config(pdf_path: str, image_out_dir: str) -> Dict[str, Any]:
    """
    获取PDF解析配置参数

    Args:
        pdf_path: PDF文件路径
        image_out_dir: 图片输出目录

    Returns:
        partition_pdf的配置参数字典
    """
    return {
        "filename": pdf_path,
        "strategy": "hi_res",
        "extract_image_block_types": ["Image"],
        "extract_images_in_pdf": True,
        "infer_table_structure": True,
        "max_characters": 4000,
        "new_after_n_chars": 3800,
        "combine_text_under_n_chars": 2000,
        "chunking_strategy": "by_title",
        "extract_image_block_output_dir": image_out_dir,
    }


# ============================================
# 图片处理函数
# ============================================

def is_base64(s: str) -> bool:
    """检查字符串是否为base64编码"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def is_image_path(filepath: str) -> bool:
    """检查路径是否为有效的图片文件"""
    try:
        path = Path(filepath)
        return all([
            path.exists(),
            path.is_file(),
            path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        ])
    except Exception:
        return False


def encode_image(img_path: str) -> str:
    """
    将图片文件编码为base64字符串

    Args:
        img_path: 图片文件路径

    Returns:
        base64编码的字符串
    """
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()
        return base64.b64encode(img_data).decode('utf-8')


def resize_base64_image4tongyi(
    base64_string: str,
    max_size: tuple = (640, 480),
    output_dir: Optional[str] = None
) -> str:
    """
    将base64图片缩放并保存到本地

    Args:
        base64_string: base64编码的图片
        max_size: 最大尺寸 (宽, 高)
        output_dir: 输出目录，默认使用配置中的RESIZE_IMAGE_DIR

    Returns:
        缩放后的图片本地路径
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    width, height = img.size
    ratio = min(max_size[0] / width, max_size[1] / height)
    new_width, new_height = int(width * ratio), int(height * ratio)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    file_name = f"{uuid.uuid4()}_resized.jpg"
    out_path = os.path.join(output_dir or "temp", file_name)
    resized_img.save(out_path)

    return out_path


def resize_image4tongyi(
    image_path: str,
    max_size: tuple = (640, 480),
    output_dir: Optional[str] = None
) -> Optional[str]:
    """
    将本地图片缩放并保存

    Args:
        image_path: 原图片路径
        max_size: 最大尺寸 (宽, 高)
        output_dir: 输出目录

    Returns:
        缩放后的图片路径，失败返回None
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        ratio = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        out_path = os.path.join(output_dir or "temp", f"{uuid.uuid4()}.jpg")
        resized_img.save(out_path)
        return out_path
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None


# ============================================
# 文本/图片分离
# ============================================

def split_image_text_types(
    docs: List[Any],
    resize_dir: Optional[str] = None
) -> Dict[str, List]:
    """
    将文档列表分离为图片和文本

    Args:
        docs: 文档列表
        resize_dir: 图片缩放输出目录

    Returns:
        {"images": [...], "texts": [...]}
    """
    images = []
    texts = []

    for doc in docs:
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)

        if is_base64(content):
            resize_image = resize_base64_image4tongyi(content, output_dir=resize_dir)
            images.append(resize_image)
        elif is_image_path(content):
            resize_image = resize_image4tongyi(content, output_dir=resize_dir)
            if resize_image:
                images.append(resize_image)
        else:
            texts.append(content)

    return {"images": images, "texts": texts}


# ============================================
# 显示函数
# ============================================

def pretty_print_docs(docs: List[Any]) -> None:
    """美观地打印文档列表"""
    print(
        f"\n{'_' * 100}\n".join(
            [f"Document {i+1}:\n\n" + (d.page_content if hasattr(d, 'page_content') else str(d))
             for i, d in enumerate(docs)]
        )
    )


# ============================================
# 模型初始化
# ============================================

def init_llm(model_name: Optional[str] = None, config: Optional[Dict] = None):
    """
    初始化LLM模型

    Args:
        model_name: 模型名称，默认从配置读取
        config: 配置字典

    Returns:
        ChatTongyi实例
    """
    from langchain_community.chat_models.tongyi import ChatTongyi

    if config is None:
        config = {}

    model = model_name or config.get("LLM_MODEL", "qwen-max")
    return ChatTongyi(model=model)


def init_vl_model(model_name: Optional[str] = None, config: Optional[Dict] = None):
    """
    初始化多模态视觉模型

    Args:
        model_name: 模型名称，默认从配置读取
        config: 配置字典

    Returns:
        ChatTongyi实例
    """
    from langchain_community.chat_models.tongyi import ChatTongyi

    if config is None:
        config = {}

    model = model_name or config.get("VL_MODEL", "qwen-vl-max")
    return ChatTongyi(model=model)


def init_embeddings(model_name: Optional[str] = None, config: Optional[Dict] = None):
    """
    初始化嵌入模型

    Args:
        model_name: 模型名称，默认从配置读取
        config: 配置字典

    Returns:
        DashScopeEmbeddings实例
    """
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings

    if config is None:
        config = {}

    model = model_name or config.get("EMBEDDING_MODEL", "text-embedding-v1")
    return DashScopeEmbeddings(model=model)


# ============================================
# 图片摘要生成
# ============================================

def image_summarize(image_path: str, model_name: Optional[str] = None) -> str:
    """
    使用多模态模型生成图片摘要

    Args:
        image_path: 图片路径
        model_name: 模型名称

    Returns:
        图片摘要文本
    """
    from langchain_core.messages import HumanMessage
    from langchain_community.chat_models.tongyi import ChatTongyi

    chat = ChatTongyi(model=model_name or "qwen-vl-max")
    local_image_path = f"file://{image_path}"

    response = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"text": "你是一名负责生成图像摘要以便检索的助理。这些摘要将被嵌入并用于检索原始图像。请生成针对检索进行了优化的简洁的图像摘要。"},
                    {"image": local_image_path}
                ]
            )
        ]
    )

    # 处理响应格式
    if isinstance(response.content, list) and len(response.content) > 0:
        if isinstance(response.content[0], dict):
            return response.content[0].get("text", "")
    return str(response.content)


# ============================================
# 多模态嵌入模型 (用于方案一)
# ============================================

from typing import List
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
import dashscope


class MultiDashScopeEmbeddings(BaseModel, Embeddings):
    """多模态嵌入模型 - 支持文本和图片"""
    model: str = "multimodal-embedding-v1"
    api_key: Optional[str] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        # 如果没有提供api_key，从环境变量获取
        if self.api_key is None:
            self.api_key = os.getenv("DASHSCOPE_API_KEY", "")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档文本"""
        text_features = []
        for text in texts:
            resp = dashscope.MultiModalEmbedding.call(
                model=self.model,
                input=[{"text": text}],
                api_key=self.api_key
            )
            embeddings_list = resp.output['embeddings'][0]['embedding']
            text_features.append(embeddings_list)
        return text_features

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=[{"text": text}],
            api_key=self.api_key
        )
        embeddings_list = resp.output['embeddings'][0]['embedding']
        return embeddings_list

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        """嵌入图片"""
        image_features = []
        for uri in uris:
            local_image_uri = f"file://{uri}"
            resp = dashscope.MultiModalEmbedding.call(
                model=self.model,
                input=[{"image": local_image_uri}],
                api_key=self.api_key
            )
            embeddings_list = resp.output['embeddings'][0]['embedding']
            print(f"{uri} 图片向量 {resp.status_code}:{embeddings_list[:3]}")
            image_features.append(embeddings_list)
        return image_features
