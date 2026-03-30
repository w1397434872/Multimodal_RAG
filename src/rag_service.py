"""
RAG服务模块
封装三个多模态RAG方案，提供统一的接口
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement, Image as ImageElement

from common import (
    load_config,
    init_directories,
    get_pdf_partition_config,
    init_llm,
    init_vl_model,
    init_embeddings,
    image_summarize,
    encode_image,
    is_image_path,
    is_base64,
    resize_image4tongyi,
    split_image_text_types,
    MultiDashScopeEmbeddings,
)
from schemas import RAGMethod, DocumentSource


class RAGService:
    """RAG服务类"""

    def __init__(self):
        self.config = load_config()
        self._vectorstores = {}  # 缓存不同方法的vectorstore
        self._retrievers = {}    # 缓存不同方法的retriever
        self._chains = {}        # 缓存不同方法的chain
        self._vectorstore_dir = os.path.join(self.config["BASE_DIR"], "vectorstore")
        os.makedirs(self._vectorstore_dir, exist_ok=True)

        # 尝试加载已存在的索引
        self._load_existing_indices()

    def _get_persist_path(self, method: RAGMethod) -> str:
        """获取持久化路径"""
        return os.path.join(self._vectorstore_dir, method.value)

    def _load_existing_indices(self):
        """加载已存在的索引"""
        for method in RAGMethod:
            persist_path = self._get_persist_path(method)
            if os.path.exists(persist_path):
                try:
                    self._load_index(method, persist_path)
                    print(f"✅ 已加载 {method.value} 的索引")
                except Exception as e:
                    print(f"⚠️ 加载 {method.value} 索引失败: {e}")

    def _load_index(self, method: RAGMethod, persist_path: str):
        """从磁盘加载指定方法的索引"""
        if method == RAGMethod.MULTIMODAL_VECTOR:
            embeddings_model = MultiDashScopeEmbeddings(api_key=self.config["DASHSCOPE_API_KEY"])
            vectorstore = Chroma(
                collection_name="multi-vector",
                embedding_function=embeddings_model,
                persist_directory=persist_path
            )
            self._vectorstores[method] = vectorstore
            self._retrievers[method] = vectorstore.as_retriever(search_kwargs={"k": 10})

        elif method == RAGMethod.MULTIMODAL_LLM:
            embeddings_model = init_embeddings(config=self.config)
            vectorstore = Chroma(
                collection_name="multi_model",
                embedding_function=embeddings_model,
                persist_directory=persist_path
            )
            self._vectorstores[method] = vectorstore
            # 注意：MultiVectorRetriever 的 docstore 无法持久化，需要重新构建

        elif method == RAGMethod.BALANCED:
            embeddings_model = init_embeddings(config=self.config)
            vectorstore = Chroma(
                collection_name="balanced",
                embedding_function=embeddings_model,
                persist_directory=persist_path
            )
            self._vectorstores[method] = vectorstore
            self._retrievers[method] = vectorstore.as_retriever(search_kwargs={"k": 10})

    def get_vectorstore(self, method: RAGMethod):
        """获取指定方法的向量存储"""
        return self._vectorstores.get(method)

    def get_retriever(self, method: RAGMethod):
        """获取指定方法的检索器"""
        return self._retrievers.get(method)

    def build_index_multimodal_vector(self, pdf_path: Optional[str] = None) -> Dict[str, int]:
        """
        方案一：基于多模态向量模型构建索引

        Args:
            pdf_path: PDF文件路径

        Returns:
            统计信息
        """
        init_directories(self.config, clean_images=True)

        pdf_config = get_pdf_partition_config(
            pdf_path or self.config["PDF_PATH"],
            self.config["IMAGE_OUT_DIR"]
        )
        pdf_data = partition_pdf(**pdf_config)

        # 提取文本
        texts = []
        for element in pdf_data:
            if isinstance(element, CompositeElement):
                texts.append(str(element))

        # 获取图片路径
        image_uris = sorted([
            os.path.join(self.config["IMAGE_OUT_DIR"], img)
            for img in os.listdir(self.config["IMAGE_OUT_DIR"])
            if img.endswith(".jpg")
        ])

        # 构建向量存储（带持久化）
        persist_path = self._get_persist_path(RAGMethod.MULTIMODAL_VECTOR)
        embeddings_model = MultiDashScopeEmbeddings(api_key=self.config["DASHSCOPE_API_KEY"])
        vectorstore = Chroma(
            collection_name="multi-vector",
            embedding_function=embeddings_model,
            persist_directory=persist_path
        )

        if image_uris:
            vectorstore.add_images(uris=image_uris)
        if texts:
            vectorstore.add_documents([Document(page_content=t) for t in texts])

        self._vectorstores[RAGMethod.MULTIMODAL_VECTOR] = vectorstore
        self._retrievers[RAGMethod.MULTIMODAL_VECTOR] = vectorstore.as_retriever(search_kwargs={"k": 10})

        return {
            "texts": len(texts),
            "images": len(image_uris)
        }

    def build_index_multimodal_llm(self, pdf_path: Optional[str] = None) -> Dict[str, int]:
        """
        方案二：基于多模态大模型（摘要索引）构建索引

        Args:
            pdf_path: PDF文件路径

        Returns:
            统计信息
        """
        init_directories(self.config, clean_images=True)

        pdf_config = get_pdf_partition_config(
            pdf_path or self.config["PDF_PATH"],
            self.config["IMAGE_OUT_DIR"]
        )
        pdf_data = partition_pdf(**pdf_config)

        # 提取文本和表格
        texts, tables = [], []
        for element in pdf_data:
            if isinstance(element, Table):
                tables.append(str(element))
            elif isinstance(element, CompositeElement):
                texts.append(str(element))

        # 生成摘要
        prompt = PromptTemplate.from_template(
            "你是一位负责生成表格和文本摘要以供检索的助理。"
            "这些摘要将被嵌入并用于检索原始文本或表格元素。"
            "请提供表格或文本的简明摘要，该摘要已针对检索进行了优化。表格或文本：{document}"
        )
        model = init_llm(config=self.config)
        summarize_chain = {"document": lambda x: x} | prompt | model | StrOutputParser()

        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5}) if texts else []
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5}) if tables else []

        # 生成图片摘要
        img_list, image_summaries = [], []
        for img_file in sorted(os.listdir(self.config["IMAGE_OUT_DIR"])):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(self.config["IMAGE_OUT_DIR"], img_file)
                img_list.append(img_path)
                summary = image_summarize(img_path, model_name=self.config["VL_MODEL"])
                image_summaries.append(summary)

        # 构建多向量检索器（带持久化）
        persist_path = self._get_persist_path(RAGMethod.MULTIMODAL_LLM)
        embeddings_model = init_embeddings(config=self.config)
        vectorstore = Chroma(
            collection_name="multi_model",
            embedding_function=embeddings_model,
            persist_directory=persist_path
        )
        docstore = InMemoryStore()
        id_key = "doc_id"

        def add_documents(doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            vectorstore.add_documents(summary_docs)
            docstore.mset(list(zip(doc_ids, doc_contents)))

        if texts and text_summaries:
            add_documents(text_summaries, texts)
        if tables and table_summaries:
            add_documents(table_summaries, tables)
        if img_list and image_summaries:
            add_documents(image_summaries, img_list)

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key,
            search_kwargs={"k": 7}
        )

        self._vectorstores[RAGMethod.MULTIMODAL_LLM] = vectorstore
        self._retrievers[RAGMethod.MULTIMODAL_LLM] = retriever

        return {
            "texts": len(texts),
            "tables": len(tables),
            "images": len(img_list)
        }

    def build_index_balanced(self, pdf_path: Optional[str] = None) -> Dict[str, int]:
        """
        方案三：成本与效果平衡方案构建索引

        Args:
            pdf_path: PDF文件路径

        Returns:
            统计信息
        """
        init_directories(self.config, clean_images=True)

        pdf_config = get_pdf_partition_config(
            pdf_path or self.config["PDF_PATH"],
            self.config["IMAGE_OUT_DIR"]
        )
        pdf_data = partition_pdf(**pdf_config)

        # 生成文档摘要
        prompt = PromptTemplate.from_template(
            "你是一位负责生成表格和文本摘要以供检索的助理。"
            "这些摘要将被嵌入并用于检索原始文本或表格元素。"
            "请提供表格或文本的简明摘要，该摘要已针对检索进行了优化。表格或文本：{document}"
        )
        model = init_llm(config=self.config)
        summarize_chain = {"document": lambda x: x.text} | prompt | model | StrOutputParser()
        summaries = summarize_chain.batch(pdf_data, {"max_concurrency": 5})

        # 生成图片摘要并融入文档
        image_count = 0
        for element in pdf_data:
            if not hasattr(element, 'metadata') or not hasattr(element.metadata, 'orig_elements'):
                continue
            for orig_element in element.metadata.orig_elements:
                if isinstance(orig_element, ImageElement):
                    image_meta = orig_element.metadata.to_dict()
                    image_path = image_meta.get("image_path")
                    if image_path and os.path.exists(image_path):
                        summary = image_summarize(image_path, model_name=self.config["VL_MODEL"])
                        orig_element.text = summary
                        image_count += 1

        # 构建多向量检索器（带持久化）
        persist_path = self._get_persist_path(RAGMethod.BALANCED)
        embeddings_model = init_embeddings(config=self.config)
        vectorstore = Chroma(
            collection_name="balanced",
            embedding_function=embeddings_model,
            persist_directory=persist_path
        )
        docstore = InMemoryStore()
        id_key = "doc_id"

        def add_documents(doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            vectorstore.add_documents(summary_docs)
            docstore.mset(list(zip(doc_ids, doc_contents)))

        add_documents(summaries, pdf_data)

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key,
            search_kwargs={"k": 2}
        )

        self._vectorstores[RAGMethod.BALANCED] = vectorstore
        self._retrievers[RAGMethod.BALANCED] = retriever

        return {
            "documents": len(pdf_data),
            "images": image_count
        }

    def build_index(self, method: RAGMethod, pdf_path: Optional[str] = None) -> Dict[str, int]:
        """根据指定方法构建索引"""
        if method == RAGMethod.MULTIMODAL_VECTOR:
            return self.build_index_multimodal_vector(pdf_path)
        elif method == RAGMethod.MULTIMODAL_LLM:
            return self.build_index_multimodal_llm(pdf_path)
        elif method == RAGMethod.BALANCED:
            return self.build_index_balanced(pdf_path)
        else:
            raise ValueError(f"不支持的方法: {method}")

    def clear_index(self, method: RAGMethod) -> bool:
        """清空指定方法的索引"""
        try:
            persist_path = self._get_persist_path(method)

            # 先关闭 vectorstore 释放资源
            if method in self._vectorstores:
                try:
                    # 尝试关闭客户端连接
                    vs = self._vectorstores[method]
                    if hasattr(vs, '_client') and vs._client:
                        vs._client.close()
                except Exception as e:
                    print(f"⚠️ 关闭 vectorstore 时出错: {e}")
                finally:
                    del self._vectorstores[method]

            # 从内存中移除其他引用
            if method in self._retrievers:
                del self._retrievers[method]
            if method in self._chains:
                del self._chains[method]

            # 删除磁盘上的索引文件
            if os.path.exists(persist_path):
                import shutil
                import time
                import stat

                def on_rm_error(func, path, exc_info):
                    """处理删除错误，尝试修改权限后重试"""
                    os.chmod(path, stat.S_IWRITE)
                    func(path)

                # 等待一下确保文件句柄释放
                time.sleep(0.5)

                # 使用 onerror 回调处理权限问题
                shutil.rmtree(persist_path, onerror=on_rm_error)
                print(f"✅ 已删除 {method.value} 的索引文件")
                return True
            else:
                print(f"⚠️ {method.value} 的索引文件不存在")
                return False

        except Exception as e:
            print(f"❌ 清空 {method.value} 索引失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def query_multimodal_vector(self, query: str, top_k: int = 5) -> Tuple[str, List[Any]]:
        """方案一：查询"""
        retriever = self._retrievers.get(RAGMethod.MULTIMODAL_VECTOR)
        if not retriever:
            raise ValueError("请先构建索引")

        from langchain_core.messages import HumanMessage

        docs = retriever.invoke(query)

        # 分离图片和文本
        result = split_image_text_types(docs, resize_dir=self.config["RESIZE_IMAGE_DIR"])
        images, texts = result["images"], result["texts"]

        # 构建消息
        messages = []
        for image in images[:top_k]:
            messages.append(HumanMessage(content=[
                {"text": f"请将图片标记标注为: {image}"},
                {"image": f"file://{image}"}
            ]))

        formatted_texts = "\n\n".join(texts)
        messages.append(HumanMessage(content=[{
            "text": (
                "你是一位专业的文档分析与问答助手。你的任务是根据提供的参考资料回答用户问题。"
                "你将获得相关的图片与文本作为参考的上下文。"
                "请根据提供的图片和文本结合你的分析能力，提供一份全面、准确的问题解答。"
                "请将提供的图片标记自然融入答案阐述的对应位置进行合理排版，"
                "答案中提及的图片统一以<image>传递的图片真实标记</image>的形式呈现。\n\n"
                f"用户的问题是：{query}\n\n"
                f"参考的文本：\n{formatted_texts}"
            )
        }]))

        llm = init_vl_model(config=self.config)
        response = llm.invoke(messages)

        return response.content, docs

    def query_multimodal_llm(self, query: str, top_k: int = 5) -> Tuple[str, List[Any]]:
        """方案二：查询"""
        retriever = self._retrievers.get(RAGMethod.MULTIMODAL_LLM)
        if not retriever:
            raise ValueError("请先构建索引")

        from langchain_core.messages import HumanMessage

        docs = retriever.invoke(query)

        images, texts = [], []
        for doc in docs[:top_k]:
            if is_image_path(doc):
                resized = resize_image4tongyi(doc, output_dir=self.config["RESIZE_IMAGE_DIR"])
                if resized:
                    images.append(resized)
            else:
                texts.append(doc)

        messages = []
        for image in images:
            messages.append(HumanMessage(content=[
                {"text": f"请将图片标记标注为: {image}"},
                {"image": f"file://{image}"}
            ]))

        formatted_texts = "\n\n".join(texts)
        messages.append(HumanMessage(content=[{
            "text": (
                "你是一位专业的文档分析与问答助手。你的任务是根据提供的参考资料回答用户问题。\n"
                "你将获得相关的图片与文本作为参考的上下文。"
                "请根据提供的图片和文本结合你的分析能力，提供一份全面、准确的问题解答。\n"
                "请将提供的图片标记自然融入答案阐述的对应位置进行合理排版，"
                "答案中提及的图片统一以'<image>传递的图片真实标记</image>'的形式呈现。\n\n"
                f"用户的问题是：{query}\n\n"
                f"参考的文本：\n{formatted_texts}"
            )
        }]))

        llm = init_vl_model(config=self.config)
        response = llm.invoke(messages)

        return response.content, docs

    def query_balanced(self, query: str, top_k: int = 5) -> Tuple[str, List[Any]]:
        """方案三：查询"""
        retriever = self._retrievers.get(RAGMethod.BALANCED)
        if not retriever:
            raise ValueError("请先构建索引")

        docs = retriever.invoke(query)

        # 转换文档为带图片标记的文本
        texts = []
        for doc in docs[:top_k]:
            if not hasattr(doc, 'metadata') or not hasattr(doc.metadata, 'orig_elements'):
                texts.append(str(doc))
                continue

            text_parts = []
            for element in doc.metadata.orig_elements:
                if isinstance(element, ImageElement):
                    image_path = element.metadata.image_path
                    image_text = getattr(element, 'text', '')
                    text_parts.append(f'<image src="{image_path}">{image_text}</image>')
                else:
                    text_parts.append(getattr(element, 'text', str(element)))
            texts.append("".join(text_parts))

        formatted_texts = "\n\n".join(texts)

        prompt_text = (
            "你是一位专业的文档分析与问答助手。你的任务是根据提供的参考资料回答用户问题。"
            "你将获得相关文档作为参考的上下文。"
            "请根据提供的文档结合你的分析能力，提供一份全面、准确的问题解答。"
            "请返回Markdown格式数据，并且当涉及到数学公式时，请使用正确的LaTeX语法。"
            "请将提供的文档中的图片`<image src='...'>`在需要时自然融入答案阐述的对应位置进行合理排版。\n\n"
            f"用户的问题是：\n{query}\n\n"
            f"参考的文档：\n{formatted_texts}"
        )

        llm = init_llm(config=self.config)
        response = llm.invoke(prompt_text)

        return response.content, docs

    def query(self, method: RAGMethod, query: str, top_k: int = 5) -> Tuple[str, List[Any]]:
        """根据指定方法查询"""
        if method == RAGMethod.MULTIMODAL_VECTOR:
            return self.query_multimodal_vector(query, top_k)
        elif method == RAGMethod.MULTIMODAL_LLM:
            return self.query_multimodal_llm(query, top_k)
        elif method == RAGMethod.BALANCED:
            return self.query_balanced(query, top_k)
        else:
            raise ValueError(f"不支持的方法: {method}")

    def convert_docs_to_sources(self, docs: List[Any]) -> List[DocumentSource]:
        """将文档转换为来源信息"""
        sources = []
        for doc in docs:
            if isinstance(doc, str):
                if is_image_path(doc):
                    sources.append(DocumentSource(type="image", content=doc))
                else:
                    sources.append(DocumentSource(type="text", content=doc))
            elif hasattr(doc, 'page_content'):
                content = doc.page_content
                if is_base64(content):
                    sources.append(DocumentSource(type="image", content="[base64_image]"))
                else:
                    sources.append(DocumentSource(type="text", content=content[:500]))
            else:
                sources.append(DocumentSource(type="unknown", content=str(doc)[:500]))
        return sources


# 全局服务实例
rag_service = RAGService()
