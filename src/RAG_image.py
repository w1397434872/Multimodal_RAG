"""
方案一：基于多模态向量模型

核心思路：
1. 使用多模态向量模型同时处理文本和图片
2. 文本和图片分别嵌入到同一向量空间
3. 检索时根据向量相似度返回相关文本或图片
4. 使用多模态大模型生成最终答案

适用场景：
- PDF中图片具备丰富信息，能单独理解
- 图片内容独立于文本也能被理解
"""

from IPython.display import HTML, display
import markdown
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement

from common import (
    load_config,
    init_directories,
    get_pdf_partition_config,
    init_vl_model,
    MultiDashScopeEmbeddings,
    encode_image,
    is_base64,
    split_image_text_types,
)


def prompt_func(data_dict):
    """构建多模态提示词"""
    images = data_dict["context"]["images"]
    texts = data_dict["context"]["texts"]

    messages = []

    # 装载图片数据
    for image in images:
        messages.append(
            HumanMessage(
                content=[
                    {"text": f"请将图片标记标注为: {image}"},
                    {"image": f"file://{image}"}
                ]
            )
        )

    # 装载文本数据
    formatted_texts = "\n\n".join(texts)
    messages.append(
        HumanMessage(content=[{
            "text":
                "你是一位专业的文档分析与问答助手。你的任务是根据提供的参考资料回答用户问题。"
                "你将获得相关的图片与文本作为参考的上下文，这些资料是根据用户输入的关键词从向量数据库中检索获取的。"
                "请根据提供的图片和文本结合你的分析能力，提供一份全面、准确的问题解答。"
                "请将提供的图片标记自然融入答案阐述的对应位置进行合理排版，答案中提及的图片统一以<image>传递的图片真实标记</image>的形式呈现。\n\n"
                f"用户的问题是：{data_dict['question']}\n\n"
                "参考的文本或者表格数据：\n"
                f"{formatted_texts}"
        }])
    )

    return messages


def show_plt_img(img_base64):
    """显示base64图片"""
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))


def display_answer(text: str):
    """显示带图片的答案"""
    start_tag = "<image>"
    end_tag = "</image>"

    parts = text.split(start_tag)
    for part in parts:
        chunks = part.split(end_tag)
        if len(chunks) > 1:
            # 存在图片
            image_path = chunks[0]
            context = chunks[1]
            img_base64 = encode_image(image_path)
            display(HTML(f'\n<img src="data:image/jpeg;base64,{img_base64}"/>\n'))
            display(HTML(markdown.markdown(context)))
        else:
            display(HTML(markdown.markdown(part)))


def build_vectorstore(config, texts, image_uris, persist_directory=None):
    """构建多模态向量存储，支持持久化"""
    # 使用多模态嵌入模型
    embeddings_model = MultiDashScopeEmbeddings(api_key=config["DASHSCOPE_API_KEY"])

    # 设置持久化目录
    if persist_directory is None:
        persist_directory = os.path.join(config["BASE_DIR"], "vectorstore", "multimodal_vector")
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma(
        collection_name="multi-vector",
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )

    # 添加图片到向量存储
    if image_uris:
        vectorstore.add_images(uris=image_uris)

    # 添加文本到向量存储
    if texts:
        vectorstore.add_documents([Document(page_content=text) for text in texts])

    return vectorstore


def load_vectorstore(config, persist_directory=None):
    """从磁盘加载已存在的向量存储"""
    if persist_directory is None:
        persist_directory = os.path.join(config["BASE_DIR"], "vectorstore", "multimodal_vector")

    if not os.path.exists(persist_directory):
        return None

    embeddings_model = MultiDashScopeEmbeddings(api_key=config["DASHSCOPE_API_KEY"])

    try:
        vectorstore = Chroma(
            collection_name="multi-vector",
            embedding_function=embeddings_model,
            persist_directory=persist_directory
        )
        return vectorstore
    except Exception as e:
        print(f"加载向量存储失败: {e}")
        return None


def parse_pdf(config):
    """解析PDF文档，提取文本和图片"""
    pdf_config = get_pdf_partition_config(config["PDF_PATH"], config["IMAGE_OUT_DIR"])
    pdf_data = partition_pdf(**pdf_config)

    # 提取表格和文本
    tables = []
    texts = []

    for element in pdf_data:
        if isinstance(element, Table):
            tables.append(str(element))
        elif isinstance(element, CompositeElement):
            texts.append(str(element))

    print(f"表格元素：{len(tables)} \n文本元素：{len(texts)}")

    # 获取图片路径列表
    image_uris = sorted([
        os.path.join(config["IMAGE_OUT_DIR"], image_name)
        for image_name in os.listdir(config["IMAGE_OUT_DIR"])
        if image_name.endswith(".jpg")
    ])

    return texts, tables, image_uris


def build_rag_chain(config, retriever):
    """构建RAG处理链"""
    # 初始化多模态大模型
    llm = init_vl_model(config=config)

    # 构建RAG链
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(
                lambda docs: split_image_text_types(docs, resize_dir=config["RESIZE_IMAGE_DIR"])
            )
        }
        | RunnableLambda(prompt_func)
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    """主程序入口"""
    # ============================================
    # 一、初始化配置和目录
    # ============================================
    config = load_config()
    init_directories(config, clean_images=True)

    # ============================================
    # 二、解析PDF文档
    # ============================================
    texts, tables, image_uris = parse_pdf(config)

    # ============================================
    # 三、创建多模态向量存储
    # ============================================
    vectorstore = build_vectorstore(config, texts, image_uris)

    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # ============================================
    # 四、构建RAG Pipeline
    # ============================================
    chain = build_rag_chain(config, retriever)

    # ============================================
    # 五、效果展示
    # ============================================
    query = "电路模型"
    docs = retriever.invoke(query)

    # 显示检索到的图片
    for doc in docs:
        if is_base64(doc.page_content):
            show_plt_img(doc.page_content)

    # 生成并显示答案
    result = chain.invoke(query)
    print(result)
    display_answer(result)


if __name__ == "__main__":
    main()
