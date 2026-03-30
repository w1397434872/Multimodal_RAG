"""
方案二：基于多模态大模型（摘要索引）

核心思路：
1. 使用多模态大模型生成图片摘要
2. 使用文本大模型生成文本摘要
3. 将摘要存入向量数据库用于检索
4. 原始内容存入文档存储
5. 使用多向量检索器关联摘要和原始内容
6. 使用多模态大模型生成最终答案

适用场景：
- 图片需要结合上下文才能理解
- 需要更精准的检索效果
- 可以接受较高的API调用成本
"""

import os
import uuid
from IPython.display import HTML, display
import markdown

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement

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
    resize_image4tongyi
)


def parse_pdf(config):
    """解析PDF文档，提取文本和表格"""
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
    return texts, tables


def generate_summaries(texts, tables, config):
    """生成文本和表格摘要"""
    # 摘要生成提示词
    prompt = PromptTemplate.from_template(
        "你是一位负责生成表格和文本摘要以供检索的助理。"
        "这些摘要将被嵌入并用于检索原始文本或表格元素。"
        "请提供表格或文本的简明摘要，该摘要已针对检索进行了优化。表格或文本：{document}"
    )

    # 初始化模型
    model = init_llm(config=config)
    summarize_chain = {"document": lambda x: x} | prompt | model | StrOutputParser()

    # 批量生成摘要
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5}) if texts else []
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5}) if tables else []

    return text_summaries, table_summaries


def generate_image_summaries(config):
    """生成图片摘要"""
    img_list = []
    image_summaries = []

    for img_file in sorted(os.listdir(config["IMAGE_OUT_DIR"])):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(config["IMAGE_OUT_DIR"], img_file)
            img_list.append(img_path)
            # 生成图片摘要
            summary = image_summarize(img_path, model_name=config["VL_MODEL"])
            image_summaries.append(summary)
            print(f"已生成摘要: {img_path}")

    return img_list, image_summaries


def build_multivector_retriever(config, text_summaries, texts, table_summaries, tables, image_summaries, img_list):
    """构建多向量检索器"""
    # 初始化嵌入模型
    embeddings_model = init_embeddings(config=config)

    # 创建向量数据库（存储摘要）
    vectorstore = Chroma(
        collection_name="multi_model",
        embedding_function=embeddings_model
    )

    # 创建内存存储（存储原始内容）
    docstore = InMemoryStore()

    # 文档ID键名
    id_key = "doc_id"

    def add_documents(doc_summaries, doc_contents):
        """将摘要和原始内容添加到存储"""
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        vectorstore.add_documents(summary_docs)
        docstore.mset(list(zip(doc_ids, doc_contents)))

    # 添加各类文档（只在非空时添加）
    if texts and text_summaries:
        add_documents(text_summaries, texts)
    if tables and table_summaries:
        add_documents(table_summaries, tables)
    if img_list and image_summaries:
        add_documents(image_summaries, img_list)

    # 构建多向量检索器
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": 7}
    )

    return retriever


def split_image_text_types(docs, resize_dir):
    """分离图片和文本"""
    images = []
    texts = []
    for doc in docs:
        if is_image_path(doc):
            resized = resize_image4tongyi(doc, output_dir=resize_dir)
            if resized:
                images.append(resized)
        else:
            texts.append(doc)
    return {"images": images, "texts": texts}


def img_prompt_func(data_dict):
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
            "text": (
                "你是一位专业的文档分析与问答助手。你的任务是根据提供的参考资料回答用户问题。\n"
                "你将获得相关的图片与文本作为参考的上下文。这些资料是根据用户输入的关键词从向量数据库中检索获取的。\n"
                "请根据提供的图片和文本结合你的分析能力，提供一份全面、准确的问题解答。\n"
                "请将提供的图片标记自然融入答案阐述的对应位置进行合理排版，答案中提及的图片统一以'<image>传递的图片真实标记</image>'的形式呈现。\n\n"
                f"用户的问题是：{data_dict['question']}\n\n"
                "参考的文本或者表格数据：\n"
                f"{formatted_texts}"
            )
        }])
    )

    return messages


def build_rag_chain(config, retriever):
    """构建RAG处理链"""
    # 初始化多模态大模型
    llm = init_vl_model(config=config)

    # 构建RAG链
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(
                lambda docs: split_image_text_types(docs, config["RESIZE_IMAGE_DIR"])
            )
        }
        | RunnableLambda(img_prompt_func)
        | llm
        | StrOutputParser()
    )

    return chain


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
            image_path = chunks[0]
            context = chunks[1]
            img_base64 = encode_image(image_path)
            display(HTML(f'\n<img src="data:image/jpeg;base64,{img_base64}"/>\n'))
            display(HTML(markdown.markdown(context)))
        else:
            display(HTML(markdown.markdown(part)))


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
    texts, tables = parse_pdf(config)

    # ============================================
    # 三、生成摘要
    # ============================================
    text_summaries, table_summaries = generate_summaries(texts, tables, config)

    # ============================================
    # 四、生成图片摘要
    # ============================================
    img_list, image_summaries = generate_image_summaries(config)

    # ============================================
    # 五、构建多向量索引
    # ============================================
    retriever = build_multivector_retriever(
        config, text_summaries, texts, table_summaries, tables, image_summaries, img_list
    )

    # ============================================
    # 六、构建RAG Pipeline
    # ============================================
    chain = build_rag_chain(config, retriever)

    # ============================================
    # 七、效果展示
    # ============================================
    query = "导体中的电流方向是什么"
    docs = retriever.invoke(query)

    # 显示检索到的图片
    for doc in docs:
        if is_image_path(doc):
            show_plt_img(encode_image(doc))

    # 生成并显示答案
    result = chain.invoke(query)
    print(result)
    display_answer(result)


if __name__ == "__main__":
    main()
