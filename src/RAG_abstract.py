"""
方案三：基于多模态大模型（成本与效果平衡）

核心思路：
1. 解析PDF文档，利用metadata.orig_elements获取元素合并信息
2. 使用文本大模型生成文档摘要用于检索
3. 使用多模态大模型生成图片摘要，融入文档对应位置
4. 检索后使用文本大模型生成答案
5. 在答案中保留图片引用标记

优化目的：
- 解决图片单独无法理解的问题
- 平衡成本和效果
- 图片摘要融入文档上下文

适用场景：
- 图片只用于展示或需要文字辅助理解
- 需要较好的检索准确性
- 希望控制API调用成本
"""

import os
import uuid
import re
from IPython.display import HTML, display, Markdown

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image as ImageElement

from common import (
    load_config,
    init_directories,
    get_pdf_partition_config,
    init_llm,
    init_embeddings,
    image_summarize,
    encode_image
)


def parse_pdf(config):
    """解析PDF文档，提取元素结构"""
    pdf_config = get_pdf_partition_config(config["PDF_PATH"], config["IMAGE_OUT_DIR"])
    pdf_data = partition_pdf(**pdf_config)

    # 查看合并后的元素结构
    print("合并后的元素类型示例:")
    if pdf_data:
        print(f"  元素类型: {type(pdf_data[0])}")
        if hasattr(pdf_data[0], 'metadata') and hasattr(pdf_data[0].metadata, 'orig_elements'):
            print(f"  子元素数量: {len(pdf_data[0].metadata.orig_elements)}")
            print(f"  子元素类型: {[type(e).__name__ for e in pdf_data[0].metadata.orig_elements]}")

    return pdf_data


def generate_summaries(pdf_data, config):
    """生成文档摘要"""
    # 摘要生成提示词
    prompt = PromptTemplate.from_template(
        "你是一位负责生成表格和文本摘要以供检索的助理。"
        "这些摘要将被嵌入并用于检索原始文本或表格元素。"
        "请提供表格或文本的简明摘要，该摘要已针对检索进行了优化。表格或文本：{document}"
    )

    # 初始化模型
    model = init_llm(config=config)
    summarize_chain = {"document": lambda x: x.text} | prompt | model | StrOutputParser()

    # 批量生成文档摘要
    summaries = summarize_chain.batch(pdf_data, {"max_concurrency": 5})

    return summaries


def generate_image_summaries(pdf_data, config):
    """生成图片摘要并融入文档"""
    image_summaries = []

    for element in pdf_data:
        if not hasattr(element, 'metadata') or not hasattr(element.metadata, 'orig_elements'):
            continue

        orig_elements = element.metadata.orig_elements
        for orig_element in orig_elements:
            # 处理图片元素
            if isinstance(orig_element, ImageElement):
                image_meta = orig_element.metadata.to_dict()
                image_path = image_meta.get("image_path")

                if image_path and os.path.exists(image_path):
                    # 生成图片摘要
                    summary = image_summarize(image_path, model_name=config["VL_MODEL"])
                    # 将摘要存入图片元素的text属性
                    orig_element.text = summary
                    image_summaries.append(summary)
                    print(f"已生成图片摘要: {image_path}")

    return image_summaries


def build_multivector_retriever(config, summaries, pdf_data):
    """构建多向量检索器"""
    # 初始化嵌入模型
    embeddings_model = init_embeddings(config=config)

    # 创建向量数据库
    vectorstore = Chroma(
        collection_name="multi_model_opt",
        embedding_function=embeddings_model
    )

    # 创建内存存储
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

    # 添加文档（包含图片摘要的完整元素）
    add_documents(summaries, pdf_data)

    # 构建多向量检索器
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": 2}
    )

    return retriever


def split_image_text_types(docs):
    """
    将文档元素转换为带图片标记的文本

    将PDF Element中的图片用<image>标签包裹，融入文本上下文
    """
    texts = []

    for doc in docs:
        text_parts = []

        # 获取所有子元素
        if not hasattr(doc, 'metadata') or not hasattr(doc.metadata, 'orig_elements'):
            texts.append(str(doc))
            continue

        orig_elements = doc.metadata.orig_elements

        for element in orig_elements:
            if isinstance(element, ImageElement):
                # 图片元素用标记包裹
                image_path = element.metadata.image_path
                image_text = getattr(element, 'text', '')
                print(f"文档上下文中图片: {image_path}")
                text_parts.append(f'<image src="{image_path}">{image_text}</image>')
            else:
                # 其他元素直接放入文本
                text_parts.append(getattr(element, 'text', str(element)))

        texts.append("".join(text_parts))

    return texts


def prompt_func(data_dict):
    """构建提示词"""
    question = data_dict["question"]
    context = data_dict["context"]

    # 合并上下文
    formatted_texts = "\n\n".join(context)

    prompt = (
        "你是一位专业的文档分析与问答助手。你的任务是根据提供的参考资料回答用户问题。"
        "你将获得相关文档作为参考的上下文。这些文档是根据用户输入的关键词从向量数据库中检索获取的。"
        "请根据提供的文档结合你的分析能力，提供一份全面、准确的问题解答。"
        "请返回Markdown格式数据，并且当涉及到数学公式时，请使用正确的LaTeX语法来编写这些公式。"
        "对于行内公式应该以单个美元符号`$`；对于独立成行的公式，使用双美元符号`$$`包裹。"
        "例如，行内公式：`$a = \\frac{1}{2}$`，而独立成行的公式则是：`$$ a = \\frac{1}{2} $$`。"
        "请将提供的文档中的图片`<image src='...'>`（不包括`<image>`到`</image>`中间的文字），"
        "在需要时自然融入答案阐述的对应位置进行合理排版。\n\n"
        f"用户的问题是：\n{question}\n\n"
        "参考的文本或者表格数据：\n"
        f"{formatted_texts}"
    )

    print("=" * 150)
    print("完整提示词:\n", prompt)
    print("=" * 150)

    return prompt


def build_rag_chain(config, retriever):
    """构建RAG处理链"""
    # 使用文本大模型（成本更低）
    llm = init_llm(config=config)

    # 构建RAG链
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(split_image_text_types)
        }
        | RunnableLambda(prompt_func)
        | llm
        | StrOutputParser()
    )

    return chain


def display_answer(text: str):
    """
    显示带图片的答案

    解析<image src="...">标签并渲染图片
    """
    # 正则表达式匹配 <image src="xxx"> 标签
    pattern = r'(<image src="([^"]*)">)'
    chunks = re.split(pattern, text)

    for i, chunk in enumerate(chunks):
        if i % 3 == 0:
            # 文本内容
            display(Markdown(chunk.replace("</image>", "")))
        elif i % 3 == 2:
            # 图片路径 (i % 3 == 1 是完整标签，i % 3 == 2 是src值)
            try:
                img_base64 = encode_image(chunk)
                display(HTML(f'\n<img src="data:image/jpeg;base64,{img_base64}"/>\n'))
            except Exception as e:
                print(f"加载图片失败 {chunk}: {e}")


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
    pdf_data = parse_pdf(config)

    # ============================================
    # 三、生成摘要
    # ============================================
    summaries = generate_summaries(pdf_data, config)

    # ============================================
    # 四、生成图片摘要并融入文档
    # ============================================
    image_summaries = generate_image_summaries(pdf_data, config)

    # ============================================
    # 五、构建索引
    # ============================================
    retriever = build_multivector_retriever(config, summaries, pdf_data)

    # ============================================
    # 六、构建RAG Pipeline
    # ============================================
    chain = build_rag_chain(config, retriever)

    # ============================================
    # 七、效果展示
    # ============================================
    queries = [
        "介绍下电路模型",
        "再详细介绍下什么是理想电压源",
        "电流是什么"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"查询: {query}")
        print(f"{'='*80}\n")

        result = chain.invoke(query)
        display_answer(result)


if __name__ == "__main__":
    main()
