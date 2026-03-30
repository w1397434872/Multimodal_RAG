# 多模态RAG系统 (Multimodal RAG System)

一个基于 FastAPI 的多模态检索增强生成系统，支持三种不同的RAG方案，能够同时处理PDF文档中的文本和图片内容。

![1774847317792](image/README/1774847317792.png)

## 📋 项目简介

本项目实现了三种多模态RAG方案：

### 方案一：多模态向量模型

- **核心思路**：使用多模态向量模型同时处理文本和图片，将文本和图片嵌入到同一向量空间
- **适用场景**：PDF中图片具备丰富信息，能单独理解的场景
- **特点**：图片和文本分别检索，适合图片内容独立于文本也能被理解的场景

### 方案二：多模态大模型（摘要索引）

- **核心思路**：使用多模态大模型生成图片摘要，将图片摘要与文本摘要一起建立索引
- **适用场景**：需要精准检索效果的场景
- **特点**：检索效果更精准，但成本较高

### 方案三：平衡方案（推荐）

- **核心思路**：图片摘要融入文档上下文，使用文本大模型回答
- **适用场景**：需要平衡成本和效果的场景
- **特点**：平衡成本和效果，推荐日常使用

## ✨ 主要功能

- 📄 **PDF文档解析**：自动提取文本、表格和图片
- 🔍 **多模态检索**：同时检索文本和图片内容
- 💬 **智能问答**：基于检索结果生成答案
- 🌐 **Web界面**：美观的前端交互界面
- 💾 **索引持久化**：索引自动保存，服务重启不丢失
- 📊 **方案对比**：同时对比三种方案的效果

## 🛠️ 技术栈

- **后端**：FastAPI + Python 3.9+
- **前端**：HTML5 + CSS3 + JavaScript (原生)
- **向量数据库**：Chroma
- **文档解析**：Unstructured
- **嵌入模型**：DashScope 多模态嵌入
- **大语言模型**：通义千问 (Qwen)
- **OCR**：Tesseract
- **PDF处理**：Poppler

## 📦 安装方法

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/Multimodal_RAG.git
cd Multimodal_RAG
```

### 2. 创建Conda环境

```bash
conda create -n rag python=3.9
conda activate rag
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装外部工具

#### Windows

1. **安装 Poppler** (PDF处理)

   - 下载地址：https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip
   - 解压到 `tools/poppler` 目录
2. **安装 Tesseract** (OCR)

   - 下载地址： https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe 
   - 安装到 `tools/Tesseract-OCR` 目录

#### Linux/Mac

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr

# Mac
brew install poppler tesseract
```

### 5. 配置环境变量

编辑 `src/.env` 文件：

```env
# API密钥配置
DASHSCOPE_API_KEY=your_api_key_here

# 外部工具路径 (Windows示例)
POPPLER_PATH=D:\Multimodal_RAG\tools\poppler\bin
TESSERACT_PATH=D:\Multimodal_RAG\tools\Tesseract-OCR

# 模型配置
LLM_MODEL=qwen-max
VL_MODEL=qwen-vl-max
EMBEDDING_MODEL=text-embedding-v1
MULTIMODAL_EMBEDDING_MODEL=multimodal-embedding-v1

# HuggingFace镜像 (国内访问)
HF_ENDPOINT=https://hf-mirror.com
```

## 🚀 启动方法

### 方法一：一键启动（推荐）

双击运行 `start_all.bat`：

```bash
start_all.bat
```

这将同时启动：

- 后端服务：http://localhost:8000
- 前端服务：http://localhost:8080

### 方法二：分别启动

#### 启动后端服务

```bash
# Windows
start_server.bat

# 或手动
conda activate rag
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 启动前端服务

```bash
# Windows
start_web.bat

# 或手动
cd web
python -m http.server 8080
```

### 方法三：Docker启动（可选）

```bash
docker-compose up -d
```

## 📖 使用指南

### 1. 访问系统

打开浏览器访问：http://localhost:8080

### 2. 构建索引

1. 进入"索引管理"页面
2. 点击"选择文件"上传PDF文档
3. 选择一种方案（推荐"平衡方案"）
4. 点击"构建索引"按钮

### 3. 智能问答

1. 进入"智能问答"页面
2. 在输入框中输入问题
3. 点击发送或按回车键
4. 查看AI回答和参考来源

### 4. 方案对比

1. 进入"方案对比"页面
2. 输入查询问题
3. 点击"开始对比"
4. 查看三种方案的不同回答

## 📁 项目结构

```
Multimodal_RAG/
├── src/                      # 后端代码
│   ├── main.py              # FastAPI入口
│   ├── rag_service.py       # RAG服务实现
│   ├── schemas.py           # 数据模型
│   ├── common.py            # 公共函数
│   ├── RAG_image.py         # 方案一实现
│   ├── RAG_image_abstract.py # 方案二实现
│   ├── RAG_abstract.py      # 方案三实现
│   └── .env                 # 环境变量配置
├── web/                      # 前端代码
│   ├── index.html           # 主页面
│   ├── css/
│   │   └── style.css        # 样式文件
│   └── js/
│       └── app.js           # 交互逻辑
├── tools/                    # 外部工具
│   ├── poppler/             # PDF处理工具
│   └── Tesseract-OCR/       # OCR工具
├── resources/                # 上传的PDF文件
├── vectorstore/              # 索引存储目录
├── start_all.bat            # 一键启动脚本
├── start_server.bat         # 启动后端脚本
├── start_web.bat            # 启动前端脚本
└── README.md                # 项目说明
```

## 🔧 API文档

启动服务后访问：

- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

### 主要API接口

| 接口                | 方法   | 说明              |
| ------------------- | ------ | ----------------- |
| `/health`         | GET    | 健康检查          |
| `/index/status`   | GET    | 获取索引状态      |
| `/index`          | POST   | 构建索引          |
| `/index/upload`   | POST   | 上传PDF并构建索引 |
| `/index/{method}` | DELETE | 清空指定索引      |
| `/query`          | POST   | 查询问答          |
| `/query/all`      | POST   | 对比三种方案      |

## ⚠️ 注意事项

1. **API密钥**：需要配置有效的 DashScope API Key
2. **网络环境**：国内用户建议配置 HuggingFace 镜像
3. **内存要求**：建议至少 8GB 内存
4. **GPU支持**：可选，CPU 也可运行

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

**Star ⭐ 本项目如果对你有帮助！**
