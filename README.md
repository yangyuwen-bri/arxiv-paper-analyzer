# ArXiv 论文智能分析平台

## 🌟 项目简介

这是一个基于 Streamlit 的 ArXiv 论文智能分析工具，支持论文搜索、摘要和全文分析。利用先进的语言模型（OpenAI、DeepSeek）提供深入的论文洞察。

## ✨ 主要功能

- 🔍 灵活的 ArXiv 论文搜索
- 📄 论文摘要和全文智能分析
- 🤖 多模型支持（OpenAI、DeepSeek）
- 📊 可视化结果展示
- 🌐 支持多领域检索

## 🚀 快速开始

### 1. 克隆仓库
bash

git clone https://github.com/yourusername/arxiv-paper-analyzer.git

cd arxiv-paper-analyzer

### 2. 创建虚拟环境
bash

创建虚拟环境

python3 -m venv myenv

source myenv/bin/activate # macOS/Linux

### 3. 安装依赖
bash

pip install -r requirements.txt

### 4. 配置 API 密钥
bash

创建 .env 文件

touch .env

在 `.env` 文件中添加：

OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

### 5. 运行应用
bash
streamlit run app.py


## 🛠 配置说明

### API 配置

- 支持 OpenAI 和 DeepSeek 模型
- 通过环境变量管理 API 密钥
- 可在 `.env` 文件中配置

### 领域分类

项目使用arxiv官方领域分类arxiv_taxonomy.xlsx/arxiv_taxonomy_cn.xlsx。

## 📦 依赖库

- Streamlit
- ArXiv API
- LangChain
- PyMuPDF
- Tiktoken

