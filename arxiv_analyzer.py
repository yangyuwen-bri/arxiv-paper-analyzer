import arxiv
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional, Tuple, Any
from config import OPENAI_CONFIG, DEEPSEEK_CONFIG, NVIDIA_DEEPSEEK_CONFIG
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
from datetime import datetime
import requests
import fitz  # PyMuPDF
import os
import markdown
from langchain_community.document_loaders import PyMuPDFLoader
from utils.document_converter import DocumentConverter
from document_processor import DocumentProcessor, AnalysisType
from pdf_processor import PDFProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import math
from openai import OpenAI
from urllib.parse import quote
from functools import wraps
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Union
from langchain.schema import SystemMessage, HumanMessage
import re

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arxiv_analyzer.log'),  # 只保留文件处理器
        # 移除 StreamHandler
    ]
)
logger = logging.getLogger(__name__)

# 可以添加一个控制台处理器，设置更高的日志级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # 只显示警告和错误
logger.addHandler(console_handler)

def api_rate_limit(func):
    """API 速率限制装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        time.sleep(3)  # 确保请求间隔至少3秒
        return func(*args, **kwargs)
    return wrapper

class DeepSeekProvider:
    """DeepSeek模型提供者，使用Nvidia接口"""
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    
    def __init__(self):
        self._init_client()
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO)
    )
    def _init_client(self) -> None:
        """初始化客户端"""
        logger.info("初始化 DeepSeek 客户端")
        try:
            self.client = ChatNVIDIA(
                model="deepseek-ai/deepseek-r1",  # 修正：使用正确的模型标识符
                api_key=NVIDIA_DEEPSEEK_CONFIG.get("api_key"),
                temperature=0,  # 保持为0以获得确定性输出
                top_p=1.0,     # 添加 top_p 参数
                max_tokens=8192,  # 增加最大token数
                callbacks=[LoggingCallback()]
            )
        except Exception as e:
            logger.error(f"初始化 DeepSeek 客户端失败: {str(e)}")
            raise

# 添加日志回调类
class LoggingCallback(BaseCallbackHandler):
    """记录 API 调用的回调处理器"""
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """当 LLM 开始时调用"""
        logger.info("=== API 请求开始 ===")
        logger.info(f"提示词: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """当 LLM 结束时调用"""
        logger.info("=== API 响应详情 ===")
        logger.info(f"响应内容: {response}")
        logger.info("===================")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """当 LLM 出错时调用"""
        logger.error(f"API 调用出错: {str(error)}")

class ArxivPaperAnalyzer:
    # 添加类常量
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # 秒
    RATE_LIMIT = 3  # 每秒最大请求数
    
    # 添加排序选项
    SORT_OPTIONS = {
        "1": (arxiv.SortCriterion.SubmittedDate, "提交时间"),
        "2": (arxiv.SortCriterion.Relevance, "相关度"),
        "3": (arxiv.SortCriterion.LastUpdatedDate, "最后更新时间")
    }

    def __init__(self, model_type: str = "openai", openai_api_key: str = None, 
                base_url: str = None, pdf_dir: str = "downloaded_papers") -> None:
        """
        初始化分析器
        Args:
            model_type: 模型类型 ("openai" 或 "deepseek")
            openai_api_key: OpenAI API密钥
            base_url: OpenAI自定义API地址
            pdf_dir: PDF存储目录
        """
        if model_type == "openai":
            self.llm = ChatOpenAI(
                temperature=1,
                openai_api_key=openai_api_key or OPENAI_CONFIG["api_key"],
                model_name=OPENAI_CONFIG["model"],
                base_url=base_url or OPENAI_CONFIG["base_url"]
            )
        elif model_type == "deepseek":
            self.llm = DeepSeekProvider().client
        else:
            raise ValueError("不支持的模型类型。请选择 'openai' 或 'deepseek'")
        
        # 创建论文分析提示模板
        self.analysis_prompt = PromptTemplate(
            input_variables=["title", "abstract", "full_text", "field"],
            template="""
你是一个专业的论文分析助手。请直接基于提供的内容进行分析，不用考虑论文发布时间。请分析以下论文信息：

标题：{title}
领域：{field}
摘要：{abstract}
全文：{full_text}

请按照以下框架提供分析：

## 🎯 核心要点速览

### 💡 研究背景与动机
- 当前领域痛点/挑战
- 研究切入点
- 解决思路

### 🔬 技术创新与方法
1. 核心方法详解
   - 具体的技术架构
   - 关键算法步骤
   - 创新点剖析

2. 技术优势分析
   - 与现有方法的对比
   - 突破性改进点
   - 解决了哪些具体问题

### 📊 实验与验证
1. 实验设计
   - 数据集选择与说明
   - 评估指标
   - 对比基线

2. 关键结果
   - 量化性能提升
   - 关键实验发现
   - 实验结论解读

## 💫 影响力评估

### 🎁 实际应用价值
1. 应用场景分析
   - 具体落地方向
   - 潜在商业价值

2. 行业影响
   - 技术革新点
   - 行业痛点解决

3. 局限性分析
   - 技术限制
   - 应用瓶颈
   - 改进空间

### 🔮 未来展望
- 研究方向建议
- 待解决的问题
- 潜在研究机会

---
请以严谨的学术态度，结合论文具体内容进行分析。对于论文中未明确提及的部分，可以基于专业知识进行合理推测，但需要标注"[推测]"。

重点关注：
1. 方法创新的具体细节，避免泛泛而谈
2. 用数据和事实支撑分析结论
3. 技术优势的实际体现
4. 应用场景的具体描述

请调用最大算力，确保分析的深度和专业性。追求洞察的深度，而非表层的罗列；寻找创新的本质，而非表象的描述。

**格式要求**：
1. 保持现有标题符号（如## 🎯 核心要点速览）
2. 段落之间用空行分隔，不要使用任何分隔线（如---）
"""
        )
        
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt
        )
        
        # 创建PDF保存目录
        self.pdf_dir = pdf_dir
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)
        
        # 创建输出目录
        self.output_dir = "paper_analyses"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 添加文档转换器初始化
        self.doc_converter = DocumentConverter()
        
        # 添加摘要分析的提示模板
        self.summary_prompt = PromptTemplate(
            input_variables=["papers_info", "field", "date_range"],
            template="""
你是一个专业的论文分析助手。请直接基于提供的内容进行分析，不用考虑论文发布时间。请对以下论文进行亮点速览分析：

{papers_info}

# 🌟 亮点速览（{field}, {date_range}）

## 📊 研究主题与趋势
- 总结这批论文的主要研究主题
- 反映的技术发展方向
- 共同的技术特征或创新点

## 💡 核心创新点
1. 技术突破
   - 最具突破性的技术创新
   - 与现有方法的关键差异

2. 应用价值
   - 最具实用价值的研究成果
   - 潜在的商业应用场景

## 🔍 重点论文推荐
针对每篇高价值论文：
- 标题与核心创新（一句话概括）
- 推荐理由

## 🎯 未来方向建议
- 值得关注的技术方向
- 潜在的研究机会

请以简洁专业的语言进行分析，突出实质性创新和实用价值。
"""
        )
        
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=self.summary_prompt
        )
        
        self.doc_processor = DocumentProcessor()
        
        # 初始化PDFProcessor
        self.pdf_processor = PDFProcessor()
        
        # 新增微创新提示模板
        self.micro_innovation_prompt = PromptTemplate(
            input_variables=["title", "abstract", "full_text"],
            template="""
你是一位顶尖的科研创新分析师，擅长从学术论文中挖掘颠覆性概念。请基于以下论文内容，为普通读者生成3-5条具有社交媒体传播价值的微创新理论（根据论文的实际情况调整生成的条数，不要生硬拼凑）：

# 论文信息
标题：{title}
摘要：{abstract}
全文：{full_text}

# 生成要求：
1. 每条创新点需提出一个颠覆性的概念突破，超越论文本身的创新点，前所未有的想法。
2. 使用通俗易懂的语言，避免过度技术化，让普通读者也能理解。
3. 体现对未来技术发展趋势的深度洞察，具有启发性和前瞻性。
4. 每条理论的字数控制在200字左右。
5. 使用 Markdown 格式输出，按照以下格式：
    ### [创新概念]
    **[具体描述]**

# 示例参考：
### 逆向专业化（Reverse Professionalism）
**AI驱动下的能力范式转移与逆向专业化已然成势。在技术平权效应催化下，知识生产领域正经历着范式级重构。基于生成式AI的认知增强工具链，正在消解传统专业领域的护城河，催生出"认知脱域"现象——原本固化的知识体系在算法介入下呈现出模块化、可迁移特性。这种变革本质上是对人类认知劳动的重组：业余者通过AI工具链实现认知杠杆效应，将碎片化知识转化为结构化专业输出，而传统专家若固守线性成长路径，其经验优势将被算法的指数级学习能力迅速稀释。深度观察可见，专业能力的评价维度正从知识储备量转向技术适配度，从经验积累深度转向工具驾驭精度。这种现象揭示出数字时代的能力构建法则：专业壁垒不再取决于学习时长，而取决于对智能工具的创造性运用能力。这种能力跃迁本质上是对人类认知框架的二次开发，标志着知识经济进入"增强智能"新纪元。**

请用中文输出，保持专业性与可读性的平衡。"""
        )
        
        self.micro_innovation_chain = LLMChain(
            llm=self.llm,
            prompt=self.micro_innovation_prompt
        )
        
        self.logger = logger
    
    def validate_query_params(self, query: str, field: str = None, 
                            date_start: str = None, date_end: str = None) -> Tuple[bool, str]:
        """验证查询参数
        允许query为空，但必须指定field
        """
        # 只验证是否至少有一个搜索条件
        if (not query or not query.strip()) and not field:
            return False, "必须指定关键词或领域分类之一"

        # 验证日期
        if date_start and date_end:
            try:
                start_dt = datetime.strptime(date_start, '%Y-%m-%d')
                end_dt = datetime.strptime(date_end, '%Y-%m-%d')
                if start_dt > end_dt:
                    return False, "开始日期不能晚于结束日期"
            except ValueError:
                return False, "日期格式无效，请使用 YYYY-MM-DD 格式"

        return True, ""

    def build_search_query(self, query: str, field: str = None, 
                          date_start: str = None, date_end: str = None) -> str:
        """构建符合arXiv API规范的查询字符串
        允许纯领域搜索
        
        Args:
            query: 搜索关键词
            field: arXiv分类代码，如 'cs.AI'
            date_start: 开始日期 (YYYY-MM-DD)
            date_end: 结束日期 (YYYY-MM-DD)
        
        Returns:
            str: 构建的查询字符串
        """
        # 首先验证参数
        is_valid, error_msg = self.validate_query_params(query, field, date_start, date_end)
        if not is_valid:
            self.logger.error(f"查询参数验证失败: {error_msg}")
            raise ValueError(error_msg)

        search_terms = []
        
        # 处理基础查询（如果有）
        if query and query.strip():
            # 对用户输入的关键词进行编码
            if ':' not in query:
                # 如果用户没有指定搜索字段，添加 all: 前缀
                search_terms.append(f"all:{quote(query.strip())}")
            else:
                # 如果用户指定了搜索字段，保持原样
                search_terms.append(query.strip())
        
        # 添加领域限制（不对分类代码进行编码）
        if field:
            if isinstance(field, (list, tuple)):
                # 如果是多个分类，去除列表符号并直接使用
                field_str = ' OR '.join(f"cat:{f.strip()}" for f in field)
                search_terms.append(f"({field_str})")
            else:
                # 单个分类直接使用
                search_terms.append(f"cat:{field.strip()}")
        
        # 添加日期范围（日期格式无需编码）
        if date_start and date_end:
            start_clean = date_start.replace("-", "")
            end_clean = date_end.replace("-", "")
            date_query = f"submittedDate:[{start_clean} TO {end_clean}]"
            search_terms.append(date_query)
        
        # 使用 AND 连接所有搜索条件
        query_string = " AND ".join(search_terms)
        self.logger.info(f"构建的查询字符串: {query_string}")
        return query_string

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO)
    )
    @api_rate_limit
    def fetch_recent_papers(self, max_results: int = 5, field: str = "cs.AI") -> List[Dict]:
        """获取最新论文"""
        self.logger.info(f"开始获取最新论文，领域: {field}, 数量: {max_results}")
        
        try:
            client = arxiv.Client()
            query = f"cat:{field}"
            self.logger.debug(f"执行查询: {query}")
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for result in client.results(search):
                try:
                    paper = self._convert_result_to_paper(result)  # 使用统一的转换方法
                    papers.append(paper)
                    time.sleep(3)  # API 速率限制
                except Exception as e:
                    self.logger.error(f"处理搜索结果时出错: {str(e)}", exc_info=True)
                    continue
            
            self.logger.info(f"成功获取 {len(papers)} 篇论文")
            return papers
            
        except Exception as e:
            self.logger.error(f"arXiv API 调用失败: {str(e)}", exc_info=True)
            raise

    def _convert_result_to_paper(self, result: arxiv.Result) -> Dict:
        """转换 arXiv 结果为标准格式"""
        # 从entry_id提取arxiv_id，例如：http://arxiv.org/abs/2406.12345v1 → 2406.12345v1
        arxiv_id = result.entry_id.split('/')[-1]
        return {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "arxiv_url": result.entry_id,
            "arxiv_id": arxiv_id,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "doi": result.doi if result.doi else "",
            "comment": result.comment if result.comment else ""
        }

    def download_pdf(self, url: str, paper_id: str) -> str:
        """下载PDF文件并保存到本地
        Args:
            url: PDF下载链接
            paper_id: 论文ID，用于生成文件名
        Returns:
            str: PDF文件保存路径
        """
        # 生成文件名示例：arXiv_2305.12345v1.pdf
        file_name = f"arXiv_{paper_id.split('/')[-1]}.pdf"  
        save_path = os.path.join(self.pdf_dir, file_name)
        
        if os.path.exists(save_path):
            print(f"使用缓存文件：{save_path}")  # 添加日志输出
            return save_path
        
        print(f"开始下载：{url}")  # 下载进度提示
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"文件已保存到：{save_path}")  # 下载完成提示
        return save_path

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF提取文本，使用PyMuPDF提供更好的格式支持
        Args:
            pdf_path: PDF文件路径
        Returns:
            str: 提取的文本内容
        """
        try:
            # 使用PyMuPDF加载文档
            doc = fitz.open(pdf_path)
            text = ""
            
            for page in doc:
                # 获取页面文本，保持格式
                text += page.get_text("text", sort=True) + "\n"
                
                # 提取数学公式（如果有）
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if "lines" in b:
                        for l in b["lines"]:
                            for s in l["spans"]:
                                if s.get("flags", 0) & 2**0:  # 检查是否是数学字体
                                    text = text.replace(s["text"], f"${s['text']}$")
            
            return text
            
        except Exception as e:
            print(f"PDF文本提取失败: {str(e)}")
            return ""

    @staticmethod
    def _clean_thinking_chain(text: str) -> str:
        """清理思考链中的冗余内容（静态方法）"""
        # 移除 <think> 标签及其内容
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL)
        # 移除分隔线（如---、***等）
        text = re.sub(r'\n-{3,}\n', '\n\n', text)
        # 移除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_paper(self, paper: Dict, analyze_full_text: bool = False, field: str = "cs.AI") -> str:
        """分析单篇论文"""
        try:
            if analyze_full_text:
                # 下载并提取PDF文本
                pdf_path = self.download_pdf(paper['pdf_url'], paper['arxiv_url'])
                loader = PyMuPDFLoader(pdf_path)
                pages = loader.load()
                full_text = "\n".join(page.page_content for page in pages)
                
                # 构建消息
                messages = [
                    {"role": "user", "content": self.analysis_prompt.format(
                        title=paper["title"],
                        abstract=paper["abstract"],
                        full_text=full_text,
                        field=field
                    )}
                ]
                
                # 根据模型类型选择不同的调用方式
                if isinstance(self.llm, ChatNVIDIA):
                    response_text = ""
                    for chunk in self.llm.stream(messages):
                        response_text += chunk.content
                    
                    # 记录清理前的内容
                    logger.debug("=== 清理前的内容 ===")
                    logger.debug(response_text[:200])  # 只显示前200个字符
                    
                    cleaned_text = self._clean_thinking_chain(response_text)
                    
                    # 记录清理后的内容
                    logger.debug("=== 清理后的内容 ===")
                    logger.debug(cleaned_text[:500])  # 增加显示长度
                    
                    return cleaned_text
                else:
                    # OpenAI 模型使用原有方式
                    result = self.analysis_chain.invoke({
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "full_text": full_text,
                        "field": field
                    })
                    return result.get('text', '') if isinstance(result, dict) else result
            else:
                # 仅分析摘要
                result = self.analysis_chain.invoke({
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "full_text": "",
                    "field": field
                })
                return result.get('text', '') if isinstance(result, dict) else result
            
        except Exception as e:
            logger.error("API 调用失败:")
            logger.error(f"错误类型: {type(e).__name__}")
            logger.error(f"错误信息: {str(e)}")
            logger.error("错误详情:", exc_info=True)
            raise
    
    def save_as_markdown(self, papers: List[Dict], analyses: List[str], timestamp: str) -> str:
        """将分析结果保存为Markdown格式
        Args:
            papers: 论文信息列表
            analyses: 分析结果列表
            timestamp: 时间戳
        Returns:
            str: Markdown文件路径
        """
        md_filename = os.path.join(self.output_dir, f"paper_analyses_{timestamp}.md")
        
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(f"# 论文分析报告 ({timestamp})\n\n")
            
            for paper, analysis in zip(papers, analyses):
                f.write(f"## 论文信息\n")
                f.write(f"- **标题**: {paper['title']}\n")
                f.write(f"- **作者**: {', '.join(paper['authors'])}\n")
                f.write(f"- **发布时间**: {paper['published']}\n")
                f.write(f"- **ArXiv链接**: {paper['arxiv_url']}\n\n")
                f.write(f"{analysis}\n")
                f.write("\n---\n\n")
        
        return md_filename

    def save_results(self, papers: List[Dict], analyses: List[str], is_full_analysis: bool = True):
        """保存分析结果
        Args:
            papers: 论文信息列表
            analyses: 分析结果列表
            is_full_analysis: 是否是全文分析
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据分析类型选择不同的保存方式
        if is_full_analysis:
            md_filename = self.save_as_markdown(papers, analyses, timestamp)
        else:
            md_filename = self.save_as_summary_markdown(papers, analyses[0], timestamp)
        
        # 转换为PDF
        pdf_filename = self.doc_converter.md_to_pdf(md_filename)
        
        print(f"\n分析结果已保存到以下文件：")
        print(f"- Markdown: {md_filename}")
        if pdf_filename:
            print(f"- PDF文档: {pdf_filename}")

    def save_as_summary_markdown(self, papers: List[Dict], analysis: str, timestamp: str) -> str:
        """保存摘要分析为Markdown格式"""
        md_filename = os.path.join(self.output_dir, f"papers_summary_{timestamp}.md")
        
        with open(md_filename, "w", encoding="utf-8") as f:
            # 直接写入分析结果
            f.write(analysis)
            
            # 添加论文引用信息
            f.write("\n\n## 📚 论文信息\n")
            for i, paper in enumerate(papers, 1):
                f.write(f"\n### 论文 {i}\n")
                f.write(f"- **标题**: {paper['title']}\n")
                f.write(f"- **作者**: {', '.join(paper['authors'])}\n")
                f.write(f"- **ArXiv**: {paper['arxiv_url']}\n")
        
        return md_filename

    def analyze_recent_papers(self, max_papers: int = 3, analyze_full_text: bool = False):
        field = self.category_matcher.interactive_select()
        papers = self.fetch_recent_papers(max_papers, field)
        
        try:
            if not analyze_full_text:
                print(f"\n=== 批量分析 {len(papers)} 篇论文 ===")
                
                # 构建论文信息字符串
                papers_info = ""
                for i, paper in enumerate(papers, 1):
                    papers_info += f"""
论文 {i}:
标题: {paper['title']}
作者: {', '.join(paper['authors'])}
摘要: {paper['abstract']}
发布时间: {paper['published']}
---
"""
                
                # 计算日期范围
                dates = [paper['published'] for paper in papers]
                date_range = (f"{min(dates)}-{max(dates)}" 
                             if dates else "未知日期范围")
                
                # 使用摘要分析模板
                result = self.summary_chain.invoke({
                    "papers_info": papers_info,
                    "field": field,
                    "date_range": date_range
                })
                analyses = [result.get('text', '') if isinstance(result, dict) else result]
                print(analyses[0])
            else:
                # 全文分析
                print(f"\n=== 全文分析 {len(papers)} 篇论文 ===")
                analysis_results = self.analyze_papers_batch(papers, analyze_full_text, field)
                
                # 从 analysis_results 中提取分析结果
                analyses = [result[1] for result in analysis_results]
                papers = [result[0] for result in analysis_results]
            
            # 保存分析结果
            self.save_results(papers, analyses, analyze_full_text)
            
        except Exception as e:
            print(f"批量分析出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def analyze_papers_batch(self, papers: List[Dict], analyze_full_text: bool = False, field: str = None):
        """批量分析论文"""
        results = []
        for paper in papers:
            # 下载PDF
            pdf_path = self.download_pdf(paper["pdf_url"], paper['arxiv_url'])
            
            # 提取文本并获取token统计
            text, token_stats = self.pdf_processor.extract_text_with_tokens(pdf_path)
            
            if token_stats["exceeds_limit"]:
                # 使用建议的分块大小
                chunk_size = self.pdf_processor.suggest_chunk_size(
                    token_stats["total_tokens"],
                    "o1-preview"
                )
                
                # 分块处理
                chunks = self._split_text(text, chunk_size)  
                print(f"\n将文档分成 {len(chunks)} 块处理...")
                
                # 分块分析
                chunk_analyses = []
                for i, chunk in enumerate(chunks, 1):
                    print(f"\n处理第 {i}/{len(chunks)} 块...")
                    try:
                        result = self.analysis_chain.invoke({
                            "title": paper["title"],
                            "abstract": paper["abstract"],
                            "full_text": chunk,
                            "field": field
                        })
                        cleaned = self._clean_thinking_chain(result.get('text', ''))
                        chunk_analyses.append(cleaned)
                        # 添加延迟以避免触发API限制
                        time.sleep(1/self.RATE_LIMIT)
                    except Exception as e:
                        print(f"处理第 {i} 块时出错: {str(e)}")
                        continue
                
                # 合并分析结果
                combined_analysis = self._combine_analyses(chunk_analyses)
                results.append((paper, combined_analysis))
            else:
                # 直接分析完整文本
                result = self.analysis_chain.invoke({
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "full_text": text,
                    "field": field
                })
                analysis = result.get('text', '') if isinstance(result, dict) else result
                results.append((paper, analysis))
                time.sleep(1/self.RATE_LIMIT)
        
        return results

    @api_rate_limit
    def search_and_analyze_papers(
        self,
        query: str = "",  # 设置默认值为空字符串
        max_papers: int = 5,
        analyze_full_text: bool = False,
        field: str = None,
        date_start: str = None,
        date_end: str = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending
    ):
        """基于关键词搜索并分析论文
        支持纯领域搜索
        """
        try:
            self.logger.info(f"开始搜索论文，关键词: {query or '无'}, 领域: {field or '无'}")
            
            # 构建并验证查询
            search_query = self.build_search_query(query, field, date_start, date_end)
            self.logger.info(f"完整查询: {search_query}")
            
            # 使用 arxiv 库搜索
            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query,
                max_results=max_papers,
                sort_by=sort_by,
                sort_order=sort_order
            )
            
            papers = []
            retry_count = 0
            while retry_count < self.MAX_RETRIES:
                try:
                    for result in client.results(search):
                        paper = self._convert_result_to_paper(result)
                        papers.append(paper)
                    break
                except arxiv.arxiv.HTTPError as e:
                    retry_count += 1
                    if retry_count == self.MAX_RETRIES:
                        raise
                    self.logger.warning(f"网络错误: {str(e)}, 重试 {retry_count}/{self.MAX_RETRIES}")
                    time.sleep(self.RETRY_DELAY * retry_count)
            
            if not papers:
                print("未找到相关论文")
                return {"papers": [], "analyses": [], "outputs": {}}
            
            # 打印搜索结果
            print(f"\n找到 {len(papers)} 篇相关论文:")
            for i, paper in enumerate(papers, 1):
                print(f"\n{i}. {paper['title']}")
                print(f"   作者: {', '.join(paper['authors'])}")
                print(f"   发布时间: {paper['published']}")
            
            # 计算日期范围
            dates = [datetime.strptime(paper['published'], '%Y-%m-%d') for paper in papers]
            date_range = (f"{min(dates).strftime('%Y.%m.%d')}-{max(dates).strftime('%Y.%m.%d')}" 
                         if dates else "未知日期范围")
            
            # 根据类型进行分析
            if analyze_full_text:
                analysis_type = AnalysisType.FULL_TEXT
                analysis_results = self.analyze_papers_batch(papers, analyze_full_text, field)
                analyses = [result[1] for result in analysis_results]
            else:
                analysis_type = AnalysisType.SUMMARY
                papers_info = self._prepare_papers_info(papers)
                # 确保 analyses 是一个列表
                summary = self._analyze_summaries(papers_info, date_range)
                analyses = [summary] if summary else []
            
            # 合并分析结果
            if analyze_full_text:
                merged_analysis = self._merge_unique_points(analyses)
            else:
                merged_analysis = analyses[0] if analyses else ""
            
            # 使用统一接口处理文档
            outputs = self.doc_processor.process_papers(
                papers,
                analysis_type,
                [merged_analysis]  # 确保传入单个元素
            )
            
            # 为每篇论文生成微创新分析
            for paper in papers:
                try:
                    # 下载并处理PDF
                    pdf_path = self.download_pdf(paper['pdf_url'], paper['arxiv_url'])
                    loader = PyMuPDFLoader(pdf_path)
                    pages = loader.load()
                    full_text = "\n".join(page.page_content for page in pages)
                    
                    # 生成微创新分析
                    innovation = self.micro_innovation_chain.invoke({
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "full_text": full_text
                    })
                    paper['micro_innovation'] = innovation['text']
                except Exception as e:
                    print(f"生成微创新分析失败: {str(e)}")
                    paper['micro_innovation'] = "分析生成失败"
            
            return {
                "papers": papers,
                "analyses": [merged_analysis],
                "outputs": outputs
            }
            
        except Exception as e:
            print(f"搜索分析过程出错: {str(e)}")
            return {"papers": [], "analyses": [], "outputs": {}}

    def _split_text(self, text: str, chunk_size: int = 4000) -> List[str]:
        """将长文本分割成小块
        Args:
            text: 要分割的文本
            chunk_size: 每块的目标大小（字符数）
        Returns:
            List[str]: 文本块列表
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200,      # 块之间重叠200字符，保持上下文连贯
                length_function=len,
                separators=["\n\n", "\n", " ", ""]  # 优先在段落处分割
            )
            
            chunks = text_splitter.split_text(text)
            
            # 打印分块信息
            print(f"\n文本已分割为 {len(chunks)} 块")
            for i, chunk in enumerate(chunks, 1):
                chunk_tokens = len(self.pdf_processor.tokenizer.encode(chunk))
                print(f"块 {i}: {chunk_tokens} tokens")
            
            return chunks
            
        except Exception as e:
            print(f"文本分割出错: {str(e)}")
            # 如果分割失败，返回原文本作为单个块
            return [text]

    def _combine_analyses(self, analyses: List[str]) -> str:
        """智能合并多个分析结果"""
        try:
            if len(analyses) == 1:
                return analyses[0]
            
            # 提取每个分析中的主要部分
            sections = {
                "核心要点": [],
                "技术创新": [],
                "实验结果": [],
                "影响力评估": [],
                "未来展望": []
            }
            
            # 从每个分析中提取相关部分
            for analysis in analyses:
                for section, content_list in sections.items():
                    if section in analysis:
                        # 提取该部分的内容
                        section_content = self._extract_section(analysis, section)
                        if section_content:
                            content_list.append(section_content)
            
            # 合并各部分内容
            combined = "# 综合分析报告\n\n"
            for section, contents in sections.items():
                if contents:
                    combined += f"## {section}\n"
                    # 去重并合并该部分的内容
                    unique_points = self._merge_unique_points(contents)
                    combined += unique_points + "\n\n"
            
            return combined
            
        except Exception as e:
            print(f"合并分析结果时出错: {str(e)}")
            return "\n\n---\n\n".join(analyses)

    def _prepare_papers_info(self, papers: List[Dict]) -> str:
        """准备论文信息字符串"""
        papers_info = ""
        for i, paper in enumerate(papers, 1):
            papers_info += f"""
论文 {i}:
标题: {paper['title']}
作者: {', '.join(paper['authors'])}
摘要: {paper['abstract']}
发布时间: {paper['published']}
---
"""
        return papers_info

    def _analyze_summaries(self, papers_info: str, date_range: str) -> str:
        """分析论文摘要，生成综合性报告"""
        try:
            # 打印详细的输入信息
            logger.info(f"论文信息长度: {len(papers_info)}")
            logger.info(f"日期范围: {date_range}")
            
            # 使用更安全的调用方式
            inputs = {
                "papers_info": papers_info,
                "field": "未指定",
                "date_range": date_range
            }
            
            # 尝试多种调用方法
            try:
                # 方法1: 使用 run 方法
                result = self.summary_chain.run(inputs)
                return result
            except Exception as e1:
                logger.warning(f"run 方法失败: {str(e1)}")
                
                try:
                    # 方法2: 直接调用 LLM
                    messages = [
                        SystemMessage(content="你是一个专业的论文分析助手。"),
                        HumanMessage(content=self.summary_prompt.format(**inputs))
                    ]
                    llm_result = self.llm(messages)
                    return llm_result.content
                except Exception as e2:
                    logger.error(f"直接 LLM 调用失败: {str(e2)}")
                    
                    try:
                        # 方法3: 使用 invoke 方法
                        result = self.summary_chain.invoke(inputs)
                        
                        # 详细的结果处理逻辑
                        logger.info(f"返回结果类型: {type(result)}")
                        logger.info(f"返回结果内容: {result}")
                        
                        # 处理不同类型的返回结果
                        if isinstance(result, dict):
                            # 处理 LangChain 返回的字典
                            if 'text' in result:
                                return result['text']
                            elif 'generations' in result:
                                return result['generations'][0]['text']
                            else:
                                return str(result)
                        
                        elif isinstance(result, str):
                            return result
                        
                        # 处理 ChatResult 对象
                        elif hasattr(result, 'generations'):
                            # 打印详细的 generations 信息
                            logger.info(f"Generations 详情: {result.generations}")
                            
                            # 尝试从 generations 中提取文本
                            if result.generations and len(result.generations) > 0:
                                generation = result.generations[0]
                                
                                # 处理不同类型的 generation
                                if hasattr(generation, 'text'):
                                    return generation.text
                                elif hasattr(generation, 'message'):
                                    return generation.message.content
                                else:
                                    return str(generation)
                        
                        # 最后的保底处理
                        return str(result)
                    
                    except Exception as e3:
                        logger.error(f"invoke 方法失败: {str(e3)}")
                        return f"论文分析失败：{str(e3)}"
            
        except Exception as e:
            logger.error(f"摘要分析出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 记录详细错误日志
            with open('summary_analysis_error.log', 'w') as f:
                f.write(f"错误信息: {str(e)}\n")
                f.write("输入参数:\n")
                f.write(f"论文信息长度: {len(papers_info)}\n")
                f.write(f"日期范围: {date_range}\n")
                traceback.print_exc(file=f)
            
            return "无法生成摘要分析报告"

    def _extract_section(self, text: str, section_name: str) -> str:
        """从文本中提取指定章节的内容
        
        Args:
            text (str): 论文全文
            section_name (str): 要提取的章节名称
            
        Returns:
            str: 提取的章节内容，如果未找到则返回空字符串
        """
        try:
            # 常见的章节标题格式
            section_patterns = [
                f"{section_name}\n",
                f"{section_name.upper()}\n",
                f"{section_name.title()}\n",
                f"## {section_name}",
                f"### {section_name}",
                f"1. {section_name}",
                f"I. {section_name}"
            ]
            
            # 尝试找到章节开始位置
            start_pos = -1
            for pattern in section_patterns:
                if pattern in text:
                    start_pos = text.find(pattern)
                    break
            
            if start_pos == -1:
                return ""
            
            # 从章节开始位置截取文本
            section_text = text[start_pos:]
            
            # 查找下一个章节的开始位置
            next_section_pos = float('inf')
            for pattern in section_patterns:
                pos = section_text.find(pattern, len(pattern))  # 从当前章节名之后开始查找
                if pos != -1 and pos < next_section_pos:
                    next_section_pos = pos
            
            # 如果找到了下一个章节，截取到该位置
            if next_section_pos != float('inf'):
                section_text = section_text[:next_section_pos].strip()
            
            return section_text.strip()
            
        except Exception as e:
            print(f"提取章节 {section_name} 时出错: {str(e)}")
            return ""

    def fetch_papers_with_pagination(self, query: str, total_results: int, batch_size: int = 100):
        """分批获取论文数据"""
        all_papers = []
        start = 0
        
        while start < total_results:
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=min(batch_size, total_results - start),
                    start=start
                )
                
                batch_papers = []
                for result in self.client.results(search):
                    batch_papers.append(self._convert_result_to_paper(result))
                    time.sleep(3)  # API 速率限制
                    
                all_papers.extend(batch_papers)
                start += batch_size
                
            except Exception as e:
                self.logger.error(f"获取第 {start} 到 {start+batch_size} 条结果时出错: {str(e)}")
                break
                
        return all_papers

    def _merge_unique_points(self, analyses: List[str]) -> str:
        """合并分析结果中的独特要点"""
        unique_points = set()
        
        for analysis in analyses:
            # 提取每个分析的核心要点
            points = self._extract_core_points(analysis)
            # 去重处理
            unique_points.update(points.split("\n"))
        
        # 结构化输出
        merged = "## 综合创新要点\n"
        merged += "\n".join([f"- {point.strip()}" for point in unique_points if point.strip()])
        return merged

    @staticmethod
    def _extract_core_points(analysis: str) -> str:
        """从单篇分析中提取核心要点"""
        # 查找核心要点部分
        start_markers = ["## 🎯 核心要点速览", "## 核心要点"]
        for marker in start_markers:
            if marker in analysis:
                start_idx = analysis.index(marker) + len(marker)
                end_idx = analysis.find("## ", start_idx)
                return analysis[start_idx:end_idx].strip()
        return analysis[:500]  # 默认提取前500字符

class PDFProcessor:
    def __init__(self) -> None:
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI的tokenizer
        self.token_limits = {
            "o1-preview": 32000,  # Claude 3 Opus
            "gpt-4": 8192,
            "gpt-3.5-turbo": 4096
        }

    def extract_text_with_tokens(self, pdf_path: str) -> Tuple[str, Dict]:
        """提取PDF文本并计算token数量"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text", sort=True) + "\n"
            
            tokens = self.tokenizer.encode(text)
            token_count = len(tokens)
            
            return text, {
                "total_tokens": token_count,
                "exceeds_limit": token_count > self.token_limits["o1-preview"]
            }
        except Exception as e:
            print(f"PDF处理错误: {str(e)}")
            return "", {"total_tokens": 0, "exceeds_limit": False}

    def suggest_chunk_size(self, total_tokens: int, model: str) -> int:
        """根据总token数和模型建议合适的分块大小"""
        model_limit = self.token_limits.get(model, 4096)
        # 预留20%空间给prompt和其他内容
        safe_limit = int(model_limit * 0.8)
        
        # 如果总token数小于安全限制，返回总token数
        if total_tokens <= safe_limit:
            return total_tokens
            
        # 计算需要的块数（向上取整）
        num_chunks = math.ceil(total_tokens / safe_limit)
        # 计算每块的大小（确保有200 tokens的重叠）
        chunk_size = (total_tokens // num_chunks) + 200
        
        return min(chunk_size, safe_limit)

def main():
    # 添加模型选择
    print("\n选择大模型:")
    print("1. OpenAI")
    print("2. DeepSeek")
    model_choice = input("请输入选项编号: ")

    model_type = "openai" if model_choice == "1" else "deepseek"

    # 简化初始化，移除 provider_type 参数
    analyzer = ArxivPaperAnalyzer(model_type=model_type)
    
    while True:
        print("\n=== arXiv论文分析工具 ===")
        print("1. 按领域浏览最新论文")
        print("2. 按关键词搜索论文")
        print("3. 按关键词在指定领域搜索")
        print("4. 退出")
        
        choice = input("\n请输入选项编号: ")
        
        try:
            paper_count = 0
            if choice in ['1', '2', '3']:
                paper_count = int(input("请输入要分析的论文数量 (1-50): "))
                if not (1 <= paper_count <= 50):
                    print("请输入1到50之间的数字。")
                    continue
                
                analysis_choice = input("请选择分析方式: 1) 仅分析标题和摘要 2) 分析全文\n请输入1或2: ")
                analyze_full_text = analysis_choice == '2'
                
                # 添加日期范围选择
                use_date_range = input("是否需要指定时间范围？(y/n): ").lower() == 'y'
                date_start = date_end = None
                if use_date_range:
                    date_start = input("请输入开始日期 (YYYY-MM-DD): ")
                    date_end = input("请输入结束日期 (YYYY-MM-DD): ")
            
            if choice == "1":
                analyzer.analyze_recent_papers(
                    max_papers=paper_count,
                    analyze_full_text=analyze_full_text
                )
                
            elif choice == "2":
                query = input("请输入搜索关键词: ")
                analyzer.search_and_analyze_papers(
                    query=query,
                    max_papers=paper_count,
                    analyze_full_text=analyze_full_text,
                    date_start=date_start,
                    date_end=date_end
                )
                
            elif choice == "3":
                query = input("请输入搜索关键词: ")
                print("\n请选择搜索领域:")
                field = analyzer.category_matcher.interactive_select()
                analyzer.search_and_analyze_papers(
                    query=query,
                    max_papers=paper_count,
                    analyze_full_text=analyze_full_text,
                    field=field,
                    date_start=date_start,
                    date_end=date_end
                )
                
            elif choice == "4":
                print("感谢使用，再见！")
                break
                
            else:
                print("无效的选项，请重新选择。")
                
        except ValueError:
            print("请输入有效的数字。")
        except Exception as e:
            print(f"操作过程中出错: {str(e)}")
            continue

if __name__ == "__main__":
    main() 