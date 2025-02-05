import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import re
import json
import logging
from pathlib import Path
import tiktoken

class PDFProcessor:
    def __init__(self, log_dir: str = "paper_process_logs"):
        # 定义常用的分隔标记
        self.section_markers = {
            'abstract': ['Abstract', 'ABSTRACT'],
            'introduction': ['Introduction', 'INTRODUCTION', '1. Introduction', 'I. Introduction'],
            'keywords': ['Keywords:', 'KEYWORDS:', 'Key words:', 'Index Terms'],
            'references': ['References', 'REFERENCES', 'Bibliography']
        }
        
        # 作者相关标记
        self.author_markers = {
            'start': ['Author', 'Authors', '∗', '*', '†', '1,', '1', 'a,', 'a'],
            'email': ['@', 'email:', 'Email:', 'E-mail:'],
            'affiliation': ['University', 'Institute', 'Laboratory', 'College', 'Corp.', 'Inc.']
        }
        
        # 设置日志
        self.log_dir = log_dir
        self._setup_logging()
        
        # 创建结果保存目录
        self.results_dir = os.path.join(log_dir, "extraction_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # 设置token限制
        self.TOKEN_LIMITS = {
            "gpt-4": 8192,           # GPT-4的上下文窗口
            "gpt-3.5": 4096,         # GPT-3.5的上下文窗口
            "o1-preview": 32768      # Claude的上下文窗口
        }

    def _setup_logging(self):
        """设置日志记录"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'pdf_processing.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_paper_info(self, pdf_path: str) -> Optional[Dict]:
        """从PDF文件中提取论文信息
        Args:
            pdf_path: PDF文件路径
        Returns:
            Dict: 包含论文信息的字典，如果处理失败则返回None
        """
        try:
            self.logger.info(f"开始处理PDF文件: {pdf_path}")
            
            # 打开PDF文件
            doc = fitz.open(pdf_path)
            
            # 提取第一页内容
            first_page = doc[0]
            
            # 获取标题（通常在第一页顶部，字体较大）
            title = self._extract_title(first_page)
            self.logger.info(f"提取标题: {title}")

            # 获取作者（通常在标题下方）
            authors = self._extract_authors(first_page)
            self.logger.info(f"提取作者: {authors}")

            # 获取摘要
            abstract = self._extract_abstract(doc)
            self.logger.info("摘要提取完成")

            # 提取全文（保留LaTeX格式）
            full_text = self._extract_full_text_with_latex(doc)
            self.logger.info(f"全文提取完成，共 {len(full_text.split())} 个词")
            
            # 构建论文信息字典
            paper_info = {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "full_text": full_text,
                "pdf_path": pdf_path
            }
            
            # 保存处理结果
            self._save_extraction_result(paper_info, pdf_path)
            
            doc.close()
            self.logger.info(f"PDF处理完成: {pdf_path}")
            return paper_info
            
        except Exception as e:
            self.logger.error(f"处理PDF文件 {pdf_path} 时出错: {str(e)}", exc_info=True)
            return None

    def _save_extraction_result(self, paper_info: Dict, pdf_path: str):
        """保存提取结果"""
        # 创建基于时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        result_file = os.path.join(self.results_dir, f"{base_name}_{timestamp}.json")
        
        # 准备保存的数据
        save_data = {
            "paper_info": {
                "title": paper_info["title"],
                "authors": paper_info["authors"],
                "abstract": paper_info["abstract"],
                "full_text_length": len(paper_info["full_text"].split())
            },
            "pdf_path": paper_info["pdf_path"]
        }
        
        # 保存为JSON文件
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"处理结果已保存到: {result_file}")

    def _extract_title(self, page: fitz.Page) -> str:
        """提取论文标题"""
        first_page = page
        blocks = first_page.get_text("dict")["blocks"]
        
        # 通过字体大小和位置识别标题
        title_candidates = []
        for block in blocks[:5]:  # 只检查前5个文本块
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # 标题通常字体较大且在页面上方
                        if span["size"] > 12 and span["origin"][1] < 200:
                            title_candidates.append((span["text"], span["size"]))
        
        if title_candidates:
            # 选择字体最大的文本作为标题
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            return title_candidates[0][0].strip()
        
        return "Unknown Title"

    def _extract_authors(self, page: fitz.Page) -> List[str]:
        """提取作者信息"""
        text = page.get_text()
        text_lines = text.split('\n')
        
        authors = []
        author_section_found = False
        
        for i, line in enumerate(text_lines[:10]):  # 检查前10行
            # 检查是否是作者行
            if any(marker in line for marker in self.author_markers['start']):
                author_section_found = True
                # 清理作者文本
                author_text = line
                for marker in self.author_markers['start']:
                    author_text = author_text.replace(marker, ',')
                
                # 分割并清理作者名
                for author in author_text.split(','):
                    author = author.strip()
                    if (author and 
                        len(author) > 2 and 
                        not any(marker in author.lower() for marker in self.author_markers['email']) and
                        not any(marker in author for marker in self.author_markers['affiliation'])):
                        authors.append(author)
                
                break
        
        # 如果没有找到明确的作者标记，尝试其他方法
        if not authors:
            abstract_index = -1
            for i, line in enumerate(text_lines):
                if any(marker in line for marker in self.section_markers['abstract']):
                    abstract_index = i
                    break
            
            if abstract_index > 0:
                # 检查摘要前的文本行
                potential_authors = text_lines[1:abstract_index]
                for line in potential_authors:
                    if (len(line.strip()) > 0 and 
                        ' ' in line and 
                        not any(marker in line for marker in 
                               [m for markers in self.section_markers.values() for m in markers])):
                        authors.append(line.strip())
        
        return [self._clean_text(author) for author in authors] if authors else ["Unknown"]

    def _extract_abstract(self, doc: fitz.Document) -> str:
        """提取摘要"""
        first_page = doc[0]
        text = first_page.get_text()
        
        # 尝试不同的分隔方法提取摘要
        abstract = ""
        for start_marker in self.section_markers['abstract']:
            if start_marker in text:
                parts = text.split(start_marker, 1)
                if len(parts) > 1:
                    abstract_text = parts[1]
                    # 分别处理每个可能的结束标记
                    end_markers = (self.section_markers['introduction'] + 
                                 self.section_markers['keywords'])
                    for end_marker in end_markers:
                        if end_marker in abstract_text:
                            abstract = abstract_text.split(end_marker)[0]
                            break
                    if abstract:  # 如果找到了摘要就跳出循环
                        break
        
        if not abstract:
            # 如果找不到明确的摘要，使用启发式方法
            blocks = first_page.get_text("dict")["blocks"]
            for block in blocks[2:6]:  # 检查前几个文本块
                if "lines" in block:
                    text = " ".join(span["text"] for line in block["lines"] 
                                  for span in line["spans"])
                    if len(text.split()) > 30:  # 假设摘要至少有30个词
                        abstract = text
                        break
        
        return self._clean_text(abstract)

    def _extract_full_text_with_latex(self, doc: fitz.Document) -> str:
        """提取全文内容，保留LaTeX格式
        Args:
            doc: PyMuPDF文档对象
        Returns:
            str: 提取的文本内容
        """
        text = []
        math_pattern = re.compile(r'(\$[^$]+\$|\$\$[^$]+\$\$)')
        
        for page in doc:
            # 获取页面的原始文本和字体信息
            blocks = page.get_text("dict")["blocks"]
            page_text = ""
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            # 检查是否是数学字体
                            if span.get("flags", 0) & 2**0:  # 数学字体标志
                                # 尝试恢复LaTeX格式
                                span_text = self._restore_latex_math(span["text"])
                            else:
                                span_text = span["text"]
                            
                            # 处理上下标
                            if span.get("size", 0) < line.get("size", 0):
                                if span["y"] < line["y"]:  # 上标
                                    span_text = f"^{{{span_text}}}"
                                else:  # 下标
                                    span_text = f"_{{{span_text}}}"
                            
                            line_text += span_text
                        
                        page_text += line_text + "\n"
            
            # 合并连续的数学公式
            page_text = self._merge_consecutive_math(page_text)
            text.append(page_text)
        
        return "\n".join(text)

    def _restore_latex_math(self, text: str) -> str:
        """尝试恢复LaTeX数学公式格式
        Args:
            text: 原始文本
        Returns:
            str: 恢复LaTeX格式后的文本
        """
        # 常见的数学符号映射
        math_symbols = {
            '∑': '\\sum',
            '∏': '\\prod',
            '∫': '\\int',
            '→': '\\rightarrow',
            '≤': '\\leq',
            '≥': '\\geq',
            '≠': '\\neq',
            '∈': '\\in',
            '⊆': '\\subseteq',
            '∪': '\\cup',
            '∩': '\\cap',
            '∞': '\\infty',
            'α': '\\alpha',
            'β': '\\beta',
            'γ': '\\gamma',
            'δ': '\\delta',
            'ε': '\\epsilon',
            'θ': '\\theta',
            'λ': '\\lambda',
            'μ': '\\mu',
            'π': '\\pi',
            'σ': '\\sigma',
            'τ': '\\tau',
            'φ': '\\phi',
            'ω': '\\omega'
        }
        
        # 替换数学符号
        for symbol, latex in math_symbols.items():
            text = text.replace(symbol, latex)
        
        # 如果看起来是独立的数学公式，添加行内公式标记
        if any(c in text for c in math_symbols.keys()) or \
           any(c in text for c in '+-*/=<>()[]{}'):
            text = f"${text}$"
        
        return text

    def _merge_consecutive_math(self, text: str) -> str:
        """合并连续的数学公式
        Args:
            text: 原始文本
        Returns:
            str: 合并后的文本
        """
        # 合并连续的 $...$ 公式
        text = re.sub(r'\$\s*\$', '', text)  # 移除空的数学公式
        text = re.sub(r'\$\s*\$([^$])', r'\1', text)  # 移除不必要的分隔
        text = re.sub(r'\$([^$]+?)\$\s*\$([^$]+?)\$', r'$\1\2$', text)  # 合并相邻公式
        
        return text

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        return text.strip()

    def extract_text_with_tokens(self, pdf_path: str) -> Tuple[str, Dict]:
        """从PDF提取文本并计算tokens
        Args:
            pdf_path: PDF文件路径
        Returns:
            Tuple[str, Dict]: (提取的文本, token统计信息)
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            token_stats = {
                "total_tokens": 0,
                "by_page": [],
                "exceeds_limit": False,
                "recommended_chunks": 1
            }
            
            # 逐页处理
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text", sort=True)
                
                # 计算当前页的tokens
                page_tokens = len(self.tokenizer.encode(page_text))
                token_stats["by_page"].append({
                    "page": page_num + 1,
                    "tokens": page_tokens
                })
                
                text += page_text + "\n"
                token_stats["total_tokens"] += page_tokens
            
            # 检查是否超过各模型的限制
            for model, limit in self.TOKEN_LIMITS.items():
                if token_stats["total_tokens"] > limit:
                    token_stats["exceeds_limit"] = True
                    # 计算建议的分块数量
                    token_stats["recommended_chunks"] = max(
                        token_stats["recommended_chunks"],
                        (token_stats["total_tokens"] // limit) + 1
                    )
            
            # 添加警告信息
            if token_stats["exceeds_limit"]:
                print(f"\n⚠️ 警告：文档总tokens ({token_stats['total_tokens']}) 超过模型限制")
                print(f"建议将文档分成 {token_stats['recommended_chunks']} 个块进行处理")
                
                # 显示每页的token分布
                print("\n每页tokens分布:")
                for page_stat in token_stats["by_page"]:
                    print(f"第 {page_stat['page']} 页: {page_stat['tokens']} tokens")
            
            return text, token_stats
            
        except Exception as e:
            print(f"PDF处理出错: {str(e)}")
            return "", {"total_tokens": 0, "error": str(e)}

    def suggest_chunk_size(self, total_tokens: int, target_model: str = "o1-preview") -> int:
        """根据总tokens建议合适的分块大小
        Args:
            total_tokens: 总token数
            target_model: 目标模型名称
        Returns:
            int: 建议的每块token数
        """
        model_limit = self.TOKEN_LIMITS.get(target_model, 4096)
        # 预留20%空间给其他内容（如提示词）
        safe_limit = int(model_limit * 0.8)
        
        if total_tokens <= safe_limit:
            return total_tokens
        
        # 计算需要的块数
        num_chunks = (total_tokens // safe_limit) + 1
        # 平均分配，确保每块大小相近
        return total_tokens // num_chunks 