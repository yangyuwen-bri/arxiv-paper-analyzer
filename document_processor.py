from enum import Enum
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import os
from pylatex import Document, NoEscape
from pdf_processor import PDFProcessor
import subprocess
import shutil
import markdown
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import platform
from utils.document_converter import DocumentConverter

class AnalysisType(Enum):
    SUMMARY = "summary"  # 摘要分析
    FULL_TEXT = "full_text"  # 全文分析

class FontManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FontManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not FontManager._initialized:
            self.system = platform.system()
            self.default_font = None
            self.fallback_font = None
            self._register_system_fonts()
            FontManager._initialized = True

    def _register_system_fonts(self) -> None:
        """注册系统默认字体"""
        # 检查是否已经注册了字体
        if self.default_font:
            return
        
        font_paths = self._get_system_font_paths()
        
        # 尝试注册字体
        for font_name, font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    # 检查字体是否已经注册
                    if font_name not in pdfmetrics.getRegisteredFontNames():
                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                        if not self.default_font:
                            self.default_font = font_name
                        print(f"成功注册字体: {font_name}")
                except Exception as e:
                    print(f"注册字体失败 {font_name}: {str(e)}")
        
        # 如果没有找到任何系统字体，使用 reportlab 默认字体
        if not self.default_font:
            print("警告：未找到系统字体，将使用 reportlab 默认字体")
            self.default_font = 'Helvetica'
            self.fallback_font = 'Times-Roman'

    def _get_system_font_paths(self) -> List[Tuple[str, str]]:
        """获取系统字体路径"""
        if self.system == "Darwin":  # macOS
            return [
                ("STHeiti", "/System/Library/Fonts/STHeiti Light.ttc"),
                ("PingFang", "/System/Library/Fonts/PingFang.ttc"),
                ("Hiragino", "/System/Library/Fonts/HiraginoSans.ttc")
            ]
        elif self.system == "Windows":
            return [
                ("MicrosoftYaHei", "C:\\Windows\\Fonts\\msyh.ttc"),
                ("SimSun", "C:\\Windows\\Fonts\\simsun.ttc"),
                ("SimHei", "C:\\Windows\\Fonts\\simhei.ttc")
            ]
        else:  # Linux
            return [
                ("WenQuanYi", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
                ("Noto", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                ("Droid", "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf")
            ]

    def get_default_font(self) -> str:
        """获取默认字体"""
        return self.default_font

    def get_fallback_font(self) -> str:
        """获取备用字体"""
        return self.fallback_font or self.default_font

class DocumentProcessor:
    """统一的文档处理类"""
    def __init__(self, output_dir: str = "paper_analyses"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.pdf_processor = PDFProcessor()
        self.document_converter = DocumentConverter()
        self.font_manager = FontManager()
        
        # 设置LaTeX包
        self.latex_packages = [
            r'\usepackage{amsmath}',
            r'\usepackage{amssymb}',
            r'\usepackage{amsfonts}',
            r'\usepackage{mathtools}',
            r'\usepackage{graphicx}',
            r'\usepackage{hyperref}',
            r'\usepackage{xcolor}'
        ]

        # 添加数学符号映射
        self.math_mappings = {
            # 基础运算符
            '±': r'\pm',
            '×': r'\times',
            '÷': r'\div',
            
            # 希腊字母
            'α': r'\alpha',
            'β': r'\beta',
            'γ': r'\gamma',
            'δ': r'\delta',
            'ε': r'\epsilon',
            'θ': r'\theta',
            'λ': r'\lambda',
            'μ': r'\mu',
            'π': r'\pi',
            'σ': r'\sigma',
            'τ': r'\tau',
            'φ': r'\phi',
            'ω': r'\omega',
            
            # 数学符号
            '∑': r'\sum',
            '∏': r'\prod',
            '∫': r'\int',
            '∞': r'\infty',
            
            # 关系运算符
            '≤': r'\leq',
            '≥': r'\geq',
            '≠': r'\neq',
            '→': r'\rightarrow',
            
            # 集合符号
            '∈': r'\in',
            '∉': r'\notin',
            '⊆': r'\subseteq',
            '⊂': r'\subset',
            '∪': r'\cup',
            '∩': r'\cap'
        }

        # 添加特殊数学字体映射
        self.math_fonts = {
            'E': r'\mathbb{E}',  # 期望符号
            'R': r'\mathbb{R}',  # 实数集
            'N': r'\mathbb{N}',  # 自然数集
            'Z': r'\mathbb{Z}',  # 整数集
            'Q': r'\mathbb{Q}',  # 有理数集
            'C': r'\mathbb{C}',  # 复数集
            'P': r'\mathbb{P}'   # 概率符号
        }
        
        # 更新 LaTeX 包，添加特殊字体支持
        self.latex_packages.extend([
            r'\usepackage{amssymb}',  # 提供 \mathbb 命令
            r'\usepackage{amsthm}',   # 数学定理环境
            r'\usepackage{mathrsfs}'  # 提供花体字符
        ])

    def process_papers(self, 
                      papers: List[Dict],
                      analysis_type: AnalysisType,
                      analysis_result: Union[str, List[str]]) -> Dict[str, str]:
        """统一的论文处理接口"""
        # 生成基础文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base_filename = f"paper_analysis_{timestamp}"
        
        # 确保 analysis_result 是列表
        if isinstance(analysis_result, str):
            analysis_result = [analysis_result]
        
        # 根据分析类型选择不同的格式化方法
        if analysis_type == AnalysisType.SUMMARY:
            content = self._format_summary_analysis(papers, analysis_result[0])
        else:
            content = self._format_full_analysis(papers, analysis_result)
        
        # 保存不同格式
        outputs = {}
        outputs['markdown'] = self._save_markdown(content, base_filename)
        
        # 尝试多种 PDF 生成方式
        pdf_path = self._generate_pdf_fallback(content, base_filename)
        if pdf_path:
            outputs['pdf'] = pdf_path
        
        return outputs

    def _format_summary_analysis(self, papers: List[Dict], analyses: Union[str, List[str]]) -> str:
        """格式化摘要分析结果"""
        content = "# 论文分析报告\n\n"
        
        content += "## 分析结果\n\n"
        
        # 处理 analyses 可能是字符串或列表的情况
        if isinstance(analyses, str):
            content += analyses
        elif isinstance(analyses, list):
            # 如果是列表，连接所有分析结果
            content += "\n\n".join(analyses)
        else:
            content += "无法生成分析结果"
        
        return content

    def _format_full_analysis(self, papers: List[Dict], analyses: List[str]) -> str:
        """格式化全文分析结果"""
        content = "# 论文深度分析报告\n\n"
        
        # 确保 analyses 长度与 papers 匹配
        if len(analyses) != len(papers):
            print(f"警告：分析结果数量({len(analyses)})与论文数量({len(papers)})不匹配")
            # 如果分析结果不足，用空字符串填充
            analyses = analyses + [''] * (len(papers) - len(analyses))
        
        content += "## 分析内容\n\n"
        
        for i, (paper, analysis) in enumerate(zip(papers, analyses), 1):
            content += f"### 论文 {i}: {paper['title']}\n\n"
            
            # 处理可能是字符串或列表的分析结果
            if isinstance(analysis, list):
                content += "\n\n".join(analysis) + "\n\n"
            else:
                content += str(analysis) + "\n\n"
        
        return content

    def _save_markdown(self, content: str, filename: str) -> str:
        """保存为Markdown文件"""
        file_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    def _check_latex_compiler(self):
        """检查可用的 LaTeX 编译器，优先选择支持中文的编译器"""
        # 优先级：XeLaTeX > LuaLaTeX > pdfLaTeX
        compilers = ['xelatex', 'lualatex', 'pdflatex']
        for compiler in compilers:
            if shutil.which(compiler):
                print(f"找到 LaTeX 编译器: {compiler}")
                return compiler
        
        print("警告：未找到任何 LaTeX 编译器")
        print("建议检查：")
        print("1. 确认 MiKTeX 已正确安装")
        print("2. 在 MiKTeX 控制台中更新包")
        print("3. 将 LaTeX 安装目录添加到系统 PATH")
        return None

    def _create_pdf(self, content: str, filename: str, title: str = "论文分析报告") -> Optional[str]:
        """创建PDF文档"""
        try:
            # 检查 LaTeX 编译器
            latex_compiler = self._check_latex_compiler()
            if not latex_compiler:
                print("未找到 LaTeX 编译器，将使用 Markdown 替代")
                return self._save_markdown(content, filename).replace('.md', '.pdf')

            # 选择支持中文的编译器和文档类
            if latex_compiler in ['xelatex', 'lualatex']:
                documentclass = 'ctexart'
                packages = [
                    r'\usepackage{fontspec}',
                    f'\\setmainfont{{{self.font_manager.get_default_font()}}}',
                    f'\\setCJKmainfont{{{self.font_manager.get_default_font()}}}',
                    r'\usepackage{xeCJK}',
                    r'\usepackage{ctex}'
                ]
            else:
                documentclass = 'article'
                packages = [
                    r'\usepackage[UTF8]{ctex}',
                    r'\usepackage{CJK}',
                    r'\usepackage{CJKpunct}',
                    r'\usepackage{CJKspace}'
                ]

            # 创建文档
            doc = Document(documentclass=documentclass)
            
            # 添加中文支持包
            for package in packages:
                doc.preamble.append(NoEscape(package))
            
            # 添加原有的 LaTeX 包
            for package in self.latex_packages:
                doc.preamble.append(NoEscape(package))
            
            # 设置文档属性
            doc.preamble.append(NoEscape(r'\title{' + title + '}'))
            doc.preamble.append(NoEscape(r'\author{ArXiv论文分析工具}'))
            doc.preamble.append(NoEscape(r'\date{\today}'))
            
            # 对于 pdflatex，需要在 CJK 环境中添加内容
            if latex_compiler == 'pdflatex':
                doc.append(NoEscape(r'\begin{CJK}{UTF8}{song}'))
            
            # 生成标题页
            doc.append(NoEscape(r'\maketitle'))
            
            # 添加内容
            doc.append(NoEscape(self._convert_markdown_to_latex(content)))
            
            # 对于 pdflatex，需要关闭 CJK 环境
            if latex_compiler == 'pdflatex':
                doc.append(NoEscape(r'\end{CJK}'))
            
            # 生成PDF
            output_path = os.path.join(self.output_dir, filename)
            doc.generate_pdf(output_path, clean_tex=False, compiler=latex_compiler)
            
            return f"{output_path}.pdf"
        
        except FileNotFoundError as e:
            print(f"缺少必要的 LaTeX 宏包: {str(e)}")
            print("MiKTeX 用户建议：")
            print("1. 打开 MiKTeX 控制台")
            print("2. 选择 'Package Manager'")
            print("3. 搜索并安装以下包：")
            print("   - ctex")
            print("   - CJK")
            print("   - CJKpunct")
            print("   - xeCJK")
            return self._save_markdown(content, filename).replace('.md', '.pdf')
        
        except Exception as e:
            print(f"PDF生成失败: {str(e)}")
            print("详细错误信息：")
            import traceback
            traceback.print_exc()
            
            # 添加更详细的错误处理和建议
            print("\n解决建议：")
            print("1. 确保已在 MiKTeX 控制台安装以下包：")
            print("   - xeCJK")
            print("   - ctex")
            print("   - CJK")
            print("   - CJKpunct")
            print("   - fontspec")
            print("2. 在 MiKTeX 控制台点击 'Check for updates'")
            print("3. 重启 MiKTeX 控制台")
            
            return self._save_markdown(content, filename).replace('.md', '.pdf')

    def _convert_markdown_to_latex(self, content: str) -> str:
        """改进的Markdown到LaTeX转换"""
        # 处理上下标
        def process_subscript(match):
            base = match.group(1)
            sub = match.group(2)
            # 检查是否需要特殊字体处理
            if base in self.math_fonts:
                base = self.math_fonts[base]
            return f"{base}_{{\\text{{{sub}}}}}"
            
        def process_superscript(match):
            base = match.group(1)
            sup = match.group(2)
            return f"{base}^{{\\text{{{sup}}}}}"
        
        def process_math_fonts(formula: str) -> str:
            """处理特殊数学字体"""
            # 处理 \mathbb{X} 形式
            for char, latex in self.math_fonts.items():
                formula = formula.replace(f'\\mathbb{{{char}}}', latex)
                
            return formula
        
        # 处理行内数学公式
        def process_inline_math(match):
            formula = match.group(1)
            # 先处理特殊字体
            formula = process_math_fonts(formula)
            # 再处理其他数学符号
            for symbol, latex in self.math_mappings.items():
                formula = formula.replace(symbol, latex)
            return f"$\\displaystyle{{{formula}}}$"
        
        # 处理行间数学公式
        def process_display_math(match):
            formula = match.group(1)
            # 先处理特殊字体
            formula = process_math_fonts(formula)
            # 再处理其他数学符号
            for symbol, latex in self.math_mappings.items():
                formula = formula.replace(symbol, latex)
            return f"\\[\n{formula}\n\\]"
        
        import re
        
        # 处理上下标（在处理其他数学公式之前）
        content = re.sub(r'([A-Za-z])_([A-Za-z0-9]+)', process_subscript, content)
        content = re.sub(r'([A-Za-z])\^([A-Za-z0-9]+)', process_superscript, content)
        
        # 处理数学公式
        content = re.sub(r'\$([^$]+)\$', process_inline_math, content)
        content = re.sub(r'\$\$([^$]+)\$\$', process_display_math, content)
        
        # 处理标题和列表（在处理数学公式之后）
        content = content.replace('# ', r'\section{')
        content = content.replace('## ', r'\subsection{')
        content = content.replace('### ', r'\subsubsection{')
        content = content.replace('\n', '}\n')
        content = content.replace('- ', r'\item ')
        
        return content 

    def _generate_pdf_with_markdown(self, content: str, filename: str) -> str:
        """使用 md-to-pdf 转换 Markdown 到 PDF"""
        md_path = os.path.join(self.output_dir, f"{filename}.md")
        pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
        
        # 写入 Markdown 文件
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 调用 DocumentConverter 的 md_to_pdf 方法
        return self.document_converter.md_to_pdf(md_path)

    def _generate_pdf_fallback(self, content: str, filename: str) -> Optional[str]:
        """尝试 PDF 生成方法"""
        try:
            pdf_path = self._generate_pdf_with_markdown(content, filename)
            if pdf_path and os.path.exists(pdf_path):
                return pdf_path
        except Exception as e:
            print(f"PDF 生成失败: {str(e)}")
        
        return None 