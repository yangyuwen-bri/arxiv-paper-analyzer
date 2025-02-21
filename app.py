import streamlit as st
# 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="AI新知库",
    page_icon="🔬",
    layout="wide"
)

import sys
import os
import webbrowser
from typing import Dict
from langchain_community.document_loaders import PyMuPDFLoader
import arxiv
from datetime import datetime

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from arxiv_analyzer import ArxivPaperAnalyzer
from document_processor import (
    AnalysisType, 
    DocumentProcessor,
)

def init_session_state():
    """初始化会话状态"""
    if 'analyzer' not in st.session_state:
        # 使用默认配置初始化
        st.session_state.analyzer = ArxivPaperAnalyzer(
            model_type="openai",
            pdf_dir="custom_download_folder"  # 正确参数位置
        )
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'analyses' not in st.session_state:
        st.session_state.analyses = []
    if 'output_files' not in st.session_state:
        st.session_state.output_files = {}
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if 'micro_innovations' not in st.session_state:
        st.session_state.micro_innovations = {}  # 结构：{arxiv_id: analysis_text}

def sidebar_model_selection():
    """侧边栏模型选择"""
    st.sidebar.header("🤖 模型选择")
    model_info = st.sidebar.radio(
        label="选择模型类型",  
        options=["OpenAI", "DeepSeek"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    return {
        "model_type": "openai" if model_info == "OpenAI" else "deepseek"
    }

def load_arxiv_taxonomy():
    """加载ArXiv领域分类"""
    import pandas as pd
    
    try:
        df = pd.read_excel("arxiv_taxonomy_cn.xlsx")
        # 构建层级字典
        taxonomy = {}
        for _, row in df.iterrows():
            if row['Group'] not in taxonomy:
                taxonomy[row['Group']] = {}
            if pd.notna(row['Subgroup']):  # 确保Subgroup不是NaN
                if row['Subgroup'] not in taxonomy[row['Group']]:
                    taxonomy[row['Group']][row['Subgroup']] = []
                taxonomy[row['Group']][row['Subgroup']].append({
                    'code': row['Code'],
                    'description': row['Description'] if pd.notna(row['Description']) else '',
                    'keywords': row['Keywords'] if pd.notna(row['Keywords']) else ''
                })
        return taxonomy
    except Exception as e:
        st.error(f"加载领域分类失败: {str(e)}")
        return {}

def sidebar_search_options():
    """侧边栏搜索选项"""
    st.sidebar.header("🔍 搜索选项")
    
    # 领域选择
    field_method = st.sidebar.radio(
        "研究领域",
        ["自动推荐", "手动选择"],
        index=0,
        horizontal=True
    )
    
    field = None
    if field_method == "手动选择":
        # 加载领域分类
        taxonomy = load_arxiv_taxonomy()
        
        # 主领域选择
        main_field = st.sidebar.selectbox(
            "主要领域",
            options=list(taxonomy.keys()),
            help="选择论文的主要研究领域"
        )
        
        # 如果选择了主领域，显示对应的子领域选择
        if main_field and main_field in taxonomy:
            sub_fields = list(taxonomy[main_field].keys())
            sub_field = st.sidebar.selectbox(
                "子领域",
                options=sub_fields
            )
            if sub_field:
                field = [item['code'] for item in taxonomy[main_field][sub_field]]
    
    # 其他搜索选项（删除分隔线）
    max_papers = st.sidebar.slider(
        "论文数量",
        min_value=1,
        max_value=100,
        value=5
    )
    analysis_type = st.sidebar.radio(
        "分析深度", 
        ["摘要分析", "全文分析"],
        index=0,
        horizontal=True
    )
    
    return {
        "field": field,
        "max_papers": max_papers,
        "analysis_type": analysis_type == "全文分析"
    }

def search_papers(model_type: str, search_options: Dict):
    """搜索论文"""
    st.session_state.analyzer = ArxivPaperAnalyzer(
        model_type=model_type,
        pdf_dir="custom_download_folder"
    )
    
    with st.spinner("正在搜索和分析论文..."):
        query = st.text_input(
            label="论文搜索",
            placeholder="输入研究主题，如：机器学习、计算机视觉、自然语言处理",
            label_visibility="collapsed"
        )
        
        if st.button("开始搜索"):
            try:
                results = st.session_state.analyzer.search_and_analyze_papers(
                    query=query,
                    max_papers=search_options['max_papers'],
                    analyze_full_text=search_options['analysis_type'],
                    field=search_options['field']
                )
                
                st.session_state.papers = results.get('papers', [])
                # 将微创新结果存入独立存储
                for paper in st.session_state.papers:
                    if 'micro_innovation' in paper:
                        paper_id = paper.get('arxiv_id')
                        if paper_id:
                            st.session_state.micro_innovations[paper_id] = paper['micro_innovation']
                
                st.session_state.analyses = results.get('analyses', [])
                st.session_state.output_files = results.get('outputs', {})
                st.session_state.doc_processor = results.get('doc_processor', DocumentProcessor())
                
                st.success(f"成功获取 {len(st.session_state.papers)} 篇论文")
            except Exception as e:
                st.error(f"搜索失败: {str(e)}")

def display_papers():
    st.markdown("""
    <style>
    .innovation-concept {
        font-size: 1.3em;
        color: #2c3e50;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem;
    }
    .innovation-section {
        margin-left: 1.5rem;
        padding: 0.8rem;
        background: #f8f9fa;
        border-radius: 6px;
        line-height: 1.8;
        text-indent: 2em;
    }
    .innovation-section strong {
        color: #27ae60;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.papers:
        # 在展示论文前添加路径检查
        download_dir = st.session_state.analyzer.pdf_dir
        if not os.path.exists(download_dir):
            st.warning(f"下载目录 {download_dir} 不存在，已自动创建")
            os.makedirs(download_dir)
        
        # 先展示分析结果
        st.header("🔬 分析结果")
        
        # 判断分析类型
        is_full_analysis = len(st.session_state.analyses) == len(st.session_state.papers)
        
        if is_full_analysis:
            # 全文分析：展示完整分析
            with st.expander("📄 论文核心要点", expanded=False):
                for i, analysis in enumerate(st.session_state.analyses, 1):
                    cleaned = ArxivPaperAnalyzer._clean_thinking_chain(analysis)
                    st.markdown(f"### 论文 {i} 分析")
                    st.markdown(cleaned)
            
            # 检查 PDF 生成结果
            if st.session_state.output_files:
                pdf_path = st.session_state.output_files.get('pdf')
                md_path = st.session_state.output_files.get('markdown')
                
                # PDF 下载按钮
                if pdf_path and os.path.exists(pdf_path):
                    st.download_button(
                        label="下载 PDF 分析报告",
                        data=open(pdf_path, 'rb').read(),
                        file_name=os.path.basename(pdf_path),
                        mime='application/pdf'
                    )
                else:
                    st.warning("PDF 生成失败，提供 Markdown 下载")
                
                # Markdown 下载按钮（作为备选）
                if md_path and os.path.exists(md_path):
                    st.download_button(
                        label="下载 Markdown 分析报告",
                        data=open(md_path, 'rb').read(),
                        file_name=os.path.basename(md_path),
                        mime='text/markdown'
                    )
        else:
            # 摘要分析：展示亮点速览
            with st.expander("📊 亮点速览", expanded=False):
                cleaned = ArxivPaperAnalyzer._clean_thinking_chain(st.session_state.analyses[0])
                st.markdown(cleaned)
            
            # 检查 PDF 生成结果
            if st.session_state.output_files:
                pdf_path = st.session_state.output_files.get('pdf')
                md_path = st.session_state.output_files.get('markdown')
                
                # PDF 下载按钮
                if pdf_path and os.path.exists(pdf_path):
                    st.download_button(
                        label="下载 PDF 分析报告",
                        data=open(pdf_path, 'rb').read(),
                        file_name=os.path.basename(pdf_path),
                        mime='application/pdf'
                    )
                else:
                    st.warning("PDF 生成失败，提供 Markdown 下载")
                
                # Markdown 下载按钮（作为备选）
                if md_path and os.path.exists(md_path):
                    st.download_button(
                        label="下载 Markdown 分析报告",
                        data=open(md_path, 'rb').read(),
                        file_name=os.path.basename(md_path),
                        mime='text/markdown'
                    )
        
        # 再展示论文列表
        st.header("📄 论文列表")
        for i, paper in enumerate(st.session_state.papers, 1):
            print(f"论文 {i} 的键: {paper.keys()}")  # 调试日志
            current_paper = paper.copy()
            
            with st.expander(f"{i}. {current_paper['title']}", expanded=False):
                st.markdown(f"**作者**: {', '.join(current_paper['authors'])}")
                st.markdown(f"**发布时间**: {current_paper['published']}")
                st.markdown(f"**摘要**: {current_paper['abstract']}")
                
                # 新增微创新按钮
                col1, col2, col3 = st.columns([1,1,3])
                with col1:
                    st.link_button("📥 下载 PDF", current_paper['pdf_url'])
                with col2:
                    st.link_button("🔗 arXiv 链接", current_paper['arxiv_url'])
                with col3:
                    # 兼容旧数据和新数据
                    paper_id = current_paper.get('arxiv_id') or current_paper['arxiv_url'].split('/')[-1]
                    if st.button("✨ 深度分析", key=f"detail_{i}"):
                        st.session_state.current_paper_id = paper_id
                        st.query_params["paper_id"] = paper_id
                        st.rerun()

def show_innovation_analysis():
    # 独立头部导航
    col1, col2 = st.columns([2, 8])
    with col1:
        if st.button("← 返回论文列表", use_container_width=True):
            del st.query_params["paper_id"]
            st.rerun()
    with col2:
        st.title("✨ 微创新中心")
    
    # 通过ID获取论文（支持历史记录）
    paper_id = st.query_params.get("paper_id")
    if not paper_id:
        st.error("未指定论文ID")
        return
    
    # 优先从微创新存储获取
    if paper_id in st.session_state.micro_innovations:
        analysis = st.session_state.micro_innovations[paper_id]
        display_analysis(paper_id, analysis)
        return
    
    # 兼容旧数据获取方式
    paper = next((p for p in st.session_state.papers if p.get('arxiv_id') == paper_id), None)
    if paper and 'micro_innovation' in paper:
        st.session_state.micro_innovations[paper_id] = paper['micro_innovation']
        display_analysis(paper_id, paper['micro_innovation'])
    else:
        st.warning("该论文尚未生成深度分析")

    print(f"当前查询的paper_id: {paper_id}")
    print(f"会话中的论文ID列表: {[p.get('arxiv_id') for p in st.session_state.papers]}")
    print(f"微创新存储的键: {st.session_state.micro_innovations.keys()}")

def display_analysis(paper_id: str, analysis: str):
    """独立展示分析内容"""
    # 获取基础信息（可扩展为数据库查询）
    paper = get_paper_metadata(paper_id)  # 需要实现元数据获取方法
    
    with st.container():
        # 信息展示区
        col_info, col_actions = st.columns([3, 1])
        with col_info:
            st.subheader(paper['title'])
            st.caption(f"作者：{', '.join(paper['authors'])} | 发布时间：{paper['published']}")
        
        with col_actions:
            if st.button("🔄 重新生成分析"):
                regenerate_analysis(paper_id)
            
            if st.download_button("📥 导出分析报告", 
                                data=generate_report(analysis),
                                file_name=f"{paper_id}_analysis.md"):
                st.toast("导出成功！")
        
        # 分析内容展示
        with st.expander("📄 原始论文摘要", expanded=True):
            st.write(paper['abstract'])
        
        formatted = DocumentProcessor.format_micro_innovation(analysis)
        st.markdown(f"""
            <div style="
                background: #f8f9fa;
                padding: 1.5rem;
                border-radius: 8px;
                margin-top: 1rem;
            ">
                {formatted}
            </div>
        """, unsafe_allow_html=True)

def get_paper_metadata(paper_id: str) -> Dict:
    """获取论文元数据（从会话状态或API）"""
    # 优先从会话状态获取
    paper = next((p for p in st.session_state.papers if p.get('arxiv_id') == paper_id), None)
    if paper:
        return paper
    
    # 如果会话状态不存在，尝试通过ArXiv API获取
    try:
        search = arxiv.Search(id_list=[paper_id])
        result = next(st.session_state.analyzer.client.results(search))
        return st.session_state.analyzer._convert_result_to_paper(result)
    except Exception as e:
        st.error(f"无法获取论文元数据: {str(e)}")
        return {
            "title": "未知标题",
            "authors": ["未知作者"],
            "published": "未知日期",
            "abstract": "无法获取摘要"
        }

def regenerate_analysis(paper_id: str):
    """重新生成分析内容"""
    with st.spinner("正在重新生成分析..."):
        try:
            # 获取论文数据
            paper = get_paper_metadata(paper_id)
            
            # 下载并处理PDF
            pdf_path = st.session_state.analyzer.download_pdf(
                paper['pdf_url'], 
                paper['arxiv_url']
            )
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            full_text = "\n".join(page.page_content for page in pages)
            
            # 生成新分析
            innovation = st.session_state.analyzer.micro_innovation_chain.invoke({
                "title": paper["title"],
                "abstract": paper["abstract"],
                "full_text": full_text
            })
            
            # 更新存储
            st.session_state.micro_innovations[paper_id] = innovation['text']
            st.rerun()
            
        except Exception as e:
            st.error(f"重新生成失败: {str(e)}")

def generate_report(analysis: str) -> str:
    """生成可下载的分析报告"""
    # 添加报告元数据
    report = f"# 论文创新分析报告\n\n"
    report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    report += "---\n\n"
    
    # 格式化内容
    formatted = DocumentProcessor.format_micro_innovation(analysis)
    report += formatted
    
    # 转换为字节流
    return report.encode('utf-8')

def main():
    init_session_state()
    
    # 新增页面路由
    if 'paper_id' in st.query_params:
        show_innovation_analysis()
        return
    
    # 原有主页面逻辑
    st.title("🌟 AI新知库")
    
    # 模型选择（简化后的版本）
    model_config = sidebar_model_selection()
    
    # 搜索选项
    search_options = sidebar_search_options()
    
    # 论文搜索区域（移除 provider_type）
    search_papers(model_config["model_type"], search_options)
    
    # 论文展示
    display_papers()

if __name__ == "__main__":
    main() 