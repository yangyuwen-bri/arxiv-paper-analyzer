import streamlit as st
# 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="ArXiv论文分析工具",
    page_icon="🔬",
    layout="wide"
)

import sys
import os
import webbrowser
from typing import Dict

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from arxiv_analyzer import ArxivPaperAnalyzer
from document_processor import AnalysisType

def init_session_state():
    """初始化会话状态"""
    if 'analyzer' not in st.session_state:
        # 使用默认配置初始化
        st.session_state.analyzer = ArxivPaperAnalyzer()
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'analyses' not in st.session_state:
        st.session_state.analyses = []
    if 'output_files' not in st.session_state:
        st.session_state.output_files = {}

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
        model_type=model_type
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
                st.session_state.analyses = results.get('analyses', [])
                st.session_state.output_files = results.get('outputs', {})
                
                st.success(f"成功获取 {len(st.session_state.papers)} 篇论文")
            except Exception as e:
                st.error(f"搜索失败: {str(e)}")

def display_papers():
    """展示论文和分析结果"""
    if st.session_state.papers:
        # 先展示分析结果
        st.header("🔬 分析结果")
        
        # 判断分析类型
        is_full_analysis = len(st.session_state.analyses) == len(st.session_state.papers)
        
        if is_full_analysis:
            # 全文分析：展示每篇论文的核心要点
            st.subheader("📋 论文核心要点")
            for i, (paper, analysis) in enumerate(zip(st.session_state.papers, st.session_state.analyses), 1):
                with st.expander(f"论文 {i}: {paper['title']} - 核心要点", expanded=True):
                    core_points = extract_core_points(analysis)
                    st.markdown(core_points)
            
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
            with st.expander("📊 亮点速览", expanded=True):
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
            with st.expander(f"{i}. {paper['title']}", expanded=False):
                st.markdown(f"**作者**: {', '.join(paper['authors'])}")
                st.markdown(f"**发布时间**: {paper['published']}")
                st.markdown(f"**摘要**: {paper['abstract']}")
                
                # 添加下载和链接按钮
                col1, col2 = st.columns(2)
                with col1:
                    st.link_button("📥 下载 PDF", paper['pdf_url'])
                with col2:
                    st.link_button("🔗 arXiv 链接", paper['arxiv_url'])

def extract_core_points(analysis: str) -> str:
    """从完整分析中提取核心要点"""
    try:
        # 定义可能的核心要点标题
        sections = [
            "## 🎯 核心要点速览", 
            "## 核心要点速览", 
            "## 🎯 核心要点", 
            "## 核心要点"
        ]
        
        # 尝试找到核心要点部分
        for section in sections:
            if section in analysis:
                # 找到该部分后，提取到下一个二级标题之前的内容
                start_index = analysis.index(section) + len(section)
                next_section_match = analysis.find("## ", start_index)
                
                if next_section_match != -1:
                    core_points = analysis[start_index:next_section_match].strip()
                else:
                    core_points = analysis[start_index:].strip()
                
                # 如果提取的内容太短，返回整个分析
                if len(core_points) > 50:
                    return core_points
        
        # 如果没有找到特定部分，尝试提取前500个字符
        return analysis[:500] + "..."
    
    except Exception as e:
        print(f"提取核心要点出错: {str(e)}")
        return "无法提取核心要点"

def main():
    init_session_state()
    
    st.title("🌟 ArXiv 论文智能分析平台")
    
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