from thefuzz import fuzz
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from .taxonomy import ArxivTaxonomy
import time
import pandas as pd
import streamlit as st

class ArxivCategoryMatcher:
    """ArXiv分类匹配器"""
    
    def __init__(self, 
                 openai_api_key: str, 
                 base_url: str = None, 
                 model_type: str = "openai"):
        """初始化匹配器
        Args:
            openai_api_key: API密钥
            base_url: 自定义API地址，例如："https://your-api-proxy.com/v1"
            model_type: 模型类型，默认为 "openai"
        """
        self.taxonomy = ArxivTaxonomy()
        
        # 根据 model_type 选择不同的配置
        if model_type == "deepseek":
            from config import DEEPSEEK_CONFIG
            api_key = DEEPSEEK_CONFIG.get("api_key")
            base_url = DEEPSEEK_CONFIG.get("base_url")
        else:
            from config import OPENAI_CONFIG
            api_key = openai_api_key or OPENAI_CONFIG["api_key"]
            base_url = base_url or OPENAI_CONFIG["base_url"]
        
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model_name="o1-preview",
            base_url=base_url,
            request_timeout=60,
            max_retries=3
        )
        
        # 加载并解析分类层级
        self.taxonomy_df = pd.read_excel('arxiv_taxonomy.xlsx')
        self.taxonomy_hierarchy = self._build_taxonomy_hierarchy()
    
    def _build_taxonomy_hierarchy(self):
        """构建分类层级结构"""
        hierarchy = {}
        for _, row in self.taxonomy_df.iterrows():
            group = row['Group']
            subgroup = row['Subgroup']
            code = row['Code']
            
            if group not in hierarchy:
                hierarchy[group] = {}
            
            if subgroup not in hierarchy[group]:
                hierarchy[group][subgroup] = []
            
            hierarchy[group][subgroup].append({
                'code': code,
                'description': row['Description']
            })
        
        return hierarchy
    
    def suggest_categories(self, description: str) -> List[str]:
        """使用LLM推荐合适的分类"""
        try:
            print(f"\n=== AI推荐开始 ===")
            print(f"输入描述: {description}")
            
            prompt = f"""
            你是一个专业的论文分类助手。请根据以下研究领域描述，推荐最相关的 arXiv 分类代码。

            研究领域描述: "{description}"

            常见对应关系：
            - 人工智能、智能系统 -> cs.AI
            - 机器学习、深度学习 -> cs.LG
            - 计算机视觉、图像处理 -> cs.CV
            - 自然语言处理、文本分析 -> cs.CL
            - 机器人、自动化 -> cs.RO
            - 神经网络 -> cs.NE

            请只返回分类代码，用逗号分隔。例如：cs.AI, cs.LG
            """
            
            print(f"发送请求到API...")
            start_time = time.time()
            response = self.llm.invoke(prompt, timeout=30)
            elapsed = time.time() - start_time
            print(f"API响应时间: {elapsed:.2f}秒")
            
            print(f"API原始响应: {response}")
            response_text = response.content if hasattr(response, 'content') else str(response)
            print(f"提取的文本: {response_text}")
            
            suggested_codes = [code.strip() for code in response_text.split(',')]
            valid_codes = [code for code in suggested_codes if code in self.taxonomy.categories]
            print(f"解析后的分类代码: {valid_codes}")
            print(f"=== AI推荐结束 ===\n")
            
            return valid_codes
            
        except Exception as e:
            print(f"AI推荐出错: {str(e)}")
            print(f"错误类型: {type(e)}")
            print(f"=== AI推荐异常结束 ===\n")
            return self._fallback_suggestions(description)
    
    def _fallback_suggestions(self, query: str) -> List[str]:
        """基于规则的后备推荐方案"""
        query = query.lower()
        fallback_rules = {
            'deep learning': ['cs.LG', 'cs.AI'],
            'machine learning': ['cs.LG'],
            'neural': ['cs.LG', 'cs.NE'],
            'nlp': ['cs.CL'],
            'natural language': ['cs.CL'],
            'vision': ['cs.CV'],
            'robotics': ['cs.RO'],
        }
        
        for key, codes in fallback_rules.items():
            if key in query:
                return codes
        
        # 默认返回最相关的分类
        return ['cs.AI']
    
    def match_category(self, query: str) -> List[Tuple[str, float, str]]:
        """根据查询匹配最相关的分类"""
        matches = []
        
        # 遍历所有分类
        for _, row in self.taxonomy_df.iterrows():
            # 使用模糊匹配计算相似度
            group_score = fuzz.partial_ratio(query.lower(), row['Group'].lower())
            subgroup_score = fuzz.partial_ratio(query.lower(), row['Subgroup'].lower())
            desc_score = fuzz.partial_ratio(query.lower(), row['Description'].lower())
            
            # 综合相似度
            total_score = max(group_score, subgroup_score, desc_score)
            
            if total_score > 60:  # 相似度阈值
                matches.append((
                    row['Code'], 
                    total_score / 100.0, 
                    f"{row['Group']} > {row['Subgroup']} > {row['Description']}"
                ))
        
        # 按相似度排序
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:5]  # 返回前5个最相似的分类
    
    def interactive_select(self):
        """交互式选择研究领域，支持层级选择"""
        st.sidebar.header("🔬 研究领域选择")
        
        # 选择一级分类（Group）
        groups = list(self.taxonomy_hierarchy.keys())
        selected_group = st.sidebar.selectbox("选择研究大类", groups)
        
        # 选择二级分类（Subgroup）
        subgroups = list(self.taxonomy_hierarchy[selected_group].keys())
        selected_subgroup = st.sidebar.selectbox(f"选择 {selected_group} 下的子领域", subgroups)
        
        # 选择具体分类代码
        categories = self.taxonomy_hierarchy[selected_group][selected_subgroup]
        category_options = [
            f"{cat['code']} - {cat['description']}" 
            for cat in categories
        ]
        selected_category = st.sidebar.selectbox(
            f"选择 {selected_subgroup} 下的具体分类", 
            category_options
        )
        
        # 提取分类代码
        selected_code = selected_category.split(' - ')[0]
        
        # 显示选择信息
        st.sidebar.info(f"""
        已选择:
        - 大类: {selected_group}
        - 子领域: {selected_subgroup}
        - 具体分类: {selected_category}
        """)
        
        return selected_code 