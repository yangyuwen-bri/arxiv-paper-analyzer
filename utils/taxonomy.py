import requests
from bs4 import BeautifulSoup
import json
import os
from typing import Dict, List
from pathlib import Path
import pandas as pd
from thefuzz import fuzz

class ArxivTaxonomy:
    """ArXiv分类体系管理类"""
    
    def __init__(self, taxonomy_file: str = 'arxiv_taxonomy.xlsx'):
        """
        初始化ArXiv分类系统
        
        Args:
            taxonomy_file: 分类信息Excel文件路径
        """
        # 确定文件路径（支持相对和绝对路径）
        self.taxonomy_file = self._find_taxonomy_file(taxonomy_file)
        
        # 加载分类数据
        self.taxonomy_df = self._load_taxonomy()
        
        # 构建分类字典
        self.categories = self._build_categories_dict()
    
    def _find_taxonomy_file(self, filename: str) -> str:
        """
        查找分类文件
        
        搜索路径优先级：
        1. 当前工作目录
        2. 脚本所在目录
        3. 项目根目录
        
        Args:
            filename: 分类文件名
        
        Returns:
            完整的文件路径
        """
        search_paths = [
            os.path.join(os.getcwd(), filename),  # 当前工作目录
            os.path.join(os.path.dirname(__file__), filename),  # 当前脚本目录
            os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)  # 项目根目录
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                print(f"找到分类文件: {path}")
                return path
        
        raise FileNotFoundError(f"未找到分类文件 {filename}")
    
    def _load_taxonomy(self) -> pd.DataFrame:
        """
        加载分类数据
        
        Returns:
            包含分类信息的DataFrame
        """
        try:
            df = pd.read_excel(self.taxonomy_file)
            
            # 验证必需的列
            required_columns = ['Group', 'Subgroup', 'Code', 'Description']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"缺少必需列: {col}")
            
            return df
        except Exception as e:
            print(f"加载分类文件出错: {e}")
            raise
    
    def _build_categories_dict(self) -> dict:
        """
        构建分类代码到详细信息的映射
        
        Returns:
            分类代码为键的字典
        """
        categories = {}
        for _, row in self.taxonomy_df.iterrows():
            code = row['Code']
            categories[code] = {
                'group': row['Group'],
                'subgroup': row['Subgroup'],
                'description': row['Description']
            }
        return categories
    
    def search_categories(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        根据查询搜索相关分类
        
        Args:
            query: 搜索关键词
            top_n: 返回的top结果数量
        
        Returns:
            相关分类列表，每个分类包含code、score和description
        """
        results = []
        query = query.lower()
        
        for _, row in self.taxonomy_df.iterrows():
            # 计算相似度
            group_score = fuzz.partial_ratio(query, row['Group'].lower())
            subgroup_score = fuzz.partial_ratio(query, row['Subgroup'].lower())
            desc_score = fuzz.partial_ratio(query, row['Description'].lower())
            
            # 取最高分
            score = max(group_score, subgroup_score, desc_score)
            
            if score > 60:  # 相似度阈值
                results.append({
                    'code': row['Code'],
                    'score': score,
                    'description': row['Description']
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_n]
    
    def get_category_info(self, code: str) -> Dict:
        """
        获取特定分类代码的详细信息
        
        Args:
            code: 分类代码
        
        Returns:
            分类详细信息字典
        """
        return self.categories.get(code, {
            'group': '未知',
            'subgroup': '未知',
            'description': '无描述'
        })
    
    def list_categories(self) -> List[Dict]:
        """列出所有分类
        Returns:
            List[Dict]: 分类列表，每个分类包含完整信息
        """
        return [{"code": code, **self.get_category_info(code)} 
                for code in self.categories]
    
    def get_group_statistics(self) -> Dict[str, Dict]:
        """获取分类统计信息
        Returns:
            Dict: 包含各组统计信息的字典
        """
        stats = {}
        for code, info in self.categories.items():
            group = info['group']
            if group not in stats:
                stats[group] = {
                    'total': 0,
                    'subgroups': set(),
                    'categories': []
                }
            stats[group]['total'] += 1
            stats[group]['subgroups'].add(info['subgroup'])
            stats[group]['categories'].append(code)
        
        # 将集合转换为列表以便JSON序列化
        for group in stats:
            stats[group]['subgroups'] = sorted(list(stats[group]['subgroups']))
            stats[group]['categories'] = sorted(stats[group]['categories'])
        
        return stats 