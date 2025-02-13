import requests
import os
from pathlib import Path
import logging

class FontManager:
    """
    字体管理类，处理开源字体的下载和缓存
    功能：
    1. 自动下载 Noto Sans CJK 字体
    2. 管理本地字体缓存
    3. 提供字体路径查询
    """
    FONT_CACHE = Path.home() / ".cache" / "arxiv_analyzer_fonts"
    NOTO_FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansCJKsc-Regular.otf"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.FONT_CACHE.mkdir(parents=True, exist_ok=True)
        
    def get_noto_font(self) -> str:
        """获取Noto字体本地路径，自动下载缺失字体"""
        font_path = self.FONT_CACHE / "NotoSansCJKsc-Regular.otf"
        if not font_path.exists():
            try:
                self.logger.info("开始下载 Noto Sans CJK 字体...")
                response = requests.get(self.NOTO_FONT_URL, timeout=10)
                font_path.write_bytes(response.content)
                self.logger.info(f"字体已保存到: {font_path}")
            except Exception as e:
                self.logger.error(f"字体下载失败: {str(e)}")
                return ""
        return str(font_path)

    def get_fallback_fonts(self) -> list:
        """获取备用字体列表"""
        return [
            "Arial Unicode MS",
            "Microsoft YaHei",
            "WenQuanYi Micro Hei",
            "sans-serif"
        ]