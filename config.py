import os
from dotenv import load_dotenv
import streamlit as st

# 优先使用 Streamlit 云的 Secrets
load_dotenv()

# 安全获取配置，如果环境变量未设置则使用默认值
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""),
    "base_url": os.getenv("OPENAI_BASE_URL") or st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
}

DEEPSEEK_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY", ""),
    "base_url": os.getenv("DEEPSEEK_BASE_URL") or st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
}

# 配置检查函数
def validate_config():
    """验证 API 配置是否正确"""
    configs = [
        ("OpenAI", OPENAI_CONFIG),
        ("DeepSeek", DEEPSEEK_CONFIG)
    ]
    
    for name, config in configs:
        if not config.get("api_key"):
            print(f"警告：{name} API 密钥未配置")
        if not config.get("base_url"):
            print(f"警告：{name} 基础 URL 未配置")
