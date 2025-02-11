import os
from dotenv import load_dotenv
import streamlit as st
import logging

# 优先使用 Streamlit 云的 Secrets
load_dotenv()

logger = logging.getLogger(__name__)

def get_config_value(env_key: str, secret_key: str, default_value: str = "") -> str:
    """获取配置值，优先级：环境变量 > Streamlit Secrets > 默认值"""
    try:
        return os.getenv(env_key) or st.secrets.get(secret_key, default_value)
    except:
        return os.getenv(env_key, default_value)

# 安全获取配置
OPENAI_CONFIG = {
    "api_key": get_config_value("OPENAI_API_KEY", "OPENAI_API_KEY"),
    "base_url": get_config_value("OPENAI_BASE_URL", "OPENAI_BASE_URL", "https://api.openai.xyz/v1"),
    "model": get_config_value("OPENAI_MODEL", "OPENAI_MODEL", "o1-preview")
}

DEEPSEEK_CONFIG = {
    "api_key": get_config_value("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    "base_url": get_config_value("DEEPSEEK_BASE_URL", "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    "model": get_config_value("DEEPSEEK_MODEL", "DEEPSEEK_MODEL", "deepseek-chat")
}

NVIDIA_DEEPSEEK_CONFIG = {
    "api_key": get_config_value("NVIDIA_DEEPSEEK_API_KEY", "NVIDIA_DEEPSEEK_API_KEY"),
    "base_url": get_config_value("NVIDIA_DEEPSEEK_BASE_URL", "NVIDIA_DEEPSEEK_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    "model": get_config_value("NVIDIA_DEEPSEEK_MODEL", "NVIDIA_DEEPSEEK_MODEL", "deepseek-ai/deepseek-r1"),
    "proxy": get_config_value("HTTPS_PROXY", "")
}

# 配置检查函数
def validate_config():
    """验证 API 配置是否正确"""
    configs = [
        ("OpenAI", OPENAI_CONFIG),
        ("DeepSeek", DEEPSEEK_CONFIG),
        ("Nvidia DeepSeek", NVIDIA_DEEPSEEK_CONFIG)
    ]
    
    for name, config in configs:
        logger.info(f"验证 {name} 配置:")
        logger.info(f"  API Key: {'已配置' if config.get('api_key') else '未配置'}")
        logger.info(f"  Base URL: {config.get('base_url')}")
        if name == "DeepSeek":
            logger.info(f"  Model: {config.get('model')}")
