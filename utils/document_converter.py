import subprocess
from typing import Optional
import os
from pathlib import Path
import json
import shutil

class DocumentConverter:
    def __init__(self):
        """初始化转换器"""
        # 使用 shutil.which 查找 md-to-pdf 路径
        self.md_to_pdf_cli = shutil.which("md-to-pdf")
        
        if not self.md_to_pdf_cli:
            print("警告: 未找到 md-to-pdf 命令行工具")
            print("请运行: npm install -g md-to-pdf")
    
    def md_to_pdf(self, md_file: str) -> Optional[str]:
        """将Markdown文件转换为PDF"""
        try:
            pdf_file = md_file.rsplit('.', 1)[0] + '.pdf'
            
            if not self.md_to_pdf_cli:
                print("错误: md-to-pdf 未安装")
                return None
            
            # 使用引号包装JSON字符串
            marked_options = '"' + json.dumps({
                "breaks": True,
                "mangle": False,
                "headerIds": False
            }).replace('"', '\\"') + '"'
            
            pdf_options = '"' + json.dumps({
                "format": "A4",
                "margin": {
                    "top": "30mm",
                    "bottom": "30mm",
                    "left": "20mm",
                    "right": "20mm"
                }
            }).replace('"', '\\"') + '"'
            
            launch_options = '"' + json.dumps({
                "args": ["--no-sandbox"]
            }).replace('"', '\\"') + '"'
            
            # CSS样式使用单引号包装
            css_styles = """'
                .math {
                    font-family: "Latin Modern Math", "STIX Two Math", serif;
                }
                .katex { 
                    font-size: 1.1em; 
                }
                .katex-display {
                    margin: 1em 0;
                    overflow-x: auto;
                    overflow-y: hidden;
                }
            '"""
            
            # 构建命令
            cmd = [
                "node",
                self.md_to_pdf_cli,
                md_file,
                "--marked-options", marked_options,
                "--pdf-options", pdf_options,
                "--launch-options", launch_options,
                "--css", css_styles.strip()
            ]
            
            print("执行PDF转换...")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            if result.stderr:
                print(f"警告: {result.stderr}")
            
            if os.path.exists(pdf_file):
                print(f"PDF生成成功: {pdf_file}")
                return pdf_file
            else:
                print("PDF生成失败: 输出文件未创建")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"PDF转换失败: {str(e)}")
            if e.stderr:
                print(f"错误详情: {e.stderr}")
            return None
        except Exception as e:
            print(f"转换过程出错: {str(e)}")
            return None 