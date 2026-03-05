"""证件识别包。

提供证件图像识别、字段提取、校验及衍生信息补全等功能。
"""

from .models import DocumentInfo, FieldConfig, FieldResult, RecognitionConfig
from .pipeline import RecognitionPipeline
from .config_loader import load_config, save_config, get_default_config_path

__all__ = [
    "DocumentInfo",
    "FieldConfig",
    "FieldResult",
    "RecognitionConfig",
    "RecognitionPipeline",
    "load_config",
    "save_config",
    "get_default_config_path",
]
