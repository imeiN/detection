"""证件识别配置的加载与保存。

支持从 JSON 文件加载识别配置，或将配置对象保存为 JSON。
"""

import json
import os
from .models import FieldConfig, RecognitionConfig


def load_config(file_path: str) -> RecognitionConfig:
    """从 JSON 文件加载识别配置。

    Args:
        file_path: 配置文件路径

    Returns:
        RecognitionConfig 对象

    Raises:
        FileNotFoundError: 配置文件不存在时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fields = [
        FieldConfig(
            name_en=fd["name_en"],
            name_cn=fd["name_cn"],
            coordinates=fd.get("coordinates"),
            recognition_params=fd.get("recognition_params", {})
        )
        for fd in data.get("fields", [])
    ]

    return RecognitionConfig(
        fields=fields,
        image_processing_params=data.get("image_processing_params", {}),
        ocr_params=data.get("ocr_params", {}),
        validation_rules=data.get("validation_rules", {})
    )


def save_config(config: RecognitionConfig, file_path: str) -> None:
    """将识别配置保存为 JSON 文件。

    Args:
        config: 要保存的 RecognitionConfig 对象
        file_path: 输出文件路径，目录不存在时会自动创建
    """
    data = {
        "fields": [
            {
                "name_en": f.name_en,
                "name_cn": f.name_cn,
                "coordinates": f.coordinates,
                "recognition_params": f.recognition_params
            }
            for f in config.fields
        ],
        "image_processing_params": config.image_processing_params,
        "ocr_params": config.ocr_params,
        "validation_rules": config.validation_rules
    }

    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_default_config_path(doc_type: str = "id_card") -> str:
    """获取指定证件类型的默认配置文件路径。

    Args:
        doc_type: 证件类型，如 'id_card'，对应 config/{doc_type}_config.json

    Returns:
        配置文件的绝对路径
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    return os.path.join(base_dir, f"{doc_type}_config.json")
