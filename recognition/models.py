"""证件识别数据模型定义。

本模块定义了识别配置、字段配置、识别结果及证件信息等核心数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class FieldConfig:
    """单个识别字段的配置。

    Attributes:
        name_en: 字段英文名，用于程序内部引用
        name_cn: 字段中文名，用于显示和日志
        coordinates: 字段在图像中的坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]，四顶点矩形
        recognition_params: 识别参数，如正则 pattern、字符白名单等
    """
    name_en: str
    name_cn: str
    coordinates: Optional[List[List[int]]] = None
    recognition_params: Dict = field(default_factory=dict)


@dataclass
class RecognitionConfig:
    """某类证件的整体识别配置。

    Attributes:
        fields: 字段配置列表
        image_processing_params: 图像预处理参数（如 reference_size、adaptive_thresh 等）
        ocr_params: OCR 引擎参数（如 psm_modes、char_whitelist 等）
        validation_rules: 校验规则配置
    """
    fields: List[FieldConfig]
    image_processing_params: Dict = field(default_factory=dict)
    ocr_params: Dict = field(default_factory=dict)
    validation_rules: Dict = field(default_factory=dict)


@dataclass
class FieldResult:
    """单个字段的识别结果。

    Attributes:
        name_en: 字段英文名
        name_cn: 字段中文名
        value: 识别出的文本值，未识别时为 None
        confidence: 置信度 0-100
        raw_text: OCR 原始输出文本
    """
    name_en: str
    name_cn: str
    value: Optional[str] = None
    confidence: float = 0.0
    raw_text: str = ""


@dataclass
class DocumentInfo:
    """识别后的证件信息汇总。

    对应身份证等证件的各字段，未识别字段为 None。
    """

    name: Optional[str] = None
    id_number: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    birth_date: Optional[str] = None
    address: Optional[str] = None
    issuing_authority: Optional[str] = None
    validity_period: Optional[str] = None
    age: Optional[int] = None

    def to_dict(self) -> Dict:
        """转换为字典，仅包含非空字段，便于 JSON 序列化。"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
