"""证件识别主流程编排模块。

流水线步骤：
  1. 文档区域检测与透视校正
  2. 按字段提取 ROI 并执行 OCR
  3. 后处理（校验与衍生字段补全）
"""

import logging
from typing import List
import cv2
from .models import DocumentInfo, RecognitionConfig, FieldResult
from .config_loader import load_config
from .image.detector import DocumentDetector
from .recognizer import FieldRecognizer
from .validator import enrich_document_info
from .ocr.base import BaseOcrEngine

logger = logging.getLogger(__name__)


class RecognitionPipeline:
    """编排完整的证件识别流程：加载图像 → 检测/校正 → 字段识别 → 后处理。"""

    def __init__(self, config: RecognitionConfig, ocr_engine: BaseOcrEngine):
        """初始化识别流水线。

        Args:
            config: 识别配置
            ocr_engine: OCR 引擎实例
        """
        self.config = config
        self.detector = DocumentDetector()
        self.recognizer = FieldRecognizer(ocr_engine, config)

    def process(self, image_path: str,
                detect_document: bool = True) -> DocumentInfo:
        """执行完整识别流程。

        Args:
            image_path: 输入图像路径
            detect_document: 是否自动检测文档区域并做透视校正

        Returns:
            识别得到的 DocumentInfo 对象
        """
        image = self._load_image(image_path, detect_document)
        logger.info("图像尺寸: %dx%d", image.shape[1], image.shape[0])

        cv2.imwrite('detection.png', image)

        logger.info("开始识别各字段...")
        field_results = self.recognizer.recognize_all(image)

        doc_info = self._build_document_info(field_results)
        doc_info = enrich_document_info(doc_info)
        return doc_info

    def _load_image(self, image_path: str, detect_document: bool):
        """加载图像，可选进行文档区域检测与透视校正。"""
        if detect_document:
            logger.info("检测文档区域并透视校正...")
            return self.detector.detect(image_path)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return image

    @staticmethod
    def _build_document_info(results: List[FieldResult]) -> DocumentInfo:
        """将各字段识别结果汇总为 DocumentInfo 对象。"""
        doc = DocumentInfo()
        for r in results:
            if r.value and hasattr(doc, r.name_en):
                setattr(doc, r.name_en, r.value)
        return doc

    @classmethod
    def from_config_file(cls, config_path: str,
                         ocr_engine: BaseOcrEngine) -> "RecognitionPipeline":
        """从配置文件路径创建识别流水线实例。"""
        config = load_config(config_path)
        return cls(config, ocr_engine)
