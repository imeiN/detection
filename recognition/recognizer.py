"""基于 ROI 提取的证件字段识别模块。

从图像中按配置坐标裁剪各字段区域，经预处理后调用 OCR 识别，并按正则匹配提取有效文本。
"""

import re
import logging
from typing import List, Optional

import cv2
import numpy as np
from .models import FieldConfig, FieldResult, RecognitionConfig
from .ocr.base import BaseOcrEngine
from .image.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

# Tesseract 默认尝试的 PSM 模式：6=块文本, 7=单行, 13=原始行
_DEFAULT_PSM_MODES = [6, 7, 13]


class FieldRecognizer:
    """通过提取 ROI 区域并调用 OCR 识别各字段文本。"""

    def __init__(self, ocr_engine: BaseOcrEngine, config: RecognitionConfig):
        """初始化字段识别器。

        Args:
            ocr_engine: OCR 引擎
            config: 识别配置
        """
        self.ocr_engine = ocr_engine
        self.config = config
        self.preprocessor = ImagePreprocessor()

    def recognize_all(self, image: np.ndarray) -> List[FieldResult]:
        """识别图像中所有配置的字段。

        Args:
            image: 输入图像（BGR 或灰度，通常为透视校正后的证件图）

        Returns:
            各字段的识别结果列表，顺序与 config.fields 一致
        """
        results = []
        for field_cfg in self.config.fields:
            result = self.recognize_field(image, field_cfg)
            results.append(result)
            if result.value:
                logger.info("字段 [%s]: %s", field_cfg.name_cn, result.value)
            else:
                logger.warning("字段 [%s]: 未识别到", field_cfg.name_cn)
        return results

    def recognize_field(self, image: np.ndarray,
                        field_cfg: FieldConfig) -> FieldResult:
        """识别单个字段：裁剪 ROI → 预处理 → 多 PSM 尝试 OCR → 正则匹配。

        Args:
            image: 输入图像
            field_cfg: 该字段的配置（坐标、正则等）

        Returns:
            FieldResult，识别成功时含 value，失败时 value 为空
        """
        empty = FieldResult(name_en=field_cfg.name_en, name_cn=field_cfg.name_cn)

        # 1. 坐标缩放（若配置了 reference_size）并裁剪 ROI
        coords = self._scale_coordinates(field_cfg.coordinates, image.shape)
        roi = self._extract_roi(image, coords)
        if roi is None:
            return empty

        # 2. 图像预处理：过小则放大，再灰度/二值化
        roi = self.preprocessor.resize_for_ocr(roi, min_height=50)
        processed = self.preprocessor.preprocess(roi, self.config.image_processing_params)
        cv2.imwrite("ROI_" + field_cfg.name_en + '.png', processed)

        # 3. 根据字段类型选择 OCR 参数
        lang = self._determine_lang(field_cfg)
        whitelist = self._determine_whitelist(field_cfg)
        psm_modes = self.config.ocr_params.get("psm_modes", _DEFAULT_PSM_MODES)
        pattern = (field_cfg.recognition_params or {}).get("pattern", r".*")

        # 4. 依次尝试不同 PSM 模式，直到正则匹配成功
        for psm in psm_modes:
            text = self.ocr_engine.recognize(
                processed, lang=lang, psm=psm, char_whitelist=whitelist
            )
            if not text:
                continue

            logger.debug("PSM %d → '%s'", psm, text)
            # 中文字段去除空格再匹配，数字/英文保留原样
            text_to_match = re.sub(r'\s+', '', text) if self._is_chinese_pattern(pattern) else text
            value = self._match_pattern(text_to_match, pattern, field_cfg.name_cn)
            if value:
                return FieldResult(
                    name_en=field_cfg.name_en, name_cn=field_cfg.name_cn,
                    value=value, confidence=80.0, raw_text=text
                )

        return empty

    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------

    def _scale_coordinates(self, coordinates, image_shape):
        """若配置了 reference_size，则按比例缩放坐标以适配当前图像尺寸。

        配置中的坐标通常基于参考尺寸（如 1000x600），实际图像可能不同，
        需按 w/ref_w、h/ref_h 缩放后使用。

        Args:
            coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 四顶点
            image_shape: 当前图像的 shape，取 (h, w)

        Returns:
            缩放后的坐标列表，无 reference_size 时原样返回
        """
        if not coordinates:
            return coordinates
        ref = self.config.image_processing_params.get("reference_size")
        if not ref:
            return coordinates
        ref_w, ref_h = ref
        h, w = image_shape[:2]
        scale_x, scale_y = w / ref_w, h / ref_h
        return [[int(x * scale_x), int(y * scale_y)] for x, y in coordinates]

    @staticmethod
    def _extract_roi(image: np.ndarray, coordinates,
                     padding: int = 5) -> Optional[np.ndarray]:
        """根据四顶点坐标裁剪 ROI，并添加 padding 像素边距。

        Args:
            image: 输入图像
            coordinates: 四顶点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]，取外接矩形
            padding: 四边扩展像素数，避免裁切过紧

        Returns:
            裁剪后的 ROI 图像，坐标无效或区域为空时返回 None
        """
        if not coordinates or len(coordinates) != 4:
            return None

        x_vals = [c[0] for c in coordinates]
        y_vals = [c[1] for c in coordinates]
        h, w = image.shape[:2]

        # 计算外接矩形并加 padding，同时限制在图像范围内
        x_min = max(0, min(x_vals) - padding)
        x_max = min(w, max(x_vals) + padding)
        y_min = max(0, min(y_vals) - padding)
        y_max = min(h, max(y_vals) + padding)

        if x_max <= x_min or y_max <= y_min:
            logger.warning("ROI坐标无效: x=[%d,%d] y=[%d,%d]",
                           x_min, x_max, y_min, y_max)
            return None

        roi = image[y_min:y_max, x_min:x_max]
        return roi if roi.size > 0 else None

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_pattern(text: str, pattern: str,
                       field_label: str) -> Optional[str]:
        """用正则匹配文本，排除字段标签（如「姓名」）干扰，返回有效值。

        OCR 可能把「姓名 张三」一起识别，需过滤掉标签只保留「张三」。
        优先返回捕获组内容，否则返回整段匹配；若匹配到的是标签则跳过。

        Args:
            text: OCR 原始输出
            pattern: 正则表达式
            field_label: 字段中文标签，用于过滤

        Returns:
            匹配到的有效文本，无匹配时返回 None
        """
        for match in re.finditer(pattern, text):
            # 有捕获组时优先取组内容
            if match.lastindex:
                for group in match.groups():
                    if group and group.strip() and group.strip() != field_label:
                        return group.strip()
            value = match.group(0).strip()
            if value and value != field_label:
                return value
        return None

    # ------------------------------------------------------------------
    # Field-aware OCR parameter selection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_chinese_pattern(pattern: str) -> bool:
        """判断正则是否包含中文字符范围，决定是否使用中文 OCR 模型。

        若 pattern 含 \u4e00-\u9fff 或 4e00/9fa5/9fff 等 Unicode 表示，
        则需用 chi_sim 且不做字符白名单限制。

        Args:
            pattern: 字段的正则表达式

        Returns:
            True 表示需中文识别
        """
        if '4e00' in pattern or '9fa5' in pattern or '9fff' in pattern:
            return True
        return bool(re.search(r'[\u4e00-\u9fff]', pattern))

    def _determine_lang(self, field_cfg: FieldConfig) -> str:
        """根据字段正则选择 OCR 语言：含中文用 chi_sim，否则用 eng。"""
        pattern = (field_cfg.recognition_params or {}).get("pattern", "")
        if self._is_chinese_pattern(pattern):
            return "chi_sim"
        return "eng"

    def _determine_whitelist(self, field_cfg: FieldConfig) -> str:
        """确定字符白名单：中文字段无白名单，数字字段用配置的 whitelist。

        身份证号等纯数字字段可限制为 0-9X，减少误识别。
        中文字段不设白名单，否则会过滤掉汉字。

        Args:
            field_cfg: 字段配置

        Returns:
            白名单字符串，空串表示不限制
        """
        params = field_cfg.recognition_params or {}
        if "char_whitelist" in params:
            return params["char_whitelist"]
        if self._is_chinese_pattern(params.get("pattern", "")):
            return ""
        return self.config.ocr_params.get("char_whitelist", "")
