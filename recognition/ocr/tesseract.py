"""Tesseract OCR 引擎实现。

基于 pytesseract 调用 Tesseract，支持中英文识别及字符白名单。
"""

import logging
from typing import List, Dict, Optional
import numpy as np
import pytesseract
from PIL import Image
from .base import BaseOcrEngine

logger = logging.getLogger(__name__)


class TesseractEngine(BaseOcrEngine):
    """基于 Tesseract 的 OCR 引擎实现。"""

    def __init__(self, tesseract_path: Optional[str] = None):
        """初始化 Tesseract 引擎。

        Args:
            tesseract_path: Tesseract 可执行文件路径，不指定则使用系统 PATH
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def recognize(self, image: np.ndarray, lang: str = "chi_sim",
                  psm: int = 6, **kwargs) -> str:
        """对图像执行 OCR，返回识别文本。支持 lang、psm、char_whitelist 等参数。"""
        char_whitelist = kwargs.get("char_whitelist", "")
        config_str = f"--oem 3 --psm {psm}"
        if char_whitelist:
            config_str += f" -c tessedit_char_whitelist={char_whitelist}"

        pil_image = Image.fromarray(image)
        try:
            return pytesseract.image_to_string(
                pil_image, config=config_str, lang=lang
            ).strip()
        except Exception as e:
            logger.warning("OCR识别失败 (lang=%s, PSM=%d): %s", lang, psm, e)
            return ""

    def recognize_with_details(self, image: np.ndarray, lang: str = "chi_sim",
                               psm: int = 6, **kwargs) -> List[Dict]:
        """执行 OCR 并返回带位置和置信度的详细结果列表。"""
        config_str = f"--oem 3 --psm {psm}"
        pil_image = Image.fromarray(image)

        try:
            data = pytesseract.image_to_data(
                pil_image, output_type=pytesseract.Output.DICT,
                config=config_str, lang=lang
            )
        except Exception as e:
            logger.warning("OCR详细识别失败 (PSM=%d): %s", psm, e)
            return []

        results = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            try:
                conf = float(data['conf'][i])
                x, y = int(data['left'][i]), int(data['top'][i])
                w, h = int(data['width'][i]), int(data['height'][i])
                if w > 0 and h > 0:
                    results.append({
                        'text': text, 'confidence': conf,
                        'bbox': (x, y, w, h)
                    })
            except (ValueError, IndexError):
                continue
        return results
