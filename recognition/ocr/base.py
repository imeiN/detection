"""OCR 引擎抽象基类。

若要新增 OCR 引擎（如 PaddleOCR、EasyOCR），继承 BaseOcrEngine 并实现两个抽象方法即可。
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseOcrEngine(ABC):
    """所有 OCR 引擎实现必须遵循的接口。"""

    @abstractmethod
    def recognize(self, image: np.ndarray, lang: str = "chi_sim",
                  psm: int = 6, **kwargs) -> str:
        """从图像中识别文本。

        Args:
            image: 预处理后的灰度或二值图像
            lang: 语言模型标识，如 chi_sim、eng
            psm: 页面分割模式（Tesseract 专用，其他引擎可忽略）

        Returns:
            识别出的文本字符串
        """

    @abstractmethod
    def recognize_with_details(self, image: np.ndarray, lang: str = "chi_sim",
                               psm: int = 6, **kwargs) -> List[Dict]:
        """识别文本并返回带位置和置信度的详细结果。

        Returns:
            字典列表，每项包含 'text'、'confidence'、'bbox'。
            bbox 格式为 (x, y, width, height)。
        """
