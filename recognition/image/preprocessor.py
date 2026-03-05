"""图像预处理模块，用于提升 OCR 识别效果。

流程：灰度化 → CLAHE 对比度增强 → 自适应二值化。
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """对图像进行预处理，提高 OCR 识别准确率。"""

    def preprocess(self, image: np.ndarray, params: dict = None) -> np.ndarray:
        """执行标准 OCR 预处理流程。

        流程：灰度化 → CLAHE 对比度增强 → 自适应二值化。

        Args:
            image: 输入图像（BGR 或灰度）
            params: 可选参数，如 adaptive_thresh_block_size、adaptive_thresh_c

        Returns:
            预处理后的二值图像
        """
        params = params or {}
        gray = self.to_grayscale(image)
        gray = self.enhance_contrast(gray)

        block_size = params.get("adaptive_thresh_block_size", 21)
        c_val = params.get("adaptive_thresh_c", 10)
        return self.adaptive_threshold(gray, block_size, c_val)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """将 BGR 图像转为灰度；若已是灰度则直接返回。"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """使用 CLAHE 增强局部对比度，适用于光照不均的背景。"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def adaptive_threshold(gray: np.ndarray, block_size: int = 21,
                           c: int = 10) -> np.ndarray:
        """使用高斯自适应阈值进行二值化。"""
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )

    @staticmethod
    def resize_for_ocr(image: np.ndarray, min_height: int = 50) -> np.ndarray:
        """若图像过小则按比例放大，避免 OCR 识别率过低。"""
        h, w = image.shape[:2]
        if h < min_height:
            scale = min_height / h
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        return image
