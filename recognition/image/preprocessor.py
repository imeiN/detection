"""图像预处理模块，用于提升 OCR 识别效果。

流程：灰度化 → 去噪（可选）→ CLAHE 对比度增强（可选）→ 自适应二值化。
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """对图像进行预处理，提高 OCR 识别准确率。"""

    def preprocess(self, image: np.ndarray, params: dict = None) -> np.ndarray:
        """执行标准 OCR 预处理流程。

        流程：灰度化 → 去噪（可选）→ CLAHE 对比度增强（可选）→ 自适应二值化。

        Args:
            image: 输入图像（BGR 或灰度）
            params: 可选参数，如 noise_reduction、noise_reduction_kernel、
                contrast_enhancement、adaptive_thresh_block_size、adaptive_thresh_c

        Returns:
            预处理后的二值图像
        """
        params = params or {}
        gray = self.to_grayscale(image)

        # 去噪：在二值化前减少椒盐噪点，避免噪声被误识别为文字
        if params.get("noise_reduction", True):
            kernel = params.get("noise_reduction_kernel", [3, 3])
            gray = self.denoise(gray, kernel)

        # CLAHE 对比度增强：光照不均时有用，但会放大噪声，默认开启
        if params.get("contrast_enhancement", True):
            clip_limit = params.get("clahe_clip_limit", 2.0)
            gray = self.enhance_contrast(gray, clip_limit)

        block_size = params.get("adaptive_thresh_block_size", 15)
        c_val = params.get("adaptive_thresh_c", 10)
        return self.adaptive_threshold(gray, block_size, c_val)

    @staticmethod
    def denoise(gray: np.ndarray, kernel_size=(3, 3)) -> np.ndarray:
        """高斯模糊去噪，减少椒盐噪点，二值化前使用效果更好。

        Args:
            gray: 灰度图像
            kernel_size: 卷积核大小，(3,3) 或 (5,5)，必须为奇数

        Returns:
            去噪后的灰度图像
        """
        kx, ky = kernel_size
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        return cv2.GaussianBlur(gray, (kx, ky), 0)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """将 BGR 图像转为灰度；若已是灰度则直接返回。"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """使用 CLAHE 增强局部对比度，适用于光照不均的背景。

        Args:
            gray: 灰度图像
            clip_limit: 对比度限制，过大可能放大噪声，建议 1.5~2.5
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
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
