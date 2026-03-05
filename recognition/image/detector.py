"""文档区域检测与透视校正模块。

从拍照图像中检测证件/文档的四角轮廓，执行透视变换得到校正后的图像。
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DocumentDetector:
    """检测图像中的文档区域并执行透视校正。"""

    def __init__(self, approx_tolerance: float = 0.015):
        """初始化检测器。

        Args:
            approx_tolerance: 轮廓近似容差，用于判断四边形（默认 0.015 * 周长）
        """
        self.approx_tolerance = approx_tolerance

    def detect(self, image_path: str) -> np.ndarray:
        """检测文档并执行透视校正。

        Returns:
            校正后的图像；若未检测到四边形轮廓则返回原图
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        contour = self._find_document_contour(image)
        if contour is not None:
            logger.info("检测到文档边框，正在进行透视校正")
            return self._perspective_transform(image, contour)

        logger.warning("未检测到文档边框，返回原图")
        return image

    def _find_document_contour(self, image: np.ndarray):
        """查找最大的四边形轮廓（灰度、高斯模糊、Canny 边缘、轮廓筛选）。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(
            edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_tolerance * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        return None

    def _perspective_transform(self, image: np.ndarray,
                               pts: np.ndarray) -> np.ndarray:
        """对四顶点执行透视变换，得到校正后的矩形图像。"""
        rect = self._order_points(pts.astype("float32"))
        tl, tr, br, bl = rect

        max_width = max(
            int(np.linalg.norm(br - bl)),
            int(np.linalg.norm(tr - tl))
        )
        max_height = max(
            int(np.linalg.norm(tr - br)),
            int(np.linalg.norm(tl - bl))
        )

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """将四个顶点按顺序排列：左上、右上、右下、左下。"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
