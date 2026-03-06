"""文档区域检测与透视校正模块。

从拍照图像中检测证件/文档的四角轮廓，执行透视变换得到校正后的图像。
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 证件最小面积占比，过滤背景杂物轮廓
_MIN_CONTOUR_AREA_RATIO = 0.15

# approxPolyDP 容差梯度（由宽松到严格）
_APPROX_TOLERANCES = [0.02, 0.03, 0.04, 0.05, 0.015]


class DocumentDetector:
    """检测图像中的文档区域并执行透视校正，并可自动纠正方向。"""

    def detect(self, image_path: str) -> np.ndarray:
        """检测文档并执行透视校正，最后自动纠正方向。

        Returns:
            校正后的图像；若未检测到四边形轮廓则返回原图
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        contour = self._find_document_contour(image)
        if contour is not None:
            logger.info("检测到文档边框，正在进行透视校正")
            image = self._perspective_transform(image, contour)
        else:
            logger.warning("未检测到文档边框，返回原图")

        image = self._correct_orientation(image)
        return image

    def _find_document_contour(self, image: np.ndarray):
        """多策略查找文档轮廓，优先使用最小外接矩形（对圆角更鲁棒）。

        策略顺序：
          1. HSV 亮度 Otsu 阈值：证件（亮白）vs 背景（暗木纹/桌面），效果最好
          2. Otsu Canny + 大核闭运算：抑制木纹纹理干扰
          3. 固定 Canny（75/200）+ 小核闭运算：作为兜底
        每种策略先尝试 minAreaRect，再尝试 approxPolyDP。
        """
        img_area = image.shape[0] * image.shape[1]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for name, binary in self._build_binary_candidates(image, gray):
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < img_area * _MIN_CONTOUR_AREA_RATIO:
                    continue

                # minAreaRect：对圆角/轻微倾斜都有效
                rect = cv2.minAreaRect(contour)
                w_r, h_r = rect[1]
                if min(w_r, h_r) > 0:
                    box = cv2.boxPoints(rect).astype("float32")
                    logger.info("[%s] minAreaRect 成功，面积占比: %.1f%%", name, area / img_area * 100)
                    cv2.imwrite("edged.png", binary)
                    return box

                # approxPolyDP 兜底
                for tolerance in _APPROX_TOLERANCES:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, tolerance * peri, True)
                    if len(approx) == 4:
                        logger.info("[%s] approxPolyDP 成功，容差: %.3f", name, tolerance)
                        cv2.imwrite("edged.png", binary)
                        return approx.reshape(4, 2).astype("float32")

        cv2.imwrite("edged.png", gray)
        return None

    @staticmethod
    def _build_binary_candidates(image: np.ndarray, gray: np.ndarray):
        """生成多种二值化图，供轮廓检测逐一尝试。"""
        results = []

        # 策略1：HSV 亮度 Otsu — 对亮色证件 vs 深色背景最有效
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, thresh_v = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k20 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        thresh_v = cv2.morphologyEx(thresh_v, cv2.MORPH_CLOSE, k20)
        thresh_v = cv2.morphologyEx(thresh_v, cv2.MORPH_OPEN, k20)
        results.append(("HSV亮度", thresh_v))

        # 策略2：Otsu Canny + 大核闭运算 — 抑制木纹纹理
        blurred11 = cv2.GaussianBlur(gray, (11, 11), 0)
        ht, _ = cv2.threshold(blurred11, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edged_otsu = cv2.Canny(blurred11, 0.5 * ht, ht)
        k15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        edged_otsu = cv2.morphologyEx(edged_otsu, cv2.MORPH_CLOSE, k15)
        results.append(("OtsuCanny", edged_otsu))

        # 策略3：固定 Canny 阈值 — 兜底
        blurred5 = cv2.GaussianBlur(gray, (5, 5), 0)
        edged_fixed = cv2.Canny(blurred5, 75, 200)
        k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged_fixed = cv2.morphologyEx(edged_fixed, cv2.MORPH_CLOSE, k5)
        results.append(("固定Canny", edged_fixed))

        return results

    def _correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """检测并纠正图像方向（颠倒 / 横竖放错）。

        优先使用 Tesseract OSD 检测旋转角度（支持 90/180/270°），
        若不可用则退回到宽高比判断（身份证横向约 1.58:1）。
        """
        try:
            import pytesseract
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
            angle = osd.get("rotate", 0)
            logger.info("OSD 检测旋转角度: %d°", angle)
            if angle != 0:
                return self._rotate_image(image, angle)
            return image
        except Exception as e:
            logger.debug("OSD 方向检测不可用: %s，改用宽高比判断", e)
            return self._fix_orientation_by_aspect(image)

    @staticmethod
    def _fix_orientation_by_aspect(image: np.ndarray) -> np.ndarray:
        """根据宽高比判断是否需要旋转 90°（身份证应为横向）。

        竖向图像旋转 90° 逆时针（CCW），使底部内容保持在底部。
        """
        h, w = image.shape[:2]
        if h > w:
            logger.info("图像为竖向 (%dx%d)，旋转 90°CCW 为横向", w, h)
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """按 Tesseract OSD 返回的角度旋转图像（支持 90/180/270°）。"""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

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
