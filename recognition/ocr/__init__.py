"""OCR 引擎实现：抽象基类与 Tesseract 实现。"""

from .base import BaseOcrEngine
from .tesseract import TesseractEngine

__all__ = ["BaseOcrEngine", "TesseractEngine"]
