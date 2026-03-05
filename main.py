"""证件识别系统命令行入口模块。

本模块提供证件识别系统的 CLI 接口，支持从图像文件识别证件信息，
并将结果输出到控制台或保存为 JSON 文件。
"""

import os
import sys
import json
import argparse
import logging

from recognition.pipeline import RecognitionPipeline
from recognition.ocr.tesseract import TesseractEngine
from recognition.config_loader import get_default_config_path
from recognition.validator import validate_id_number


def setup_logging(level: str = "INFO"):
    """配置日志输出格式和级别。

    Args:
        level: 日志级别，可选 DEBUG/INFO/WARNING/ERROR
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def print_results(doc_info):
    """以可读格式打印识别结果到控制台。

    Args:
        doc_info: 包含证件各字段信息的 DocumentInfo 对象
    """
    print("\n" + "=" * 40)
    print("  证件识别结果")
    print("=" * 40)
    print(f"  姓名:       {doc_info.name or '未识别'}")
    print(f"  身份证号:   {doc_info.id_number or '未识别'}")
    if doc_info.id_number:
        valid = validate_id_number(doc_info.id_number)
        print(f"  号码有效性: {'有效' if valid else '无效'}")
    print(f"  性别:       {doc_info.gender or '未识别'}")
    print(f"  出生日期:   {doc_info.birth_date or '未识别'}")
    if doc_info.age is not None:
        print(f"  年龄:       {doc_info.age}")
    print(f"  民族:       {doc_info.ethnicity or '未识别'}")
    print(f"  地址:       {doc_info.address or '未识别'}")
    print(f"  签发机关:   {doc_info.issuing_authority or '未识别'}")
    print(f"  有效期限:   {doc_info.validity_period or '未识别'}")
    print("=" * 40)


def save_results(doc_info, output_path: str):
    """将识别结果保存为 JSON 文件。

    Args:
        doc_info: 包含证件各字段信息的 DocumentInfo 对象
        output_path: 输出文件路径，若目录不存在会自动创建
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(doc_info.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {output_path}")


def main():
    """主入口函数：解析命令行参数并执行证件识别流程。"""
    parser = argparse.ArgumentParser(description="证件识别系统")
    parser.add_argument("image_path", nargs="?", default="./idcard.png",
                        help="证件图像路径")
    parser.add_argument("-c", "--config", help="配置文件路径")
    parser.add_argument("-o", "--output", help="输出结果JSON文件路径")
    parser.add_argument("--tesseract-path",
                        default=os.environ.get("TESSERACT_PATH",
                                               r"D:\soft\Tesseract-OCR\tesseract.exe"),
                        help="Tesseract可执行文件路径")
    parser.add_argument("--no-detect", action="store_true",
                        help="跳过文档区域自动检测")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件不存在: {args.image_path}")
        return 1

    config_path = args.config or get_default_config_path()
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return 1

    ocr_engine = TesseractEngine(tesseract_path=args.tesseract_path)
    pipeline = RecognitionPipeline.from_config_file(config_path, ocr_engine)
    doc_info = pipeline.process(
        args.image_path, detect_document=not args.no_detect
    )

    print_results(doc_info)

    if args.output:
        save_results(doc_info, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
