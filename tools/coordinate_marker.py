"""ID card image coordinate marker tool.

Helps users visually identify field coordinates on document images.
"""
# 导入系统相关模块
import sys  # 提供系统相关功能，如命令行参数、路径操作
import os   # 提供操作系统相关功能，如文件路径检查
import cv2  # OpenCV库，用于图像处理和显示
import numpy as np  # 数值计算库，虽然未直接使用但常用
import argparse  # 命令行参数解析模块

# 获取项目根目录路径
# __file__是当前文件的完整路径，os.path.abspath获取绝对路径
# os.path.dirname两次获取父目录的父目录，即项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python模块搜索路径的开头
sys.path.insert(0, project_root)

# 从recognition模块导入配置相关的类
from recognition.models import FieldConfig, RecognitionConfig  # 字段配置和识别配置类
from recognition.config_loader import load_config  # 配置加载函数


class CoordinateMarker:
    """Interactive tool for marking field regions on document images."""

    def __init__(self, image_path: str, config_path: str = None):
        """初始化坐标标记器

        Args:
            image_path: 证件图像文件路径
            config_path: 配置文件路径（可选）
        """
        # 使用OpenCV读取图像
        self.image = cv2.imread(image_path)
        # 检查图像是否加载成功
        if self.image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        # 保存原始图像的副本，用于重置显示
        self.original_image = self.image.copy()
        # 标记状态：是否正在绘制矩形
        self.drawing = False
        # 矩形起始点坐标
        self.ix, self.iy = -1, -1
        # 存储手动标记的矩形区域列表
        self.rectangles = []
        # 存储从配置加载的预定义区域列表
        self.predefined_rectangles = []

        # 如果提供了配置文件路径，则加载预定义区域
        if config_path:
            self._load_predefined(config_path)

    def _load_predefined(self, config_path: str):
        """从配置文件加载预定义的字段区域

        Args:
            config_path: 配置文件路径
        """
        try:
            # 加载配置文件
            config = load_config(config_path)
            # 遍历配置中的所有字段
            for field in config.fields:
                # 检查字段坐标是否存在且为4个点（矩形四个顶点）
                if not field.coordinates or len(field.coordinates) != 4:
                    continue
                # 提取所有x坐标和y坐标
                xs = [c[0] for c in field.coordinates]
                ys = [c[1] for c in field.coordinates]
                # 创建预定义区域字典，存储左上角和右下角坐标
                self.predefined_rectangles.append({
                    'pt1': (min(xs), min(ys)),  # 左上角
                    'pt2': (max(xs), max(ys)),  # 右下角
                    'label': f"{field.name_cn} ({field.name_en})",  # 显示标签
                    'original_coords': field.coordinates  # 保存原始坐标
                })
            print(f"从配置文件加载了 {len(self.predefined_rectangles)} 个预定义区域")
        except Exception as e:
            print(f"加载配置文件失败: {e}")

    def _draw_all(self, img=None):
        """绘制所有矩形区域

        Args:
            img: 要绘制的图像，如果为None则使用原始图像副本

        Returns:
            绘制后的图像
        """
        # 如果没有提供图像，使用原始图像的副本
        if img is None:
            img = self.original_image.copy()
        # 绘制预定义区域（黄色）
        for r in self.predefined_rectangles:
            # 绘制黄色矩形框，线宽2像素
            cv2.rectangle(img, r['pt1'], r['pt2'], (0, 255, 255), 2)
            # 在矩形上方添加文字标签
            cv2.putText(img, r['label'], (r['pt1'][0], r['pt1'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # 绘制手动标记区域（蓝色）
        for r in self.rectangles:
            # 绘制蓝色矩形框，线宽2像素
            cv2.rectangle(img, r['pt1'], r['pt2'], (255, 0, 0), 2)
            # 在矩形上方添加文字标签
            cv2.putText(img, r['label'], (r['pt1'][0], r['pt1'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return img

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，处理鼠标事件

        Args:
            event: 鼠标事件类型
            x, y: 鼠标当前坐标
            flags: 事件标志
            param: 用户参数
        """
        window = 'Coordinate Marker'
        # 鼠标左键按下事件
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True  # 开始绘制
            self.ix, self.iy = x, y  # 记录起始点坐标
        # 鼠标移动事件（且正在绘制中）
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 先绘制所有已存在的矩形
            img = self._draw_all()
            # 绘制当前正在拖拽的矩形（绿色）
            cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            # 更新显示
            cv2.imshow(window, img)
        # 鼠标左键释放事件
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False  # 结束绘制
            # 计算矩形的左上角和右下角坐标
            pt1 = (min(self.ix, x), min(self.iy, y))
            pt2 = (max(self.ix, x), max(self.iy, y))
            # 检查矩形是否足够大（防止误触）
            if abs(pt2[0] - pt1[0]) > 10 and abs(pt2[1] - pt1[1]) > 10:
                # 提示用户输入字段名
                x1, y1 = pt1  # 左上角
                x2, y2 = pt2  # 右下角
                # 生成四个顶点的坐标（顺时针顺序：左上、右上、右下、左下）
                coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                name = input(f"请输入区域 ({pt1}, {pt2}) , coords={coords}的字段名: ")
                # 将新区域添加到列表中
                self.rectangles.append({'pt1': pt1, 'pt2': pt2, 'label': name})
            # 更新显示
            cv2.imshow(window, self._draw_all())

    def run(self):
        """运行交互式坐标标记工具的主循环"""
        window = 'Coordinate Marker'
        # 显示初始图像
        cv2.imshow(window, self._draw_all())
        # 设置鼠标回调函数
        cv2.setMouseCallback(window, self._mouse_callback)

        # 主循环，处理键盘事件
        while True:
            key = cv2.waitKey(1) & 0xFF  # 等待按键，1ms超时
            if key == ord('q'):  # 按q退出
                break
            elif key == ord('r'):  # 按r重置所有手动标记
                self.rectangles.clear()
                cv2.imshow(window, self._draw_all())
            elif key == ord(' '):  # 按空格打印手动标记的坐标
                self._print_manual_coords()
            elif key == ord('c'):  # 按c打印预定义的坐标
                self._print_predefined_coords()

        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()

    def _print_manual_coords(self):
        """打印手动标记的区域坐标"""
        print("\n=== 手动标记区域坐标 ===")
        for i, r in enumerate(self.rectangles):
            x1, y1 = r['pt1']  # 左上角
            x2, y2 = r['pt2']  # 右下角
            # 生成四个顶点的坐标（顺时针顺序：左上、右上、右下、左下）
            coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            print(f"  区域 {i + 1} ({r['label']}): coordinates={coords}")

    def _print_predefined_coords(self):
        """打印从配置文件加载的预定义坐标"""
        print("\n=== 预定义区域坐标 ===")
        for i, r in enumerate(self.predefined_rectangles):
            print(f"  区域 {i + 1} ({r['label']}): coordinates={r['original_coords']}")


def main():
    """主函数：程序的入口点"""
    print("证件图像坐标标记工具")
    print("=" * 50)
    print("操作: 拖拽标记区域 | R-重置 | 空格-打印坐标 | C-打印预定义 | Q-退出")
    print("颜色: 黄色=预定义区域 | 蓝色=手动标记\n")

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='证件图像坐标标记工具')
    # 添加位置参数：图像路径（可选）
    parser.add_argument('image_path', nargs='?', help='证件图像路径')
    # 添加可选参数：配置文件路径（默认值：../config/id_card-config.json）
    parser.add_argument('-c', '--config', default='../config/id_card-config.json', help='配置文件路径')
    # 解析命令行参数
    args = parser.parse_args()

    # 如果没有通过命令行提供图像路径，则提示用户输入
    image_path = args.image_path or input("请输入证件图像路径: ").strip()
    # 检查图像文件是否存在
    if not image_path or not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return

    # 创建坐标标记器实例
    marker = CoordinateMarker(image_path, args.config)
    # 显示图像尺寸
    print(f"图像尺寸: {marker.image.shape[1]} x {marker.image.shape[0]}")
    # 运行标记工具
    marker.run()


# 如果直接运行此脚本（而不是被导入），则执行main函数
if __name__ == "__main__":
    main()