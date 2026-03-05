"""ID card image coordinate marker tool.

Helps users visually identify field coordinates on document images.
"""
# 导入系统相关模块
import sys
import os
import cv2
import argparse
import numpy as np

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
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img = self._draw_all()
            cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(window, img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            pt1 = (min(self.ix, x), min(self.iy, y))
            pt2 = (max(self.ix, x), max(self.iy, y))
            if abs(pt2[0] - pt1[0]) > 10 and abs(pt2[1] - pt1[1]) > 10:
                # 使用OpenCV窗口输入，而不是控制台input()
                self._get_field_name_with_opencv(pt1, pt2)
            cv2.imshow(window, self._draw_all())

    def _get_field_name_with_opencv(self, pt1, pt2):
        """使用OpenCV窗口获取字段名"""
        window = '输入字段名'
        cv2.namedWindow(window)

        # 创建一个黑色图像用于输入
        img = np.zeros((100, 400, 3), np.uint8)
        cv2.putText(img, "输入字段名，然后按Enter:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示输入提示
        cv2.imshow(window, img)

        # 这里需要实现一个简单的文本输入
        # 但OpenCV没有内置的文本输入功能，所以这比较复杂

        # 临时解决方案：使用预定义字段名列表
        self._show_field_selection_menu(pt1, pt2)

    def _show_field_selection_menu(self, pt1, pt2):
        """显示字段选择菜单"""
        print("\n请选择字段类型：")
        print("1. 姓名")
        print("2. 身份证号")
        print("3. 地址")
        print("4. 出生日期")
        print("5. 其他（自定义）")

        choice = input("请输入数字选择: ").strip()

        field_names = {
            '1': '姓名',
            '2': '身份证号',
            '3': '地址',
            '4': '出生日期',
            '5': None  # 需要自定义
        }

        if choice in field_names:
            if choice == '5':
                name = input("请输入自定义字段名: ").strip()
            else:
                name = field_names[choice]
            self.rectangles.append({'pt1': pt1, 'pt2': pt2, 'label': name})
        else:
            print("无效选择，请重试")
            self._show_field_selection_menu(pt1, pt2)

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
            if key == 255:
                continue
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