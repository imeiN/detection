@echo off
rem 快速启动脚本 - 证件识别系统

echo.
echo ================================
echo 证件识别系统
echo ================================
echo.

if "%~1"=="" (
    echo 使用方法:
    echo   %0 ^<证件图像路径^>
    echo.
    echo 示例:
    echo   %0 idcard.png
    echo.
    pause
    goto :eof
)

if not exist "%~1" (
    echo 错误: 文件 "%~1" 不存在
    pause
    goto :eof
)

echo 正在处理图像: %~1
echo.

python main.py "%~1"

echo.
echo 处理完成！
pause
