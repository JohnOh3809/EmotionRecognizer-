@echo off
echo ========================================
echo YOLOv11 + DeepFace Live Demo
echo ========================================
echo.
echo Activating Python 3.11 virtual environment...
call venv_deepface\Scripts\activate.bat

echo.
echo Running YOLOv11 + DeepFace Demo...
echo DeepFace will download models on first run (this may take a moment)
echo.
python YOLODEMO.py --yolo_model yolo11n-pose.pt

pause

