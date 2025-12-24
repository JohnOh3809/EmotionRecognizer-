# PowerShell script to run YOLOv11 + DeepFace demo
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "YOLOv11 + DeepFace Live Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activating Python 3.11 virtual environment..." -ForegroundColor Green
& .\venv_deepface\Scripts\Activate.ps1

Write-Host ""
Write-Host "Running YOLOv11 + DeepFace Demo..." -ForegroundColor Green
Write-Host "DeepFace will download models on first run (this may take a moment)" -ForegroundColor Yellow
Write-Host ""
python YOLODEMO.py --yolo_model yolo11n-pose.pt

Read-Host "Press Enter to exit"

