@echo off
setlocal
cd /d %~dp0

REM Create venv if it doesn't exist
if not exist "venv\Scripts\python.exe" (
    C:\Python313\python.exe -m venv venv
)

REM Activate venv
call "venv\Scripts\activate"

REM Upgrade pip and install requirements
python -m pip install --upgrade pip
if exist requirements.txt (
    python -m pip install -r requirements.txt
) else (
    python -m pip install flask pillow requests flask-cors numpy
)

REM Run the app
python app.py

endlocal
pause

