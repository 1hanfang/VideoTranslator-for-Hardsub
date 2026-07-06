@echo off
rem Launcher for the video subtitle translator.
rem NOTE: keep this file pure ASCII - cmd misparses multi-byte chars under chcp 65001.
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
title Video Subtitle Translator

set "PY="
if exist "runtime\venv\Scripts\python.exe" set "PY=runtime\venv\Scripts\python.exe"
if "!PY!"=="" (
    rem verify python actually runs - the MS Store stub does not
    python -c "import sys" >nul 2>nul && set "PY=python"
)
if "!PY!"=="" (
    if exist "runtime\pyembed\python.exe" set "PY=runtime\pyembed\python.exe"
)
if "!PY!"=="" (
    echo [setup] No Python found. Downloading portable Python runtime, about 11MB...
    curl.exe -sL -o "%TEMP%\pyembed.zip" "https://registry.npmmirror.com/-/binary/python/3.11.9/python-3.11.9-embed-amd64.zip" || goto :neterr
    powershell -NoProfile -Command "Expand-Archive -Force '%TEMP%\pyembed.zip' 'runtime\pyembed'" || goto :neterr
    powershell -NoProfile -Command "$f='runtime\pyembed\python311._pth'; (Get-Content $f) -replace '#import site','import site' | Set-Content $f" || goto :neterr
    curl.exe -sL -o "%TEMP%\get-pip.py" "https://bootstrap.pypa.io/get-pip.py" || goto :neterr
    "runtime\pyembed\python.exe" "%TEMP%\get-pip.py" -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet || goto :neterr
    "runtime\pyembed\python.exe" -m pip install -r app\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet || goto :neterr
    mkdir "runtime\venv\Scripts" 2>nul
    xcopy /e /i /q /y "runtime\pyembed" "runtime\venv\Scripts" >nul
    set "PY=runtime\venv\Scripts\python.exe"
)

"!PY!" app\setup.py
if errorlevel 1 (
    echo.
    echo [error] Setup failed. Please screenshot the messages above.
    pause >nul
)
exit /b

:neterr
echo [error] Network download failed. Check your connection and run this script again.
pause >nul
