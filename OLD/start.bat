@echo off
chcp 65001 >nul
title KRONOS Terminal

echo.
echo  ==========================================
echo   KRONOS Terminal Launcher
echo  ==========================================
echo   [1] Chinese
echo   [2] English
echo   [3] Japanese
echo  ==========================================
echo.
set /p LANG_CHOICE=Select [1/2/3]: 

:: Map choice to lang code
set LANG=zh
if "%LANG_CHOICE%"=="2" set LANG=en
if "%LANG_CHOICE%"=="3" set LANG=ja

:: Write lang file so the browser can read it
echo %LANG%> "%~dp0.kronos_lang"

set CONDA_BAT=D:\conda\Scripts\activate.bat
if not exist "%CONDA_BAT%" (
    echo [ERROR] Cannot find activate.bat at: %CONDA_BAT%
    pause & exit /b 1
)

echo [OK] conda: %CONDA_BAT%  lang: %LANG%

call "%CONDA_BAT%" kronos
if errorlevel 1 (
    echo [ERROR] Failed to activate env 'kronos'.
    pause & exit /b 1
)

cd /d "%~dp0"

python -c "import flask" 2>nul || pip install flask flask-cors -q
python -c "import yfinance" 2>nul || pip install yfinance -q

echo [1/3] Starting Kronos API server (port 5000)...
start "KRONOS_API" cmd /c "call "%CONDA_BAT%" kronos && cd /d "%~dp0" && python kronos_server.py && pause"
timeout /t 5 /nobreak >nul

echo [2/3] Starting HTTP server (port 8080)...
start "KRONOS_HTTP" cmd /c "cd /d "%~dp0" && python -m http.server 8080"
timeout /t 2 /nobreak >nul

echo [3/3] Opening browser...
start "" http://localhost:8080/kronos_terminal.html?lang=%LANG%

echo.
echo  ==========================================
echo   OK! KRONOS Terminal is running.
echo   API : http://localhost:5000
echo   Web : http://localhost:8080/kronos_terminal.html
echo   Press any key to STOP all servers.
echo  ==========================================
pause >nul

taskkill /fi "WINDOWTITLE eq KRONOS_API" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq KRONOS_HTTP" /f >nul 2>&1
echo Stopped.
timeout /t 1 /nobreak >nul
