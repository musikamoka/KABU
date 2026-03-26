@echo off
chcp 65001 >nul
title KRONOS Terminal Launcher

:: ── Language selection ─────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════╗
echo  ║        KRONOS Terminal Launcher      ║
echo  ╠══════════════════════════════════════╣
echo  ║  [1] 中文                            ║
echo  ║  [2] English                         ║
echo  ║  [3] 日本語                          ║
echo  ╚══════════════════════════════════════╝
echo.
set /p LANG_CHOICE= Select language / 言語を選んでください / 选择语言 [1/2/3]: 

if "%LANG_CHOICE%"=="1" (
    set LANG=zh
    set MSG_CONDA=正在激活 conda 环境...
    set MSG_SERVER=启动 Kronos API 服务器...
    set MSG_HTTP=启动 HTTP 服务器...
    set MSG_BROWSER=打开浏览器...
    set MSG_READY=启动完成！浏览器将自动打开
    set MSG_STOP=按任意键关闭所有服务...
) else if "%LANG_CHOICE%"=="3" (
    set LANG=ja
    set MSG_CONDA=conda 環境を起動中...
    set MSG_SERVER=Kronos API サーバー起動中...
    set MSG_HTTP=HTTP サーバー起動中...
    set MSG_BROWSER=ブラウザを開いています...
    set MSG_READY=起動完了！ブラウザが自動で開きます
    set MSG_STOP=任意のキーで全サービスを停止...
) else (
    set LANG=en
    set MSG_CONDA=Activating conda environment...
    set MSG_SERVER=Starting Kronos API server...
    set MSG_HTTP=Starting HTTP server...
    set MSG_BROWSER=Opening browser...
    set MSG_READY=Ready! Browser will open automatically.
    set MSG_STOP=Press any key to stop all services...
)

:: Write selected language to temp file so HTML can read it
echo %LANG% > "%~dp0.lang"

echo.
echo  [1/4] %MSG_CONDA%
call conda activate kronos 2>nul
if errorlevel 1 (
    echo  ERROR: conda environment 'kronos' not found.
    echo  Run: conda create -n kronos python=3.10
    pause & exit /b 1
)

echo  [2/4] %MSG_SERVER%
start "Kronos API Server" /min cmd /c "conda activate kronos && cd /d %~dp0 && python kronos_server.py"

:: Wait for server to start
timeout /t 3 /nobreak >nul

echo  [3/4] %MSG_HTTP%
start "Kronos HTTP Server" /min cmd /c "conda activate kronos && cd /d %~dp0 && python -m http.server 8080"

:: Wait for HTTP server
timeout /t 2 /nobreak >nul

echo  [4/4] %MSG_BROWSER%
start "" "http://localhost:8080/kronos_terminal.html"

echo.
echo  ✅ %MSG_READY%
echo  ─────────────────────────────────────
echo  API Server : http://localhost:5000
echo  Frontend   : http://localhost:8080/kronos_terminal.html
echo  ─────────────────────────────────────
echo.
echo  %MSG_STOP%
pause >nul

:: Cleanup
taskkill /f /fi "WINDOWTITLE eq Kronos API Server*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Kronos HTTP Server*" >nul 2>&1
echo  Stopped.
