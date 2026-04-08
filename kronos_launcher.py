# -*- coding: utf-8 -*-
from __future__ import annotations

import configparser
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "KRONOS GUI Launcher"
API_PORT = 5000
HTTP_PORT = 8080


I18N = {
    "zh": {
        "title": "KRONOS GUI 启动器",
        "config_group": "配置",
        "info_group": "信息",
        "log_group": "日志",
        "lang": "语言",
        "env": "Conda 环境",
        "model_dir": "模型目录",
        "browse": "浏览...",
        "save": "保存配置",
        "start": "启动服务",
        "stop": "停止服务",
        "open": "打开浏览器",
        "status": "状态",
        "idle": "未启动",
        "starting": "启动中",
        "running": "运行中",
        "stopped": "已停止",
        "partial": "部分运行",
        "config_saved": "配置已保存",
        "select_dir": "选择 Kronos 模型目录",
        "invalid_model_dir": "模型目录不存在。",
        "server_file_missing": "未找到 kronos_server.py。",
        "html_file_missing": "未找到 kronos_terminal.html。",
        "conda_not_found": "未找到 conda.exe，请确认已安装 Miniconda/Anaconda。",
        "env_check_failed": "Conda 环境检查失败，请确认环境名正确且可用。",
        "dep_checking": "正在检查依赖...",
        "dep_failed": "依赖检查或安装失败，请查看日志。",
        "api_start": "正在启动 API 服务...",
        "http_start": "正在启动 HTTP 服务...",
        "browser_open": "正在打开浏览器...",
        "all_started": "服务已成功启动。",
        "start_failed": "服务未成功启动，请查看日志。",
        "all_stopped": "服务已停止。",
        "not_running": "当前没有运行中的服务。",
        "api_label": "API 地址",
        "web_label": "Web 地址",
        "close_confirm": "检测到服务仍在运行，是否先停止再退出？",
        "windows_only": "当前版本仅支持 Windows。",
        "log_prefix_ok": "[OK]",
        "log_prefix_info": "[INFO]",
        "log_prefix_error": "[ERROR]",
        "health_wait": "等待服务健康检查通过...",
        "api_health_fail": "API 健康检查未通过。",
        "http_health_fail": "Web 页面未可用。",
        "stop_failed": "停止服务时出现异常。",
        "checking_env": "正在检查 Conda 环境...",
        "using_conda": "使用 conda.exe",
        "kronos_lang_written": "已写入 .kronos_lang",
        "installing_missing_dep": "正在安装缺失依赖...",
        "ports_in_use": "检测到目标地址已可访问，本次跳过重复启动。",
        "api_launch": "API 启动命令已发送",
        "http_launch": "HTTP 启动命令已发送",
    },
    "en": {
        "title": "KRONOS GUI Launcher",
        "config_group": "Configuration",
        "info_group": "Info",
        "log_group": "Log",
        "lang": "Language",
        "env": "Conda Environment",
        "model_dir": "Model Directory",
        "browse": "Browse...",
        "save": "Save Config",
        "start": "Start Services",
        "stop": "Stop Services",
        "open": "Open Browser",
        "status": "Status",
        "idle": "Idle",
        "starting": "Starting",
        "running": "Running",
        "stopped": "Stopped",
        "partial": "Partially Running",
        "config_saved": "Configuration saved",
        "select_dir": "Select Kronos model directory",
        "invalid_model_dir": "Model directory does not exist.",
        "server_file_missing": "kronos_server.py was not found.",
        "html_file_missing": "kronos_terminal.html was not found.",
        "conda_not_found": "conda.exe was not found. Please install Miniconda/Anaconda.",
        "env_check_failed": "Conda environment check failed. Please verify the environment name.",
        "dep_checking": "Checking dependencies...",
        "dep_failed": "Dependency check/install failed. See log for details.",
        "api_start": "Starting API server...",
        "http_start": "Starting HTTP server...",
        "browser_open": "Opening browser...",
        "all_started": "Services started successfully.",
        "start_failed": "Services did not start successfully. See log.",
        "all_stopped": "Services stopped.",
        "not_running": "No running services.",
        "api_label": "API URL",
        "web_label": "Web URL",
        "close_confirm": "Services are still running. Stop them before exit?",
        "windows_only": "This version supports Windows only.",
        "log_prefix_ok": "[OK]",
        "log_prefix_info": "[INFO]",
        "log_prefix_error": "[ERROR]",
        "health_wait": "Waiting for service health checks...",
        "api_health_fail": "API health check failed.",
        "http_health_fail": "Web page is not available.",
        "stop_failed": "An error occurred while stopping services.",
        "checking_env": "Checking conda environment...",
        "using_conda": "Using conda.exe",
        "kronos_lang_written": ".kronos_lang written",
        "installing_missing_dep": "Installing missing dependency...",
        "ports_in_use": "Target URLs are already reachable; skipped duplicate launch.",
        "api_launch": "API launch command sent",
        "http_launch": "HTTP launch command sent",
    },
    "ja": {
        "title": "KRONOS GUI ランチャー",
        "config_group": "設定",
        "info_group": "情報",
        "log_group": "ログ",
        "lang": "言語",
        "env": "Conda 環境",
        "model_dir": "モデルディレクトリ",
        "browse": "参照...",
        "save": "設定を保存",
        "start": "起動",
        "stop": "停止",
        "open": "ブラウザを開く",
        "status": "状態",
        "idle": "未起動",
        "starting": "起動中",
        "running": "実行中",
        "stopped": "停止済み",
        "partial": "一部実行中",
        "config_saved": "設定を保存しました",
        "select_dir": "Kronos モデルディレクトリを選択",
        "invalid_model_dir": "モデルディレクトリが存在しません。",
        "server_file_missing": "kronos_server.py が見つかりません。",
        "html_file_missing": "kronos_terminal.html が見つかりません。",
        "conda_not_found": "conda.exe が見つかりません。Miniconda / Anaconda を確認してください。",
        "env_check_failed": "Conda 環境の確認に失敗しました。環境名を確認してください。",
        "dep_checking": "依存関係を確認しています...",
        "dep_failed": "依存関係の確認またはインストールに失敗しました。ログを確認してください。",
        "api_start": "API サービスを起動しています...",
        "http_start": "HTTP サービスを起動しています...",
        "browser_open": "ブラウザを開いています...",
        "all_started": "サービスを起動しました。",
        "start_failed": "サービス起動に失敗しました。ログを確認してください。",
        "all_stopped": "サービスを停止しました。",
        "not_running": "実行中のサービスはありません。",
        "api_label": "API URL",
        "web_label": "Web URL",
        "close_confirm": "サービスが実行中です。停止してから終了しますか？",
        "windows_only": "このバージョンは Windows のみ対応です。",
        "log_prefix_ok": "[OK]",
        "log_prefix_info": "[INFO]",
        "log_prefix_error": "[ERROR]",
        "health_wait": "サービスのヘルスチェックを待っています...",
        "api_health_fail": "API ヘルスチェックに失敗しました。",
        "http_health_fail": "Web ページにアクセスできません。",
        "stop_failed": "サービス停止時にエラーが発生しました。",
        "checking_env": "Conda 環境を確認しています...",
        "using_conda": "使用する conda.exe",
        "kronos_lang_written": ".kronos_lang を書き込みました",
        "installing_missing_dep": "不足している依存関係をインストールしています...",
        "ports_in_use": "対象 URL はすでに到達可能です。重複起動をスキップしました。",
        "api_launch": "API 起動コマンドを送信しました",
        "http_launch": "HTTP 起動コマンドを送信しました",
    },
}


def is_windows() -> bool:
    return os.name == "nt"


def url_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def check_api_and_web(api_port: int, web_port: int) -> tuple[bool, bool]:
    # 同时尝试 IPv4 (127.0.0.1) 和 IPv6 ([::1])，任意一个通即视为成功
    api_ok = (
        url_ok(f"http://127.0.0.1:{api_port}/api/health", 3.0) or
        url_ok(f"http://[::1]:{api_port}/api/health", 3.0)
    )
    web_ok = (
        url_ok(f"http://127.0.0.1:{web_port}/kronos_terminal.html", 3.0) or
        url_ok(f"http://[::1]:{web_port}/kronos_terminal.html", 3.0)
    )
    return api_ok, web_ok


class KronosLauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.base_dir = Path(__file__).resolve().parent
        self.cfg_path = self.base_dir / "kronos_config.ini"
        self.server_py = self.base_dir / "kronos_server.py"
        self.web_html = self.base_dir / "kronos_terminal.html"
        self.lang_file = self.base_dir / ".kronos_lang"

        self.api_proc: subprocess.Popen | None = None
        self.http_proc: subprocess.Popen | None = None
        self.conda_exe: str | None = None

        self.lang_var = tk.StringVar(value="zh")
        self.env_var = tk.StringVar(value="kronos")
        self.model_dir_var = tk.StringVar(value=str(self.base_dir))
        self.status_var = tk.StringVar(value="")
        self.api_url_var = tk.StringVar(value=f"http://localhost:{API_PORT}")
        self.web_url_var = tk.StringVar(value=f"http://localhost:{HTTP_PORT}/kronos_terminal.html")

        self.load_config()
        self.build_ui()
        self.apply_language()

    def tr(self, key: str) -> str:
        return I18N.get(self.lang_var.get(), I18N["zh"]).get(key, key)

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()

    def build_ui(self) -> None:
        self.root.title(APP_TITLE)
        self.root.geometry("780x600")
        self.root.minsize(760, 560)

        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        self.config_group = ttk.LabelFrame(main, padding=12)
        self.config_group.pack(fill="x")

        self.lang_label = ttk.Label(self.config_group)
        self.lang_label.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)

        self.lang_combo = ttk.Combobox(
            self.config_group,
            textvariable=self.lang_var,
            state="readonly",
            values=["zh", "en", "ja"],
            width=10,
        )
        self.lang_combo.grid(row=0, column=1, sticky="w", pady=6)
        self.lang_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_language())

        self.env_label = ttk.Label(self.config_group)
        self.env_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)

        self.env_entry = ttk.Entry(self.config_group, textvariable=self.env_var, width=40)
        self.env_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=6)

        self.model_label = ttk.Label(self.config_group)
        self.model_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)

        self.model_entry = ttk.Entry(self.config_group, textvariable=self.model_dir_var, width=58)
        self.model_entry.grid(row=2, column=1, sticky="ew", pady=6)

        self.browse_btn = ttk.Button(self.config_group, command=self.browse_model_dir)
        self.browse_btn.grid(row=2, column=2, sticky="e", padx=(8, 0), pady=6)

        self.config_group.columnconfigure(1, weight=1)

        self.button_row = ttk.Frame(main, padding=(0, 12, 0, 12))
        self.button_row.pack(fill="x")

        self.save_btn = ttk.Button(self.button_row, command=self.save_config_ui)
        self.save_btn.pack(side="left")

        self.start_btn = ttk.Button(self.button_row, command=self.start_services)
        self.start_btn.pack(side="left", padx=(8, 0))

        self.stop_btn = ttk.Button(self.button_row, command=self.stop_services)
        self.stop_btn.pack(side="left", padx=(8, 0))

        self.open_btn = ttk.Button(self.button_row, command=self.open_browser)
        self.open_btn.pack(side="left", padx=(8, 0))

        self.info_group = ttk.LabelFrame(main, padding=12)
        self.info_group.pack(fill="x")

        self.status_label = ttk.Label(self.info_group)
        self.status_label.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)

        self.status_value = ttk.Label(self.info_group, textvariable=self.status_var)
        self.status_value.grid(row=0, column=1, sticky="w", pady=4)

        self.api_label = ttk.Label(self.info_group)
        self.api_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)

        self.api_entry = ttk.Entry(self.info_group, textvariable=self.api_url_var, state="readonly")
        self.api_entry.grid(row=1, column=1, sticky="ew", pady=4)

        self.web_label = ttk.Label(self.info_group)
        self.web_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)

        self.web_entry = ttk.Entry(self.info_group, textvariable=self.web_url_var, state="readonly")
        self.web_entry.grid(row=2, column=1, sticky="ew", pady=4)

        self.info_group.columnconfigure(1, weight=1)

        self.log_group = ttk.LabelFrame(main, padding=12)
        self.log_group.pack(fill="both", expand=True)

        self.log_text = tk.Text(self.log_group, wrap="word", height=16)
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.configure(state="disabled")

        scrollbar = ttk.Scrollbar(self.log_group, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def apply_language(self) -> None:
        self.root.title(self.tr("title"))
        self.config_group.configure(text=self.tr("config_group"))
        self.info_group.configure(text=self.tr("info_group"))
        self.log_group.configure(text=self.tr("log_group"))

        self.lang_label.configure(text=self.tr("lang"))
        self.env_label.configure(text=self.tr("env"))
        self.model_label.configure(text=self.tr("model_dir"))
        self.browse_btn.configure(text=self.tr("browse"))
        self.save_btn.configure(text=self.tr("save"))
        self.start_btn.configure(text=self.tr("start"))
        self.stop_btn.configure(text=self.tr("stop"))
        self.open_btn.configure(text=self.tr("open"))
        self.status_label.configure(text=self.tr("status"))
        self.api_label.configure(text=self.tr("api_label"))
        self.web_label.configure(text=self.tr("web_label"))

        self.refresh_status_label()

    def refresh_status_label(self) -> None:
        api_ok, http_ok = check_api_and_web(API_PORT, HTTP_PORT)
        if api_ok and http_ok:
            self.status_var.set(self.tr("running"))
        elif api_ok or http_ok:
            self.status_var.set(self.tr("partial"))
        elif self.api_proc is None and self.http_proc is None:
            self.status_var.set(self.tr("idle"))
        else:
            self.status_var.set(self.tr("stopped"))

    def browse_model_dir(self) -> None:
        selected = filedialog.askdirectory(title=self.tr("select_dir"))
        if selected:
            self.model_dir_var.set(selected)

    def load_config(self) -> None:
        if not self.cfg_path.exists():
            return
        parser = configparser.ConfigParser()
        content = self.cfg_path.read_text(encoding="utf-8")
        if "[main]" not in content:
            content = "[main]\n" + content
        parser.read_string(content)
        section = parser["main"]
        self.env_var.set(section.get("CONDA_ENV", self.env_var.get()))
        self.model_dir_var.set(section.get("MODEL_DIR", self.model_dir_var.get()))
        self.lang_var.set(section.get("LANG", self.lang_var.get()))

    def save_config(self) -> None:
        text = (
            f"CONDA_ENV={self.env_var.get().strip()}\n"
            f"MODEL_DIR={self.model_dir_var.get().strip()}\n"
            f"LANG={self.lang_var.get().strip()}\n"
        )
        self.cfg_path.write_text(text, encoding="utf-8")

    def save_config_ui(self) -> None:
        self.save_config()
        self.log(f"{self.tr('log_prefix_ok')} {self.tr('config_saved')}: {self.cfg_path}")
        messagebox.showinfo(APP_TITLE, self.tr("config_saved"))

    def find_conda_exe(self) -> str | None:
        candidates = [
            Path.home() / "Miniconda3" / "Scripts" / "conda.exe",
            Path.home() / "Anaconda3" / "Scripts" / "conda.exe",
            Path(r"C:\Miniconda3\Scripts\conda.exe"),
            Path(r"C:\Anaconda3\Scripts\conda.exe"),
            Path(r"D:\conda\Scripts\conda.exe"),
            Path(r"D:\Miniconda3\Scripts\conda.exe"),
            Path(r"D:\Anaconda3\Scripts\conda.exe"),
        ]
        for path in candidates:
            if path.exists():
                return str(path)

        try:
            result = subprocess.run(
                ["where", "conda"],
                shell=True,
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    p = Path(line.strip())
                    if p.name.lower() == "conda.exe" and p.exists():
                        return str(p)
                    if p.name.lower() in {"conda.bat", "conda"}:
                        alt = p.with_name("conda.exe")
                        if alt.exists():
                            return str(alt)
        except Exception:
            pass
        return None

    def run_conda_capture(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        assert self.conda_exe is not None
        return subprocess.run(
            [self.conda_exe] + args,
            cwd=str(self.base_dir),
            text=True,
            capture_output=True,
        )

    def ensure_dependencies(self) -> bool:
        assert self.conda_exe is not None
        env_name = self.env_var.get().strip()
        checks = [
            (
                ["run", "-n", env_name, "python", "-c", "import flask"],
                ["run", "-n", env_name, "pip", "install", "flask", "flask-cors", "-q"],
            ),
            (
                ["run", "-n", env_name, "python", "-c", "import yfinance"],
                ["run", "-n", env_name, "pip", "install", "yfinance", "-q"],
            ),
        ]

        self.log(f"{self.tr('log_prefix_info')} {self.tr('dep_checking')}")
        for check_args, install_args in checks:
            check = self.run_conda_capture(check_args)
            if check.returncode == 0:
                continue

            if check.stdout.strip():
                self.log(check.stdout.strip())
            if check.stderr.strip():
                self.log(check.stderr.strip())

            self.log(f"{self.tr('log_prefix_info')} {self.tr('installing_missing_dep')}")
            install = self.run_conda_capture(install_args)
            if install.stdout.strip():
                self.log(install.stdout.strip())
            if install.stderr.strip():
                self.log(install.stderr.strip())
            if install.returncode != 0:
                return False
        return True

    def update_status_from_health(self) -> None:
        api_ok, http_ok = check_api_and_web(API_PORT, HTTP_PORT)
        if api_ok and http_ok:
            self.status_var.set(self.tr("running"))
        elif api_ok or http_ok:
            self.status_var.set(self.tr("partial"))
        else:
            self.status_var.set(self.tr("stopped"))

    def open_browser(self) -> None:
        url = f"http://localhost:{HTTP_PORT}/kronos_terminal.html?lang={self.lang_var.get()}"
        webbrowser.open(url)
        self.log(f"{self.tr('log_prefix_info')} {self.tr('browser_open')} {url}")

    def cleanup_dead_process_refs(self) -> None:
        if self.api_proc is not None and self.api_proc.poll() is not None:
            self.api_proc = None
        if self.http_proc is not None and self.http_proc.poll() is not None:
            self.http_proc = None

    def wait_for_health(self, timeout_seconds: int = 60) -> tuple[bool, bool]:
        self.log(f"{self.tr('log_prefix_info')} {self.tr('health_wait')}")

        # 给进程预热时间，避免还没开始监听就检测
        time.sleep(3)
        self.root.update()

        deadline = time.time() + timeout_seconds
        api_ok = False
        http_ok = False

        while time.time() < deadline:
            api_ok, http_ok = check_api_and_web(API_PORT, HTTP_PORT)
            if api_ok and http_ok:
                return True, True
            elapsed = int(deadline - time.time())
            self.log(
                f"{self.tr('log_prefix_info')} "
                f"api={'✓' if api_ok else '✗'}  "
                f"web={'✓' if http_ok else '✗'}  "
                f"({elapsed}s left)"
            )
            self.root.after(2000)
            self.root.update()

        return api_ok, http_ok

    def start_services(self) -> None:
        if not is_windows():
            messagebox.showerror(APP_TITLE, self.tr("windows_only"))
            return

        self.cleanup_dead_process_refs()
        self.save_config()

        model_dir = Path(self.model_dir_var.get().strip())
        env_name = self.env_var.get().strip()

        if not model_dir.exists():
            messagebox.showerror(APP_TITLE, self.tr("invalid_model_dir"))
            return

        if not self.server_py.exists():
            messagebox.showerror(APP_TITLE, self.tr("server_file_missing"))
            return

        if not self.web_html.exists():
            self.log(f"{self.tr('log_prefix_info')} {self.tr('html_file_missing')}")

        api_ready, web_ready = check_api_and_web(API_PORT, HTTP_PORT)
        if api_ready or web_ready:
            self.log(f"{self.tr('log_prefix_info')} {self.tr('ports_in_use')}")
            self.update_status_from_health()
            return

        self.conda_exe = self.find_conda_exe()
        if not self.conda_exe:
            messagebox.showerror(APP_TITLE, self.tr("conda_not_found"))
            self.log(f"{self.tr('log_prefix_error')} {self.tr('conda_not_found')}")
            return

        self.status_var.set(self.tr("starting"))
        self.log(f"{self.tr('log_prefix_info')} {self.tr('using_conda')}: {self.conda_exe}")
        self.log(f"{self.tr('log_prefix_info')} env = {env_name}")
        self.log(f"{self.tr('log_prefix_info')} model_dir = {model_dir}")
        self.log(f"{self.tr('log_prefix_info')} {self.tr('checking_env')}")

        env_check = self.run_conda_capture(["run", "-n", env_name, "python", "-V"])
        if env_check.stdout.strip():
            self.log(env_check.stdout.strip())
        if env_check.stderr.strip():
            self.log(env_check.stderr.strip())
        if env_check.returncode != 0:
            self.status_var.set(self.tr("idle"))
            messagebox.showerror(APP_TITLE, self.tr("env_check_failed"))
            return

        if not self.ensure_dependencies():
            self.status_var.set(self.tr("idle"))
            messagebox.showerror(APP_TITLE, self.tr("dep_failed"))
            return

        try:
            self.lang_file.write_text(self.lang_var.get(), encoding="utf-8")
            self.log(f"{self.tr('log_prefix_ok')} {self.tr('kronos_lang_written')}: {self.lang_file}")
        except Exception as exc:
            self.log(f"{self.tr('log_prefix_error')} Failed to write language file: {exc}")

        creationflags = subprocess.CREATE_NEW_CONSOLE

        try:
            self.log(f"{self.tr('log_prefix_info')} {self.tr('api_start')}")
            self.api_proc = subprocess.Popen(
                [
                    self.conda_exe,
                    "run",
                    "-n",
                    env_name,
                    "python",
                    str(self.server_py),
                    "--model-dir",
                    str(model_dir),
                    "--port",
                    str(API_PORT),
                    "--host",
                    "0.0.0.0",   # 同时监听 IPv4 和 IPv6
                ],
                cwd=str(self.base_dir),
                creationflags=creationflags,
            )
            self.log(f"{self.tr('log_prefix_info')} {self.tr('api_launch')}")

            self.log(f"{self.tr('log_prefix_info')} {self.tr('http_start')}")
            self.http_proc = subprocess.Popen(
                [
                    self.conda_exe,
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "-m",
                    "http.server",
                    str(HTTP_PORT),
                ],
                cwd=str(self.base_dir),
                creationflags=creationflags,
            )
            self.log(f"{self.tr('log_prefix_info')} {self.tr('http_launch')}")

            api_ok, http_ok = self.wait_for_health(60)

            if api_ok and http_ok:
                self.status_var.set(self.tr("running"))
                self.log(f"{self.tr('log_prefix_ok')} {self.tr('all_started')}")
                self.open_browser()
            else:
                if not api_ok:
                    self.log(f"{self.tr('log_prefix_error')} {self.tr('api_health_fail')}")
                if not http_ok:
                    self.log(f"{self.tr('log_prefix_error')} {self.tr('http_health_fail')}")
                self.update_status_from_health()
                messagebox.showerror(APP_TITLE, self.tr("start_failed"))

        except Exception as exc:
            self.status_var.set(self.tr("idle"))
            self.log(f"{self.tr('log_prefix_error')} {exc}")
            messagebox.showerror(APP_TITLE, f"{self.tr('start_failed')}\n{exc}")

    def stop_services(self) -> None:
        self.cleanup_dead_process_refs()

        api_ok, http_ok = check_api_and_web(API_PORT, HTTP_PORT)
        nothing_running = (
            self.api_proc is None and
            self.http_proc is None and
            not api_ok and
            not http_ok
        )
        if nothing_running:
            self.log(f"{self.tr('log_prefix_info')} {self.tr('not_running')}")
            self.status_var.set(self.tr("stopped"))
            return

        try:
            for proc in (self.api_proc, self.http_proc):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()

            self.api_proc = None
            self.http_proc = None

            for _ in range(10):
                api_ok, http_ok = check_api_and_web(API_PORT, HTTP_PORT)
                if not api_ok and not http_ok:
                    break
                time.sleep(0.5)

            self.status_var.set(self.tr("stopped"))
            self.log(f"{self.tr('log_prefix_ok')} {self.tr('all_stopped')}")
        except Exception as exc:
            self.log(f"{self.tr('log_prefix_error')} {self.tr('stop_failed')} {exc}")
            messagebox.showerror(APP_TITLE, f"{self.tr('stop_failed')}\n{exc}")

    def on_close(self) -> None:
        self.cleanup_dead_process_refs()
        if self.api_proc is not None or self.http_proc is not None:
            if messagebox.askyesno(APP_TITLE, self.tr("close_confirm")):
                self.stop_services()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()

    try:
        from ctypes import windll  # type: ignore
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    try:
        style = ttk.Style(root)
        style.theme_use("vista")
    except Exception:
        pass

    app = KronosLauncherApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
