"""
video-quick-eval GUI — Gemini 风格
"""

import os
import sys
import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
os.chdir(_HERE)

# ── 常量 ────────────────────────────────────────────────────────────────────
CONFIG_FILE = _HERE / "config.json"
PROMPTS_DIR = _HERE / "prompts"
OUTPUT_DIR  = _HERE / "output"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

# Gemini 配色
G_BG       = "#131314"   # 主背景（近纯黑）
G_SURFACE  = "#1e1e1e"   # 卡片/面板背景
G_INPUT    = "#282a2c"   # 输入框背景
G_DIVIDER  = "#2e2e2e"   # 分割线
G_ACCENT   = "#8ab4f8"   # Google 蓝
G_ACCENT_H = "#aecbfa"   # 悬停蓝
G_TEXT     = "#e3e3e3"   # 主文字
G_SUBTEXT  = "#9aa0a6"   # 次要文字
G_SUCCESS  = "#81c995"   # 成功绿
G_ERROR    = "#f28b82"   # 错误红（仅错误用）


# ── 工具函数 ─────────────────────────────────────────────────────────────────
def load_config() -> dict:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def list_prompts():
    prompts = []
    if PROMPTS_DIR.exists():
        for p in sorted(PROMPTS_DIR.glob("*.md")):
            if p.stat().st_size > 0:
                prompts.append(p.stem)
    return prompts

def open_folder(path: str):
    if os.path.exists(path):
        subprocess.Popen(f'explorer "{path}"')

# ── TTK 样式 ─────────────────────────────────────────────────────────────────
def apply_styles():
    style = ttk.Style()
    style.theme_use("default")
    style.configure("TCombobox",
        fieldbackground=G_INPUT, background=G_INPUT,
        foreground=G_TEXT, selectbackground=G_INPUT,
        selectforeground=G_TEXT, borderwidth=0, arrowcolor=G_SUBTEXT
    )
    style.map("TCombobox",
        fieldbackground=[("readonly", G_INPUT)],
        foreground=[("readonly", G_TEXT)]
    )
    style.configure("Horizontal.TProgressbar",
        troughcolor=G_DIVIDER, background=G_ACCENT,
        borderwidth=0, thickness=3
    )


# ── 主界面 ───────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("video-quick-eval")
        self.geometry("960x680")
        self.minsize(820, 560)
        self.configure(bg=G_BG)
        apply_styles()

        self._running = False
        self._cfg = load_config()
        self._llm_chars = 0

        self._build_ui()
        self._log("就绪。填写视频 URL 或选择本地文件，选择提示词后点击「处理」。")

    # ── UI 构建 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # 顶栏
        nav = tk.Frame(self, bg=G_SURFACE, height=52)
        nav.grid(row=0, column=0, sticky="ew")
        tk.Label(nav, text="  ✦  video-quick-eval",
            bg=G_SURFACE, fg=G_TEXT,
            font=("Segoe UI", 13, "bold"), pady=14
        ).pack(side="left")
        tk.Frame(self, bg=G_DIVIDER, height=1).grid(row=0, column=0, sticky="sew")

        # 主体
        body = tk.Frame(self, bg=G_BG)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0, minsize=270)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_log_panel(body)

        # 底部进度条
        self._progress = ttk.Progressbar(
            self, mode="indeterminate",
            style="Horizontal.TProgressbar"
        )
        self._progress.grid(row=2, column=0, sticky="ew")

    # ── 侧边栏 ───────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=G_SURFACE, width=270)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.columnconfigure(0, weight=1)
        sb.grid_propagate(False)
        tk.Frame(parent, bg=G_DIVIDER, width=1).grid(row=0, column=0, sticky="nse")

        row = 0

        def section(text, r, top=20):
            tk.Label(sb, text=text, bg=G_SURFACE, fg=G_SUBTEXT,
                     font=("Segoe UI", 8, "bold")
                     ).grid(row=r, column=0, sticky="w", pady=(top, 4), padx=20)

        section("输入源", row); row += 1

        tk.Label(sb, text="在线视频 URL（B站 / YouTube）",
                 bg=G_SURFACE, fg=G_SUBTEXT, font=("Segoe UI", 9)
                 ).grid(row=row, column=0, sticky="w", padx=20); row += 1

        self._url_var = tk.StringVar()
        self._mk_entry(sb, self._url_var, row); row += 1

        tk.Label(sb, text="本地视频文件",
                 bg=G_SURFACE, fg=G_SUBTEXT, font=("Segoe UI", 9)
                 ).grid(row=row, column=0, sticky="w", padx=20, pady=(10, 0)); row += 1

        lf = tk.Frame(sb, bg=G_SURFACE)
        lf.grid(row=row, column=0, sticky="ew", padx=20); row += 1
        lf.columnconfigure(0, weight=1)
        self._local_var = tk.StringVar()
        tk.Entry(lf, textvariable=self._local_var,
                 bg=G_INPUT, fg=G_TEXT, insertbackground=G_TEXT,
                 relief="flat", font=("Segoe UI", 9), bd=8, width=16
                 ).grid(row=0, column=0, sticky="ew", ipady=3)
        tk.Button(lf, text="浏览", command=self._browse_file,
                  bg=G_INPUT, fg=G_TEXT, relief="flat",
                  font=("Segoe UI", 9), padx=10, pady=3,
                  cursor="hand2", activebackground=G_DIVIDER
                  ).grid(row=0, column=1, padx=(6, 0))

        section("提示词", row); row += 1
        self._prompt_vars = {}
        for name in list_prompts():
            var = tk.BooleanVar(value=(name == "summary"))
            tk.Checkbutton(
                sb, text=name, variable=var,
                bg=G_SURFACE, fg=G_TEXT,
                selectcolor=G_INPUT, activebackground=G_SURFACE,
                activeforeground=G_TEXT, font=("Segoe UI", 10),
                anchor="w", highlightthickness=0, bd=0
            ).grid(row=row, column=0, sticky="w", padx=20); row += 1
            self._prompt_vars[name] = var

        section("Whisper 模型", row); row += 1
        cfg_size = self._cfg.get("transcribe", {}).get("model_size", "base")
        self._model_var = tk.StringVar(value=cfg_size)
        ttk.Combobox(sb, textvariable=self._model_var,
                     values=WHISPER_MODELS, state="readonly",
                     font=("Segoe UI", 10), width=14
                     ).grid(row=row, column=0, sticky="w", padx=20, pady=(0, 2)); row += 1
        tk.Label(sb, text="tiny 快/低精  ·  base 推荐  ·  small 高精",
                 bg=G_SURFACE, fg=G_SUBTEXT, font=("Segoe UI", 8)
                 ).grid(row=row, column=0, sticky="w", padx=20); row += 1

        self._no_llm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            sb, text="仅转写，不调用 LLM",
            variable=self._no_llm_var,
            bg=G_SURFACE, fg=G_SUBTEXT, selectcolor=G_INPUT,
            activebackground=G_SURFACE, font=("Segoe UI", 9),
            anchor="w", highlightthickness=0, bd=0
        ).grid(row=row, column=0, sticky="w", padx=20, pady=(10, 0)); row += 1

        # 按钮区
        bf = tk.Frame(sb, bg=G_SURFACE)
        bf.grid(row=row, column=0, sticky="ew", padx=20, pady=(18, 0))
        bf.columnconfigure(0, weight=1); row += 1

        self._run_btn = tk.Button(
            bf, text="处理", bg=G_ACCENT, fg="#0d1117",
            relief="flat", font=("Segoe UI", 11, "bold"),
            pady=9, cursor="hand2",
            activebackground=G_ACCENT_H, activeforeground="#0d1117",
            command=self._start
        )
        self._run_btn.grid(row=0, column=0, sticky="ew")

        tk.Button(
            bf, text="输出目录",
            bg=G_INPUT, fg=G_TEXT, relief="flat",
            font=("Segoe UI", 10), pady=7, cursor="hand2",
            activebackground=G_DIVIDER, activeforeground=G_TEXT,
            command=lambda: open_folder(str(OUTPUT_DIR))
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        tk.Button(
            sb, text="⚙  配置 API Key / 模型",
            bg=G_SURFACE, fg=G_SUBTEXT, relief="flat",
            font=("Segoe UI", 9), cursor="hand2",
            activebackground=G_SURFACE,
            command=self._open_settings
        ).grid(row=row, column=0, sticky="w", padx=18, pady=(8, 0)); row += 1

    # ── 日志面板 ──────────────────────────────────────────────────────────────
    def _build_log_panel(self, parent):
        panel = tk.Frame(parent, bg=G_BG)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        hdr = tk.Frame(panel, bg=G_BG)
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 6))

        tk.Label(hdr, text="运行日志", bg=G_BG, fg=G_TEXT,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(hdr, text="清空", bg=G_BG, fg=G_SUBTEXT,
                  relief="flat", font=("Segoe UI", 9), cursor="hand2",
                  activebackground=G_BG, command=self._clear_log
                  ).pack(side="right")

        # LLM 流式进度标签
        self._llm_progress_var = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._llm_progress_var,
                 bg=G_BG, fg=G_ACCENT, font=("Segoe UI", 9)
                 ).pack(side="right", padx=12)

        self._log_box = scrolledtext.ScrolledText(
            panel,
            bg=G_INPUT, fg=G_TEXT,
            font=("Consolas", 10),
            relief="flat", padx=12, pady=10,
            state="disabled", wrap="word",
            insertbackground=G_TEXT,
        )
        self._log_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))
        self._log_box.tag_config("normal",  foreground=G_TEXT)
        self._log_box.tag_config("success", foreground=G_SUCCESS)
        self._log_box.tag_config("error",   foreground=G_ERROR)

    # ── 控件工厂 ──────────────────────────────────────────────────────────────
    def _mk_entry(self, parent, var, row):
        e = tk.Entry(parent, textvariable=var,
                     bg=G_INPUT, fg=G_TEXT, insertbackground=G_TEXT,
                     relief="flat", font=("Segoe UI", 10), bd=8)
        e.grid(row=row, column=0, sticky="ew", padx=20, ipady=4)
        return e

    # ── 日志 ──────────────────────────────────────────────────────────────────
    def _log(self, msg: str, level: str = "normal"):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}\n"
        self._log_box.configure(state="normal")
        self._log_box.insert("end", line, level)
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _clear_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
        self._llm_progress_var.set("")

    # ── 浏览文件 ──────────────────────────────────────────────────────────────
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.webm *.m4v"),
                ("全部", "*.*")
            ]
        )
        if path:
            self._local_var.set(path)
            self._url_var.set("")

    # ── 开始处理 ──────────────────────────────────────────────────────────────
    def _start(self):
        if self._running:
            return

        url   = self._url_var.get().strip()
        local = self._local_var.get().strip()
        target = local if local else url
        if not target:
            messagebox.showwarning("提示", "请输入视频 URL 或选择本地文件")
            return

        selected_prompts = [k for k, v in self._prompt_vars.items() if v.get()]
        no_llm = self._no_llm_var.get()
        if not no_llm and not selected_prompts:
            if not messagebox.askyesno("确认", "未选择任何提示词，将仅保存原始转写文本，继续？"):
                return

        self._running = True
        self._llm_chars = 0
        self._run_btn.configure(state="disabled", text="处理中…")
        self._progress.start(10)
        self._clear_log()

        threading.Thread(
            target=self._worker,
            args=(target, self._model_var.get(), selected_prompts, no_llm),
            daemon=True
        ).start()

    # ── 后台线程 ──────────────────────────────────────────────────────────────
    def _worker(self, target, model_size, prompt_names, no_llm):
        import io

        # 只有真正的报错才标红
        ERROR_KEYWORDS = (
            "[ERROR]", "ERROR:", "调用失败", "下载失败",
            "提取失败", "转写失败", "❌", "Not Found",
            "404", "403", "Exception", "Traceback"
        )
        SUCCESS_KEYWORDS = ("完成", "✅", "成功", "已保存")

        class LogRedirect(io.TextIOBase):
            def __init__(self, cb, force_err=False):
                self._cb = cb
                self._force_err = force_err
            def write(self, s):
                stripped = s.strip()
                if stripped:
                    if self._force_err:
                        level = "error"
                    elif any(k in stripped for k in ERROR_KEYWORDS):
                        level = "error"
                    elif any(k in stripped for k in SUCCESS_KEYWORDS):
                        level = "success"
                    else:
                        level = "normal"
                    self._cb(stripped, level)
                return len(s)
            def flush(self): pass

        old_out, old_err = sys.stdout, sys.stderr
        cb = lambda m, l: self.after(0, self._log, m, l)
        sys.stdout = LogRedirect(cb)
        sys.stderr = LogRedirect(cb, force_err=False)  # stderr 也走关键词判断，不全标红

        try:
            import transcribe as tc

            if not no_llm:
                def _stream_cb(chars: int, chunk: str):
                    self.after(0, self._llm_progress_var.set,
                               f"LLM 生成中…  已输出 {chars} 字符")
                tc.set_llm_stream_callback(_stream_cb)
            else:
                tc.clear_llm_stream_callback()

            result = tc.process_video(
                video_url=target,
                model_size=model_size,
                enable_llm_optimization=not no_llm,
                prompt_names=prompt_names if not no_llm else []
            )
            tc.clear_llm_stream_callback()

            if result.get("success"):
                self.after(0, self._llm_progress_var.set, "")
                self.after(0, self._log,
                           f"全部完成！输出目录：{OUTPUT_DIR}", "success")
                self.after(0, self._on_done, True)
            else:
                err = result.get("error", "未知错误")
                self.after(0, self._log, f"处理失败：{err}", "error")
                self.after(0, self._on_done, False)

        except Exception as e:
            self.after(0, self._log, f"异常：{e}", "error")
            self.after(0, self._on_done, False)
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def _on_done(self, success: bool):
        self._running = False
        self._progress.stop()
        self._llm_progress_var.set("")
        self._run_btn.configure(state="normal", text="处理")

    # ── 设置弹窗 ──────────────────────────────────────────────────────────────
    def _open_settings(self):
        win = tk.Toplevel(self)
        win.title("配置")
        win.geometry("500x280")
        win.configure(bg=G_SURFACE)
        win.resizable(False, False)
        win.grab_set()

        cfg = load_config()
        llm = cfg.get("llm", {})
        fields = [
            ("API Key",  "api_key",  llm.get("api_key", ""),  True),
            ("Base URL", "base_url", llm.get("base_url", ""), False),
            ("模型名称", "model",    llm.get("model", ""),    False),
        ]
        entries = {}

        for i, (label, key, value, secret) in enumerate(fields):
            tk.Label(win, text=label, bg=G_SURFACE, fg=G_TEXT,
                     font=("Segoe UI", 10)
                     ).grid(row=i, column=0, sticky="w", padx=20, pady=10)
            var = tk.StringVar(value=value)
            tk.Entry(
                win, textvariable=var,
                bg=G_INPUT, fg=G_TEXT, insertbackground=G_TEXT,
                relief="flat", font=("Segoe UI", 10), width=36, bd=8,
                show="●" if secret else ""
            ).grid(row=i, column=1, padx=(0, 20), pady=10, ipady=5)
            entries[key] = var

        def _save():
            cfg["llm"] = cfg.get("llm", {})
            for key, var in entries.items():
                cfg["llm"][key] = var.get().strip()
            save_config(cfg)
            self._cfg = cfg
            self._log("配置已保存", "success")
            win.destroy()

        tk.Button(
            win, text="保存",
            bg=G_ACCENT, fg="#0d1117",
            relief="flat", font=("Segoe UI", 11, "bold"),
            pady=8, padx=30, cursor="hand2",
            activebackground=G_ACCENT_H, activeforeground="#0d1117",
            command=_save
        ).grid(row=len(fields), column=0, columnspan=2, pady=(6, 16))


# ── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
