import os
import warnings
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn,
)
from rich.text import Text

console = Console()
_progress = None


# ---------- setup ----------
def quiet_noise():
    """Call at the very top of main.py, BEFORE importing mediapipe / torch stuff."""
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom")


def _ensure():
    """One shared live display. Started on first use, so every file works standalone."""
    global _progress
    if _progress is None:
        _progress = Progress(
            TextColumn("{task.description:<26}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        _progress.start()
    return _progress


def stop():
    global _progress
    if _progress is not None:
        _progress.stop()
        _progress = None


@contextmanager
def pipeline():
    """Optional wrapper for main.py. Guarantees the display is closed, even on errors."""
    _ensure()
    try:
        yield
    finally:
        stop()


# ---------- messages (use these instead of print) ----------
def log(msg, style=""):
    console.print(Text(str(msg), style=style))   # Text() so "[youtube]" isn't read as markup

def info(msg):  log(f"• {msg}", "cyan")
def ok(msg):    log(f"✓ {msg}", "green")
def warn(msg):  log(f"! {msg}", "yellow")
def error(msg): log(f"✗ {msg}", "bold red")


# ---------- bars ----------
class Bar:
    """One line in the display.
    total=number  -> normal bar
    total=None    -> indeterminate (use for ffmpeg, API calls, anything with no progress info)
    """

    def __init__(self, name, total=1.0):
        self._p = _ensure()
        self.total = total
        self.id = self._p.add_task(name, total=total)

    def update(self, completed, total=None):
        kw = {"completed": completed}
        if total:
            self.total = total
            kw["total"] = total
        self._p.update(self.id, **kw)

    def advance(self, n=1):
        self._p.advance(self.id, n)

    def finish(self):
        if self.total is None:
            self.total = 1
        self._p.update(self.id, completed=self.total, total=self.total)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.finish()


def track(items, name, total=None):
    """for seg in track(segments, "audio signals"): ...   (bar updates itself)"""
    if total is None:
        total = len(items) if hasattr(items, "__len__") else None
    bar = Bar(name, total)
    for i, item in enumerate(items, 1):
        yield item
        bar.update(i)
    bar.finish()


# ---------- library adapters ----------
class DiarizationHook:
    """Replacement for pyannote's ProgressHook."""

    def __init__(self):
        self.bars = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for b in self.bars.values():
            b.finish()

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if completed is None:
            completed = total = 1
        if step_name not in self.bars:
            self.bars[step_name] = Bar(step_name, total)
        self.bars[step_name].update(completed, total)


def ytdlp_hook():
    """ydl_opts['progress_hooks'] = [ytdlp_hook()]   (one bar per stream: video, audio)"""
    bars = {}

    def hook(d):
        key = d.get("filename", "")
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if key not in bars:
                bars[key] = Bar(f"download: stream {len(bars) + 1}", total or None)
            bars[key].update(d.get("downloaded_bytes", 0), total)
        elif d["status"] == "finished" and key in bars:
            bars[key].finish()

    return hook