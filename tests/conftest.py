"""
vOS-VirtualVoice — pytest 全域設定 + 自管暫存目錄 fixture

為什麼不用 pytest 內建的 tmp_path
---------------------------------
pytest 的 tmp_path 會在 basetemp 底下建立一個「symlink farm」結構，
在部分受限環境（Windows 沙箱、公司端點防護、系統 Temp 唯讀）會直接失敗：

    PermissionError: [WinError 5] 存取被拒

而且症狀是所有用到 tmp_path 的測試一起 error，看起來像程式壞了，
排查成本很高。本檔提供的 `tmp_dir` fixture 用 os.makedirs + shutil.rmtree
自管暫存目錄，完全不依賴系統 Temp，也不建立 symlink。

用法（與 tmp_path 完全相同，只是名字不同）：

    def test_something(tmp_dir: Path) -> None:
        (tmp_dir / "a.wav").write_bytes(b"...")

另外也會把 TMPDIR/TEMP/TMP 指向專案內，讓測試啟動的子程序
（例如 phase1_prep.py）不會誤用不可寫的系統 Temp。
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import subprocess

    import numpy as np

#: 環境變數名稱：可覆寫 repo 根目錄（端到端測試的沙箱隔離基礎）
ENV_REPO_ROOT = "VOS_REPO_ROOT"

REPO_ROOT = Path(__file__).resolve().parent.parent
TMP_ROOT = REPO_ROOT / ".pytest_tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# tempfile 在首次使用時讀環境變數，因此必須早於任何 tmp 相關呼叫
_TMP_ENV = TMP_ROOT / "os"
_TMP_ENV.mkdir(parents=True, exist_ok=True)
for _var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_var] = str(_TMP_ENV)

# 讓測試能 import scripts/ 內的模組
_SCRIPTS = str(REPO_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


# ---------------------------------------------------------------------------
# 共用測試素材產生器
# ---------------------------------------------------------------------------
def synth_voice(
    duration: float, sr: int = 48_000, freq: float = 220.0, amp: float = 0.5
) -> "np.ndarray":
    """
    合成一段「像人聲」的測試訊號：基頻 + 幾個諧波 + 淡入淡出。

    為什麼不用純正弦：純正弦的 RMS/Peak 比例不真實，測不出品質關卡的實際行為，
    也測不出重採樣的頻譜變化。多次諧波讓它更接近語音的頻譜形狀。
    """
    import numpy as np

    t = np.arange(int(duration * sr), dtype=np.float64) / sr
    sig = np.zeros_like(t)
    for k, gain in enumerate((1.0, 0.5, 0.33, 0.25, 0.2), start=1):
        sig += gain * np.sin(2 * np.pi * freq * k * t)
    sig /= np.max(np.abs(sig)) or 1.0
    # 5 ms 淡入淡出，避免邊緣爆音（也和真實遊戲語音的處理習慣一致）
    fade = min(int(0.005 * sr), len(sig) // 4)
    if fade > 1:
        sig[:fade] *= np.linspace(0, 1, fade)
        sig[-fade:] *= np.linspace(1, 0, fade)
    return (sig * amp).astype(np.float32)


@pytest.fixture
def tmp_dir() -> Path:
    """
    每個測試一個乾淨的暫存資料夾，結束後自動刪除。

    位置固定在 <repo>/.pytest_tmp/<uuid>/，不碰系統 Temp、不用 symlink。
    """
    path = TMP_ROOT / f"t_{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# 共用：CLI 子程序執行器
# ---------------------------------------------------------------------------
def run_cli(
    args: list[str],
    repo_root: Path,
    *,
    env_extra: dict[str, str] | None = None,
    timeout: int = 600,
) -> "subprocess.CompletedProcess[str]":
    """
    以子程序執行真正的 CLI，並用 VOS_REPO_ROOT 把它的 repo 根目錄指向沙箱。

    沒有這個覆寫，端到端測試就會把測試素材寫進開發者真正的 data/ 裡
    （本專案真的發生過）。所有測試檔共用這一個實作，避免各檔自己寫一份而分歧。
    """
    import subprocess

    env = dict(os.environ)
    env[ENV_REPO_ROOT] = str(repo_root)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "phase1_prep.py"), *args],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


@pytest.fixture
def cli():
    """回傳 run_cli 的 fixture 版本（讓測試不必自己 import）。"""
    return run_cli


# ---------------------------------------------------------------------------
# 共用：假 vgmstream-cli
# ---------------------------------------------------------------------------
def make_fake_exe(dir_path: Path) -> Path:
    """
    建立一個可執行的 vgmstream-cli 替身，回傳其路徑。

    Windows：產生 vgmstream-cli.cmd（Windows 沒有 shebang 機制）
    Unix   ：產生 vgmstream-cli（可執行 shell script）
    """
    fake_script = Path(__file__).resolve().parent / "fake_vgmstream.py"
    dir_path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        exe = dir_path / "vgmstream-cli.cmd"
        exe.write_text(
            f'@echo off\r\n"{sys.executable}" "{fake_script}" %*\r\n', encoding="utf-8"
        )
    else:
        exe = dir_path / "vgmstream-cli"
        exe.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{fake_script}" "$@"\n', encoding="utf-8"
        )
        exe.chmod(0o755)
    return exe


@pytest.fixture
def fake_exe(tmp_dir: Path) -> Path:
    """假 vgmstream-cli 的路徑。"""
    return make_fake_exe(tmp_dir / "bin")

