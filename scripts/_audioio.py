"""
vOS-VirtualVoice — 音訊 I/O 抽象層 (Phase 1)

三個後端，統一介面：
  soundfile  : libsndfile。WAV/FLAC/OGG/OPUS/AIFF，純 Python 依賴，最快。
  ffmpeg     : 萬能後備。任何 libsndfile 吃不下、但 ffmpeg 吃得下的格式。
  vgmstream  : 遊戲容器（Wwise .wem / CriWare .awb / FMOD .fsb）。

設計重點
--------
* 一律回傳 float32 numpy 陣列 + 實際 sample rate，**不做隱式重採樣**。
  重採樣只在 Phase 1 轉碼階段明確發生一次，避免多次插值累積失真。
* 後端不可用時不拋例外打死整個批次，而是回 (None, reason)，讓呼叫端記進 manifest。
* 所有外部程序呼叫都用 sys.executable / shutil.which 解析，Windows/Mac/Linux 一致。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import find_external_bin

# ---------------------------------------------------------------------------
# 後端能力探測（匯入時只探測一次，之後用快取）
# ---------------------------------------------------------------------------
try:
    import soundfile as sf  # type: ignore

    HAS_SOUNDFILE = True
    try:
        _SF_VERSION = sf.__version__
    except AttributeError:  # pragma: no cover
        _SF_VERSION = "unknown"
except Exception:  # pragma: no cover - 環境沒裝 soundfile 也要能跑 doctor
    sf = None  # type: ignore
    HAS_SOUNDFILE = False
    _SF_VERSION = "missing"

try:
    import soxr  # type: ignore

    HAS_SOXR = True
except Exception:  # pragma: no cover
    soxr = None  # type: ignore
    HAS_SOXR = False

try:
    from scipy import signal as _scipy_signal  # type: ignore

    HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_signal = None  # type: ignore
    HAS_SCIPY = False


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    version: str
    note: str = ""


def probe_backends() -> list[BackendStatus]:
    """回傳所有後端的可用狀態（給 doctor 用）。"""
    statuses: list[BackendStatus] = []

    statuses.append(
        BackendStatus(
            "soundfile",
            HAS_SOUNDFILE,
            _SF_VERSION,
            "libsndfile " + (sf.__libsndfile_version__ if HAS_SOUNDFILE else "-"),
        )
    )

    ffmpeg = find_external_bin("ffmpeg")
    statuses.append(
        BackendStatus(
            "ffmpeg",
            ffmpeg is not None,
            ffmpeg or "-",
            "未找到。建議安裝以支援 m4a/aac/wma 等格式（見 docs/PHASE1-INGEST.md）",
        )
    )

    vgm = find_external_bin("vgmstream-cli")
    statuses.append(
        BackendStatus(
            "vgmstream",
            vgm is not None,
            vgm or "-",
            "未找到。處理 .wem/.awb/.fsb 遊戲容器時必需",
        )
    )

    statuses.append(
        BackendStatus(
            "soxr (重採樣)",
            HAS_SOXR,
            getattr(soxr, "__version__", "?") if HAS_SOXR else "-",
            "高品質重採樣；缺席時自動退回 scipy polyphase",
        )
    )
    statuses.append(
        BackendStatus(
            "scipy (後備重採樣)",
            HAS_SCIPY,
            getattr(_scipy_signal, "__name__", "?") if HAS_SCIPY else "-",
        )
    )
    return statuses


# ---------------------------------------------------------------------------
# 外部程序工具
# ---------------------------------------------------------------------------
def _no_window_kwargs() -> dict[str, object]:
    """Windows 上避免彈出黑色 console 視窗。"""
    if sys.platform.startswith("win"):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": si, "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def run_tool(
    args: list[str], *, timeout: int = 600, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """
    執行外部命令列工具，統一的錯誤處理。

    跨平台注意：
      * 不使用 shell=True（避免 Windows cmd 與 Unix sh 的引號差異）
      * 一律用 list 傳參，路徑含空白也安全
      * text=True + encoding='utf-8' + errors='replace' 防止非 UTF-8 輸出炸掉
    """
    try:
        proc = subprocess.run(  # noqa: S603 - 參數由本模組控制，非使用者直輸
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            **_no_window_kwargs(),  # type: ignore[arg-type]
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到執行檔：{args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"命令逾時 ({timeout}s)：{' '.join(args[:3])} ...") from exc

    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"命令失敗 (exit {proc.returncode})：{' '.join(args[:4])} ...\n" + "\n".join(tail)
        )
    return proc


# ---------------------------------------------------------------------------
# 解碼
# ---------------------------------------------------------------------------
@dataclass
class DecodeResult:
    samples: np.ndarray | None  # shape (frames,) 單聲道 或 (frames, channels)
    sample_rate: int
    backend: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.samples is not None


def _decode_soundfile(path: Path) -> DecodeResult:
    if not HAS_SOUNDFILE:
        return DecodeResult(None, 0, "soundfile", "未安裝 soundfile")
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # libsndfile 會丟各種例外
        return DecodeResult(None, 0, "soundfile", f"{type(exc).__name__}: {exc}")
    return DecodeResult(np.asarray(data, dtype=np.float32), int(sr), "soundfile")


def _decode_ffmpeg(path: Path, sample_rate: int | None = None) -> DecodeResult:
    """
    用 ffmpeg 解碼成 f32le 原始位元組，再用 numpy 讀進來。

    選 f32le 而非 wav：省一次暫存檔寫入，且無 24-bit 對齊的麻煩。
    若指定 sample_rate（例如 48000），ffmpeg 會一併完成重採樣。
    """
    ffmpeg = find_external_bin("ffmpeg")
    if not ffmpeg:
        return DecodeResult(None, 0, "ffmpeg", "未找到 ffmpeg")

    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path)]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    # 保留原聲道數，單/多聲道的處理交給呼叫端
    args += ["-f", "f32le", "-acodec", "pcm_f32le", "-"]
    try:
        proc = subprocess.run(  # noqa: S603
            args,
            capture_output=True,
            timeout=600,
            check=False,
            **_no_window_kwargs(),  # type: ignore[arg-type]
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return DecodeResult(None, 0, "ffmpeg", str(exc))

    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        return DecodeResult(None, 0, "ffmpeg", err[-1] if err else "ffmpeg 無輸出")

    samples = np.frombuffer(proc.stdout, dtype="<f4")
    sr = sample_rate or _ffmpeg_sample_rate(path)
    if not sr:
        return DecodeResult(None, 0, "ffmpeg", "無法取得取樣率")
    return DecodeResult(samples.astype(np.float32, copy=False), int(sr), "ffmpeg")


def _ffmpeg_sample_rate(path: Path) -> int:
    """用 ffprobe 問取樣率；沒有 ffprobe 就回 0。"""
    ffprobe = find_external_bin("ffprobe")
    if not ffprobe:
        return 0
    proc = run_tool(
        [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        check=False,
        timeout=60,
    )
    try:
        return int((proc.stdout or "0").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def _decode_vgmstream(path: Path, sample_rate: int | None = None) -> DecodeResult:
    """
    用 vgmstream-cli 解成 WAV（暫存檔），再交給 soundfile 讀。

    vgmstream-cli 的關鍵參數：
      -o <file>   輸出檔
      -w          強制輸出 WAV（而非其他格式）
      --loop-count 1  不重複（遊戲語音有 loop 標記時很重要）
    """
    vgm = find_external_bin("vgmstream-cli")
    if not vgm:
        return DecodeResult(None, 0, "vgmstream", "未找到 vgmstream-cli")

    with tempfile.TemporaryDirectory(prefix="vos_vgm_") as td:
        out = Path(td) / "out.wav"
        args = [vgm, "-w", "-o", str(out), "-i", "--loop-count", "1", str(path)]
        proc = run_tool(args, check=False, timeout=600)
        if not out.is_file() or out.stat().st_size < 64:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            return DecodeResult(None, 0, "vgmstream", " | ".join(tail) or "vgmstream 未產生輸出")

        if sample_rate:
            # vgmstream 無法直接指定輸出取樣率，交給 ffmpeg；沒有 ffmpeg 就先讀原樣
            if find_external_bin("ffmpeg"):
                return _decode_ffmpeg(out, sample_rate)
        res = _decode_soundfile(out)
        res.backend = "vgmstream"
        return res


def decode(
    path: str | Path,
    *,
    preferred: str | None = None,
    sample_rate: int | None = None,
) -> DecodeResult:
    """
    解碼任意音訊檔為 float32 陣列。

    Args:
        path:        來源檔
        preferred:   指定後端（"soundfile" / "ffmpeg" / "vgmstream"）；None = 自動依序嘗試
        sample_rate: 指定輸出取樣率（僅 ffmpeg / vgmstream 路徑有效；soundfile 維持原樣）

    Returns:
        DecodeResult；失敗時 samples=None 且 error 說明原因。
    """
    p = Path(path)
    if not p.is_file():
        return DecodeResult(None, 0, "none", f"檔案不存在：{p}")

    order = [preferred] if preferred else ["soundfile", "ffmpeg", "vgmstream"]
    errors: list[str] = []
    for backend in order:
        if backend == "soundfile":
            res = _decode_soundfile(p)
        elif backend == "ffmpeg":
            res = _decode_ffmpeg(p, sample_rate)
        elif backend == "vgmstream":
            res = _decode_vgmstream(p, sample_rate)
        else:
            continue
        if res.ok:
            return res
        errors.append(f"{backend}: {res.error}")

    return DecodeResult(None, 0, "none", " ; ".join(errors) or "無可用後端")


# ---------------------------------------------------------------------------
# 重採樣 / 聲道處理 / 寫檔
# ---------------------------------------------------------------------------
def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    高品質重採樣。優先 soxr（SoX VHQ 等級），否則 scipy polyphase。

    48 kHz 是 GPT-SoVITS / SoVITS 系列的原生訓練取樣率：
      * 來源 < 48 kHz → 升採樣（不增加資訊，但統一格式，訓練端不必再處理）
      * 來源 > 48 kHz → 降採樣（同時抗鋸齒，實務上更乾淨）
    """
    if orig_sr == target_sr:
        return samples

    x = samples.astype(np.float32, copy=False)
    if HAS_SOXR:
        y = soxr.resample(x, orig_sr, target_sr, quality="VHQ")
    elif HAS_SCIPY:
        from math import gcd

        g = gcd(int(orig_sr), int(target_sr))
        y = _scipy_signal.resample_poly(
            x, target_sr // g, orig_sr // g, window=("kaiser", 8.6)
        )
    else:
        raise RuntimeError(
            "重採樣需要 soxr 或 scipy，兩者皆未安裝：pip install soxr scipy"
        )
    return np.asarray(y, dtype=np.float32)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """
    轉單聲道（等功率平均，而非只取左聲道）。

    GPT-SoVITS 訓練只需單聲道；保留雙聲道會讓資料量翻倍且無增益。
    """
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        if samples.shape[1] == 1:
            return samples[:, 0]
        return samples.mean(axis=1, dtype=np.float32)
    raise ValueError(f"不支援的陣列維度：{samples.ndim}")


def measure(samples: np.ndarray) -> tuple[float, float]:
    """
    回傳 (peak_dbfs, rms_dbfs)。全靜音時回 (-inf, -inf)，
    呼叫端需自行處理 -inf（JSON 不支援，寫入前請用 _safe_db）。
    """
    if samples.size == 0:
        return float("-inf"), float("-inf")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak_db = 20.0 * np.log10(peak) if peak > 0 else float("-inf")
    rms_db = 20.0 * np.log10(rms) if rms > 0 else float("-inf")
    return float(peak_db), float(rms_db)


def safe_db(value: float, digits: int = 2) -> str:
    """把 -inf 轉成可讀字串，方便寫進 CSV/JSON。"""
    if value is None or value != value or value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    return f"{value:.{digits}f}"


def write_wav(
    path: str | Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    subtype: str = "PCM_24",
) -> None:
    """
    寫出 WAV。使用 soundfile（libsndfile），確保 24-bit 在各平台一致。

    PCM_24 是 GPT-SoVITS 前處理的本命格式：
      * 24-bit 保留足夠動態範圍，避免 16-bit 量化在低電平細節上的損失
      * 不建議 32-bit float 中繼檔，Phase 2 的 UVR5 / loudnorm 對整數 PCM 較友善
    """
    if not HAS_SOUNDFILE:
        raise RuntimeError("寫出 WAV 需要 soundfile：pip install soundfile")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), samples, int(sample_rate), subtype=subtype, format="WAV")


__all__ = [
    "BackendStatus",
    "DecodeResult",
    "HAS_SCIPY",
    "HAS_SOUNDFILE",
    "HAS_SOXR",
    "decode",
    "measure",
    "probe_backends",
    "resample",
    "run_tool",
    "safe_db",
    "to_mono",
    "write_wav",
]
