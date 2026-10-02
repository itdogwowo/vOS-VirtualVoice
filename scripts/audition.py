"""
vOS-VirtualVoice — 人工試聽清單產生器 (Phase 1, 驗收步驟)

為什麼需要這個工具
------------------
Phase 1 有一項**無法自動化**的驗收步驟：人工試聽。自動化能抓時長、電平、
削波，但「這句台詞有沒有感情」「這句話是不是被截斷了」「這個檔案裡是不是
有兩個人在講話」只有耳朵知道 —— 而這正是整個專案的核心價值。

問題是：合格素材動輒數千個檔案，你不可能一個一個開。

這個模組把你的耳朵變成可抽樣的：
  1. 依群組（角色）分層抽樣，確保每個角色都聽到
  2. 產生「連續試聽帶」（單一 WAV，每段之間插入靜音與序號提示）
  3. 可選響度對齊，讓大聲與小聲的檔案在同一次播放中都聽得清楚
  4. 輸出對照清單（第 N 段 = 哪個檔案），聽到問題可以直接回查

設計原則
--------
* **可重現**：預設固定亂數種子，同一批素材永遠抽出同一組樣本。
  這樣你「修完再聽一次」才有意義。
* **不修改原檔**：試聽帶寫在 _reports/ 底下，合格區完全不動。
* **單一檔案**：輸出成一個 WAV，用任何播放器都能順聽，不需要播放清單。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from _audioio import HAS_SOUNDFILE, measure, safe_db

if HAS_SOUNDFILE:
    import soundfile as sf


#: 每段之間的靜音秒數（給你時間反應／記筆記）
DEFAULT_GAP_SECONDS = 0.9
#: 響度對齊的目標 RMS（dBFS）。-20 是「聽得清楚但不刺耳」的實務值。
DEFAULT_TARGET_RMS_DBFS = -20.0
#: 第一階段（軟）增益上限：一般電平差異在這個範圍內解決。
SOFT_GAIN_LIMIT_DB = 12.0
#: 第二階段（硬）增益上限：給遊戲刻意做小聲的耳語／內心戲台詞。
#: 這類素材對情感模型很有價值，不該因為「太小聲聽不到」而被排除在試聽之外，
#: 但也不能無上限放大（那只是在放大底噪）。
LOUD_GAIN_LIMIT_DB = 30.0


@dataclass
class Sample:
    """一個被抽中的試聽樣本。"""

    index: int          # 在試聽帶中的序號（1-based）
    path: Path
    group: str
    duration: float = 0.0
    peak_dbfs: float = float("nan")
    rms_dbfs: float = float("nan")
    offset_sec: float = 0.0   # 在試聽帶中的起始秒數
    error: str = ""


@dataclass
class AuditionPlan:
    """一次試聽的完整計畫（抽樣結果 + 統計）。"""

    samples: list[Sample] = field(default_factory=list)
    total_seconds: float = 0.0
    reel_path: Path | None = None
    reel_written: bool = False

    @property
    def count(self) -> int:
        return len(self.samples)

    def group_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.samples:
            counts[s.group] = counts.get(s.group, 0) + 1
        return counts

    def write_list(self, path: Path) -> None:
        """寫出對照清單（Markdown 表格），讓你能從序號回查檔案。"""
        lines = [
            "# Phase 1 人工試聽清單",
            "",
            f"- 樣本數：{self.count}",
            f"- 試聽帶總長：{self.total_seconds / 60:.1f} 分鐘",
            f"- 試聽帶：`{self.reel_path.name if self.reel_path else '（未產生）'}`",
            "",
            "> 聽到問題時，記下「第幾段」，用下表回查是哪個檔案。",
            "> 建議逐項確認：① 有無底噪／電流聲 ② 有無被截頭截尾 ③ 是否同一人 "
            "④ 情感種類是否多樣",
            "",
            "| 段 | 群組 | 起始(秒) | 長度(秒) | Peak | RMS | 檔案 |",
            "|---:|------|--------:|--------:|-----:|----:|------|",
        ]
        for s in self.samples:
            name = s.path.name if not s.error else f"⚠ {s.path.name}（{s.error}）"
            lines.append(
                f"| {s.index} | {s.group} | {s.offset_sec:.1f} | {s.duration:.2f} | "
                f"{safe_db(s.peak_dbfs, 1)} | {safe_db(s.rms_dbfs, 1)} | `{name}` |"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 抽樣
# ---------------------------------------------------------------------------
def list_candidates(root: Path) -> dict[str, list[Path]]:
    """
    列出 root 底下（含子資料夾）所有 .wav，依「相對於 root 的第一層資料夾」分組。

    這正好對應 convert 的 group_by="folder" 產物結構：
    02_wav24_48k/<角色名>/xxx.wav → 每個角色一組。單一角色時只有 default 一組。
    """
    groups: dict[str, list[Path]] = {}
    if not root.is_dir():
        return groups
    for path in sorted(root.rglob("*.wav")):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        group = rel.parts[0] if len(rel.parts) > 1 else "default"
        groups.setdefault(group, []).append(path)
    return groups


def plan_samples(
    groups: dict[str, list[Path]],
    *,
    per_group: int = 5,
    seed: int = 20240601,
    include_groups: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """
    分層抽樣：每個群組各抽 per_group 個（不足則全取）。

    為什麼要分層而不是全域隨機：全域隨機在「素材量懸殊」時會幾乎只抽到大群組
    （例如 凛 3000 條、旁白 40 條），小群組的問題永遠聽不出來 ——
    而那通常正是最需要檢查的地方。
    """
    picked: list[tuple[str, Path]] = []
    for group in sorted(groups):
        if include_groups and group not in include_groups:
            continue
        files = groups[group]
        if not files:
            continue
        rng = random.Random(f"{seed}:{group}")  # 每組各自的種子 → 單組變動不影響他組
        n = min(per_group, len(files))
        for path in rng.sample(files, n):
            picked.append((group, path))
    return picked


# ---------------------------------------------------------------------------
# 試聽帶
# ---------------------------------------------------------------------------
def _load_mono(path: Path) -> tuple[np.ndarray | None, int, str]:
    """讀成單聲道 float32；失敗回 (None, 0, 原因)。"""
    if not HAS_SOUNDFILE:
        return None, 0, "未安裝 soundfile"
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        return None, 0, f"{type(exc).__name__}: {exc}"
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1, dtype=np.float32)
    return arr, int(sr), ""


def _align_rms(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """
    把一段音訊的 RMS 對齊到目標值。這是**試聽專用**的處理，不會寫回合格區。

    為什麼需要：原始素材的電平差異可能達 30 dB。若不對齊，你聽小聲的檔案時
    要把音量轉大，下一段大聲的就會爆掉耳朵 —— 於是你就放棄試聽了。

    兩階段增益：
      1. 軟上限（+12 dB）：處理一般的電平差異
      2. 硬上限（+30 dB）：處理遊戲刻意做小聲的耳語／內心戲台詞。
         這些素材情感價值高，不該被排除在試聽之外，但也不能無上限放大。
      3. 負增益不設下限（太大聲就該降下來），最後再做防削波。
    """
    _, rms = measure(samples)
    if not np.isfinite(rms) or rms == float("-inf"):
        return samples
    gain_db = target_dbfs - rms
    if gain_db > 0:
        gain_db = min(gain_db, LOUD_GAIN_LIMIT_DB)
    gain = 10.0 ** (gain_db / 20.0)
    out = samples * gain
    # 對齊後仍要防削波
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def build_reel(
    picked: list[tuple[str, Path]],
    out_path: Path,
    *,
    sample_rate: int = 48_000,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    align_rms: bool = True,
    target_dbfs: float = DEFAULT_TARGET_RMS_DBFS,
    resampler=None,
) -> AuditionPlan:
    """
    把抽中的樣本接成單一「試聽帶」WAV。

    Args:
        resampler: 可選的 callable(samples, src_sr, dst_sr)，預設用 _audioio.resample。
                   抽出來是為了讓測試不需要真的重採樣。

    Returns:
        AuditionPlan（含每段的起始秒數，供對照清單使用）。
    """
    plan = AuditionPlan()
    if not picked:
        return plan

    gap = np.zeros(int(gap_seconds * sample_rate), dtype=np.float32)
    chunks: list[np.ndarray] = []
    cursor = 0.0

    for idx, (group, path) in enumerate(picked, 1):
        sample = Sample(index=idx, path=path, group=group)
        audio, sr, err = _load_mono(path)
        if audio is None:
            sample.error = err
            plan.samples.append(sample)
            continue

        if sr != sample_rate:
            if resampler is not None:
                audio = resampler(audio, sr, sample_rate)
            else:
                from _audioio import resample as _default_resample

                audio = _default_resample(audio, sr, sample_rate)

        peak, rms = measure(audio)
        sample.peak_dbfs, sample.rms_dbfs = peak, rms
        sample.duration = len(audio) / sample_rate
        sample.offset_sec = cursor

        if align_rms:
            audio = _align_rms(audio, target_dbfs)

        chunks.append(audio)
        chunks.append(gap)
        cursor += sample.duration + gap_seconds
        plan.samples.append(sample)

    if not chunks:
        return plan

    reel = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    plan.total_seconds = len(reel) / sample_rate
    plan.reel_path = out_path

    if HAS_SOUNDFILE:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 試聽帶用 16-bit 就好：它的用途是「聽」，不是訓練素材
        sf.write(str(out_path), reel, sample_rate, subtype="PCM_16", format="WAV")
        plan.reel_written = True

    return plan


def filesystem_safe(name: str) -> str:
    """清掉檔名中跨平台不安全的字元（Windows 的 : * ? " < > | 等）。"""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return cleaned[:80] or "reel"


__all__ = [
    "DEFAULT_GAP_SECONDS",
    "DEFAULT_TARGET_RMS_DBFS",
    "LOUD_GAIN_LIMIT_DB",
    "SOFT_GAIN_LIMIT_DB",
    "AuditionPlan",
    "Sample",
    "build_reel",
    "filesystem_safe",
    "list_candidates",
    "plan_samples",
]
