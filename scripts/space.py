"""
vOS-VirtualVoice - 磁碟空間預檢 (Phase 1)

為什麼需要這個
--------------
Phase 1 會**放大**資料量，而且放大倍率不小：遊戲容器內是壓縮音訊
（Vorbis / Opus / HCA / ADPCM），解開後變成 PCM WAV，而我們又統一成
24-bit / 48 kHz / 單聲道。

實測基準：**24-bit 48 kHz 單聲道 = 144 KB/s 約 8.6 MB/分鐘**。

於是一個 800 MB 的 .awb（內含壓縮語音）解包 + 標準化後，很容易變成 8~15 GB。
磁碟寫滿的後果不只是「失敗」——

* unpack 寫到一半失敗 -> 該容器只解出一部分 subsong，而且**沒有明顯錯誤**
* convert 寫到一半失敗 -> 合格區混著完整與截斷的檔案
* 最糟的是：Phase 2 會把截斷的檔案當成正常素材餵進模型

所以要在**開始之前**就算給使用者看，而不是寫爆之後才發現。

（注意：本模組**不做**破壞性操作，只讀取空間資訊與計算預估。）
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

#: 24-bit / 48 kHz / 單聲道 PCM 的位元組率（48000 樣本/秒 x 3 bytes）。
BYTES_PER_SECOND_24BIT_48K_MONO = 48_000 * 3  # = 144000

#: 來源壓縮率假設（來源位元組 -> 解碼後 PCM 位元組的倍率）。
#: 遊戲語音常見 64~128 kbps 的 Vorbis/HCA：144000 B/s 除以 16~8 KB/s = 9~18。
UNPACK_EXPANSION_LOW = 8.0
UNPACK_EXPANSION_HIGH = 18.0

#: 安全邊際：留給作業系統、暫存檔、以及估算誤差。
SAFETY_FACTOR = 1.5
#: 絕對保留量：無論如何不要用到最後這些空間。
MIN_FREE_BYTES = 512 * 1024 * 1024  # 512 MiB


def human_bytes(value: int) -> str:
    """位元組轉人類可讀字串。"""
    step = 1024.0
    out = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if out < step:
            return f"{out:.1f} {unit}"
        out /= step
    return f"{out:.1f} PiB"


@dataclass
class SpaceEstimate:
    """一次磁碟空間預估。"""

    label: str
    low_bytes: int
    high_bytes: int
    free_bytes: int
    target: Path

    @property
    def needed_bytes(self) -> int:
        """含安全邊際的建議保留量（以高估值計算）。"""
        return int(self.high_bytes * SAFETY_FACTOR) + MIN_FREE_BYTES

    @property
    def sufficient(self) -> bool:
        return self.free_bytes >= self.needed_bytes

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.needed_bytes - self.free_bytes)

    def render(self) -> str:
        """人類可讀的預估說明。"""
        lines = [
            f" {self.label}",
            f"   預估產物：{human_bytes(self.low_bytes)} ~ {human_bytes(self.high_bytes)}"
            f"（依來源壓縮率而異）",
            f"   建議保留：{human_bytes(self.needed_bytes)}"
            f"（含 {SAFETY_FACTOR:g}x 安全邊際與 {human_bytes(MIN_FREE_BYTES)} 緩衝）",
            f"   目前可用：{human_bytes(self.free_bytes)}",
        ]
        if self.sufficient:
            lines.append("   [OK] 空間充足")
        else:
            lines.append(f"   [!!] 空間不足，還缺約 {human_bytes(self.shortfall_bytes)}")
        return "\n".join(lines)


def free_bytes(path: Path) -> int:
    """
    回傳 path 所在磁碟的可用位元組數。

    path 不存在時往上找最近的既有祖先 —— 使用者常常在還沒建立的
    輸出目錄上問空間，那不該直接失敗。
    """
    probe = Path(path)
    for _ in range(40):
        try:
            if probe.exists():
                return shutil.disk_usage(str(probe)).free
        except OSError:
            pass
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(str(Path.cwd())).free
    except OSError:
        return 0


def estimate_from_source_bytes(
    source_bytes: int,
    target: Path,
    *,
    label: str = "解包 + 標準化",
    already_pcm: bool = False,
) -> SpaceEstimate:
    """
    依來源總位元組數預估需要的空間。

    already_pcm=True 表示來源已經是未壓縮音訊（.wav 等），
    此時不乘壓縮展開倍率，只考慮位元深度/取樣率的變化。
    """
    if already_pcm:
        low = int(source_bytes * 0.9)
        high = int(source_bytes * 1.2)
    else:
        low = int(source_bytes * UNPACK_EXPANSION_LOW)
        high = int(source_bytes * UNPACK_EXPANSION_HIGH)

    return SpaceEstimate(
        label=label,
        low_bytes=low,
        high_bytes=high,
        free_bytes=free_bytes(target),
        target=Path(target),
    )


def estimate_for_seconds(
    seconds: float, target: Path, *, label: str = "產物"
) -> SpaceEstimate:
    """已知總時長時的直接估算（Phase 1 產物大小非常可預測）。"""
    exact = int(seconds * BYTES_PER_SECOND_24BIT_48K_MONO)
    return SpaceEstimate(
        label=label,
        low_bytes=exact,
        high_bytes=exact,
        free_bytes=free_bytes(target),
        target=Path(target),
    )


__all__ = [
    "BYTES_PER_SECOND_24BIT_48K_MONO",
    "MIN_FREE_BYTES",
    "SAFETY_FACTOR",
    "UNPACK_EXPANSION_HIGH",
    "UNPACK_EXPANSION_LOW",
    "SpaceEstimate",
    "estimate_for_seconds",
    "estimate_from_source_bytes",
    "free_bytes",
    "human_bytes",
]