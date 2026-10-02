#!/usr/bin/env python3
"""
假的 vgmstream-cli —— 測試專用替身（不是給正式流程用的）。

為什麼需要它
------------
`unpack` 是 Phase 1 唯一依賴外部二進位檔的環節，也是最貴的環節。
沒有這個替身，我們就只能「希望」命令列參數是對的 —— 而那正是本專案
曾經踩過的坑（`-o` 少了 subsong 萬用字元，導致多 subsong 互相覆蓋）。

這個替身模仿 vgmstream-cli 的關鍵介面（取自官方 doc/USAGE.md）：

    vgmstream-cli -m <file>                            只印資訊
    vgmstream-cli -i -o <out> <file>                   解出單一 subsong
    vgmstream-cli -i -o <out> -S 0 <file>              解出所有 subsong
    vgmstream-cli -i -o "<name>#?s.wav" -S 0 <file>    每個 subsong 各一個檔

它以檔名的 `fake_subsongs=N` 決定要產生幾個 subsong，
並在輸出檔開頭寫入 `SUBSONG <n>` 以便測試驗證內容對應關係。

用法：把本檔複製成 `<某目錄>/vgmstream-cli`（或 Windows 的 .cmd 包裝），
      再把該目錄放進 PATH 或設 unpack.vgmstream_bin 指向它。
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

SUBSONG_RE = re.compile(r"fake_subsongs=(\d+)")
#: 檔名裡的 `fake_dur=<秒>`：讓替身模擬不同長度的 subsong。
#: 用途：測試「解包產物必須有長有短，才能驗證時長篩選真的有作用」。
#:
#: ⚠ 數字的字元集**必須含小數點**。用 `[0-9.]+` 會貪婪地吃進後面的點，
#: 用 `[0-9]+` 則會把 `2.5` 截成 `2`、把 `0.004` 截成 `0` ——
#: 這個錯誤會讓「耳語測試」拿到 amp=0（全靜音）、「削波測試」拿到邊緣值，
#: 測試照樣綠燈但完全沒測到想測的東西。
NUMBER = r"([0-9]*\.?[0-9]+)"
DURATION_RE = re.compile(rf"fake_dur={NUMBER}")
#: 檔名裡的 `fake_amp=<振幅>`：模擬小聲（耳語）或過大（削波）的素材。
AMP_RE = re.compile(rf"fake_amp={NUMBER}")
#: 檔名裡出現 `fake_clip` 時，振幅**不做歸一化**，讓它真的被 int16 削平。
#: 用途：測試品質關卡能不能抓到「波形被削平」的素材。
#: 單純把 amp 調大沒有用 —— 歸一化會把波形縮回 1.0 以下，你得到的是
#: 「大聲但完全乾淨」的方波，peak 恰好貼在門檻內側，反而測不到削波偵測。
CLIP_RE = re.compile(r"fake_clip")


def count_subsongs(path: Path) -> int:
    """從檔名讀出要模擬的 subsong 數量（預設 1）。"""
    match = SUBSONG_RE.search(path.name)
    return max(1, int(match.group(1))) if match else 1


def _float_from_name(pattern: re.Pattern[str], name: str, fallback: float) -> float:
    match = pattern.search(name)
    if not match:
        return fallback
    try:
        return float(match.group(1))
    except ValueError:
        return fallback


def synth_marker(
    marker: str,
    seconds: float,
    amp: float,
    sample_rate: int = 22050,
    *,
    normalize: bool = True,
) -> bytes:
    """
    合成一段「有聲音」的 16-bit PCM 資料，長度由 seconds 決定。

    為什麼不能像第一版一樣寫死 0.05 秒：那樣解包產物永遠會被時長篩選淘汰，
    就測不到「解包 → 篩選 → 試聽 → 報告」這條完整鏈路。

    ⚠ 縮放是必需的：float → int16 的 astype 是**截斷**（0.9 → 0），
    所以一定要先乘上 32767。少了這一步，產出會是「格式完全正確的靜音」——
    測試會一路綠燈，但整條生產線其實在處理無聲檔案。

    Args:
        normalize: True（預設）時把波形歸一化到 amp，得到「大聲但乾淨」的音訊。
                   False 時直接讓波形爆掉，用於測試削波偵測。
    """
    frames = max(1, int(seconds * sample_rate))
    # 用 marker 的雜湊決定基頻，讓不同 subsong 的波形不同（可驗證沒有互相覆蓋）
    freq = 180.0 + (abs(hash(marker)) % 12) * 20.0
    t = np.arange(frames, dtype=np.float64) / sample_rate
    wave = np.sin(2 * np.pi * freq * t)

    peak = float(np.max(np.abs(wave))) or 1.0
    if normalize:
        # 先歸一化再乘 amp：確保真正的峰值等於 amp
        wave = wave / peak * amp
    else:
        # 不歸一化 → 波形超出 [-1, 1]，後面的 clip 會真的削平它
        wave = wave * amp

    # 5 ms 淡入淡出，避免邊緣爆音
    fade = min(int(0.005 * sample_rate), frames // 4)
    if fade > 1:
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
    ints = np.round(np.clip(wave, -1.0, 1.0) * 32767.0)
    return ints.astype("<i2").tobytes()


def write_wav(
    path: Path,
    marker: str,
    seconds: float = 0.05,
    amp: float = 0.5,
    *,
    normalize: bool = True,
) -> None:
    """
    寫一個合法的 WAV。內容依 marker 而異，長度與振幅可控，
    讓測試能驗證「哪個 subsong 寫到哪個檔」以及後續的時長篩選。
    """
    sample_rate = 22050
    payload = synth_marker(marker, seconds, amp, sample_rate, normalize=normalize)

    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(payload))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )


def main(argv: list[str]) -> int:
    info_only = False
    all_subsongs = False
    out_pattern: str | None = None
    inputs: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-m":
            info_only = True
        elif arg == "-i":
            pass  # 「不要循環」：對這個替身沒有差別
        elif arg == "-S":
            all_subsongs = True
            # `-S 0` 的 0 是範圍參數，要吃掉
            if i + 1 < len(argv) and argv[i + 1].isdigit():
                i += 1
        elif arg == "-o":
            i += 1
            if i < len(argv):
                out_pattern = argv[i]
        elif arg in ("-s", "-l", "-f", "-d", "-F"):
            i += 1  # 這些帶一個參數，吃掉
        elif arg.startswith("-"):
            pass  # 未知旗標：忽略，讓測試能塞額外參數
        else:
            inputs.append(arg)
        i += 1

    if not inputs:
        print("fake-vgmstream: no input file", file=sys.stderr)
        return 1

    src_path = Path(inputs[0])
    if not src_path.is_file() or src_path.stat().st_size == 0:
        # 模擬真實 vgmstream 遇到無法解析的檔案：非零退出 + 錯誤訊息
        print(f"failed to open {inputs[0]}", file=sys.stderr)
        return 1

    total = count_subsongs(src_path)
    # 讓不同 subsong 有不同長度，才能驗證「時長篩選真的在運作」
    base_dur = _float_from_name(DURATION_RE, src_path.name, 0.05)
    amp = _float_from_name(AMP_RE, src_path.name, 0.5)
    # fake_clip → 不歸一化，讓波形真的被削平（用於測試削波偵測）
    normalize = CLIP_RE.search(src_path.name) is None

    if info_only:
        print(f"fake vgmstream-cli (test double)")
        print(f"input: {inputs[0]}")
        print(f"encoding: fake-pcm16")
        print(f"channels: 1")
        print(f"sample rate: 22050")
        # 真實 vgmstream 一律會回報 subsong 數量（含只有 1 個的情況），
        # 這裡必須照做 —— 否則測試會走進與生產環境不同的分支。
        print(f"subsongs: {total}")
        return 0

    # 真實 vgmstream 的預設輸出檔名：沒有 -S 時是 `?f.wav`，
    # 有 -S 時是 `?f#?s.wav`。這裡照做，測試才會走生產環境的分支。
    if out_pattern is None:
        out_pattern = f"{inputs[0]}.wav" if not all_subsongs else f"{inputs[0]}#?s.wav"

    if all_subsongs and "?s" in out_pattern:
        # 正確用法：每個 subsong 各寫一個檔
        for n in range(1, total + 1):
            resolved = out_pattern.replace("?s", str(n)).replace("?f", Path(inputs[0]).name)
            resolved = re.sub(r"\?0(\d+)s", lambda m: str(n).zfill(int(m.group(1))), resolved)
            # 第 n 個 subsong 比基準長度多 (n-1)*0.2 秒，讓長度有分布
            write_wav(
                Path(resolved),
                f"SUBSONG {n} of {total}",
                seconds=base_dur + (n - 1) * 0.2,
                amp=amp,
                normalize=normalize,
            )
        print(f"fake-vgmstream: wrote {total} subsongs")
        return 0

    if all_subsongs and "?s" not in out_pattern:
        # 錯誤用法（真實 vgmstream 會全部覆蓋同一檔）——替身刻意模擬這個行為，
        # 讓「萬用字元必須存在」這件事有回歸測試守著
        for n in range(1, total + 1):
            write_wav(Path(out_pattern), f"SUBSONG {n} of {total}", seconds=base_dur, amp=amp, normalize=normalize)
        print(
            f"fake-vgmstream: WARNING subsongs overwrite the same file (missing ?s wildcard)",
            file=sys.stderr,
        )
        return 0

    write_wav(Path(out_pattern), "SUBSONG 1 of 1", seconds=base_dur, amp=amp, normalize=normalize)
    print("fake-vgmstream: wrote 1 file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
