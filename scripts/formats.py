"""
vOS-VirtualVoice — 音訊格式偵測 (Phase 1, 任務 1)

為什麼需要「看檔案內容而不是看副檔名」
--------------------------------------
遊戲語音常見的坑：

1. 大量 `.wem` 其實只是「改了副檔名的 Ogg Vorbis」，可以直接餵 libsndfile/ffmpeg，
   根本不需要解包，也不需要 vgmstream。
2. 反之，某些 `.ogg` 其實是 Wwise Vorbis（RIFF 容器內含 `vorb` chunk），
   副檔名騙人。
3. `.acb` 是 CriWare 的 **cue sheet（索引表）**，音訊本體在 `.awb`；
   只解 `.acb` 常常什麼都拿不到。
4. 批次處理最怕「靜默失敗」——所以我們先 sniff，再決定用哪個解碼器，
   並且把判斷結果寫進 manifest 供事後稽核。

本模組只做「唯讀嗅探」，不改檔、不搬檔，零副作用。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 格式登記表
# ---------------------------------------------------------------------------
# key: 內部格式代號
# family: 高階分類（riff / ogg / criware / fmod / wwise_pkg / plain_audio / unknown）
# needs_unpack: True 代表「必須先解包成獨立音訊檔」才能進轉碼階段
FORMAT_TABLE: dict[str, dict[str, object]] = {
    "wav_riff": {
        "label": "RIFF/WAVE (PCM 或 ADPCM)",
        "family": "riff",
        "needs_unpack": False,
        "decoder": "soundfile",
    },
    "wem_vorbis": {
        "label": "Wwise Vorbis (.wem, RIFF+vorb)",
        "family": "riff",
        "needs_unpack": True,
        "decoder": "vgmstream",
    },
    "wem_other": {
        "label": "Wwise 其他編碼 (.wem, RIFF)",
        "family": "riff",
        "needs_unpack": True,
        "decoder": "vgmstream",
    },
    "ogg_vorbis": {
        "label": "Ogg Vorbis",
        "family": "ogg",
        "needs_unpack": False,
        "decoder": "soundfile",
    },
    "ogg_opus": {
        "label": "Ogg Opus",
        "family": "ogg",
        "needs_unpack": False,
        "decoder": "soundfile",
    },
    "flac": {
        "label": "FLAC",
        "family": "plain_audio",
        "needs_unpack": False,
        "decoder": "soundfile",
    },
    "adx": {
        "label": "CRI ADX (ADX4/MPEG 有損)",
        "family": "plain_audio",
        "needs_unpack": False,
        "decoder": "vgmstream",
    },
    "hca": {
        "label": "CRI HCA",
        "family": "plain_audio",
        "needs_unpack": False,
        "decoder": "vgmstream",
    },
    "fsb": {
        "label": "FMOD Sample Bank (.fsb/.bank)",
        "family": "fmod",
        "needs_unpack": True,
        "decoder": "vgmstream",
    },
    "cri_acb": {
        "label": "CriWare ACB (cue sheet, 音訊在 .awb)",
        "family": "criware",
        "needs_unpack": True,
        "decoder": "vgmstream",
    },
    "cri_awb": {
        "label": "CriWare AWB (音訊庫)",
        "family": "criware",
        "needs_unpack": True,
        "decoder": "vgmstream",
    },
    "cri_cpk": {
        "label": "CRI CPK 封裝包",
        "family": "criware",
        "needs_unpack": True,
        "decoder": "vgmstream",
    },
    "wwise_pck": {
        "label": "Wwise PCK（整包）",
        "family": "wwise_pkg",
        "needs_unpack": True,
        "decoder": "vgmstream",
    },
    "wwise_akpk": {
        "label": "Wwise .akpk / 新版 SoundBank",
        "family": "wwise_pkg",
        "needs_unpack": True,
        "decoder": "vgmstream",
    },
    "mp3": {"label": "MP3", "family": "plain_audio", "needs_unpack": False, "decoder": "soundfile"},
    "m4a_mp4": {"label": "MP4/M4A (AAC)", "family": "plain_audio", "needs_unpack": False, "decoder": "ffmpeg"},
    "aiff": {"label": "AIFF", "family": "plain_audio", "needs_unpack": False, "decoder": "soundfile"},
    "unknown": {
        "label": "未知格式",
        "family": "unknown",
        "needs_unpack": False,
        "decoder": "none",
    },
}

# 嗅探時讀取的位元組數（RIFF chunk 走訪需要在檔案內隨機讀，見 _probe_riff）
SNIFF_BYTES = 64


@dataclass(frozen=True)
class FormatInfo:
    """一個檔案的格式判斷結果。"""

    code: str
    label: str
    family: str
    needs_unpack: bool
    decoder: str
    detail: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "format_code": self.code,
            "format_label": self.label,
            "format_family": self.family,
            "needs_unpack": "yes" if self.needs_unpack else "no",
            "decoder": self.decoder,
            "format_detail": self.detail,
        }


def _make(code: str, detail: str = "") -> FormatInfo:
    meta = FORMAT_TABLE[code]
    return FormatInfo(
        code=code,
        label=str(meta["label"]),
        family=str(meta["family"]),
        needs_unpack=bool(meta["needs_unpack"]),
        decoder=str(meta["decoder"]),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# RIFF 走訪：分辨「真 WAV」與「Wwise .wem」
# ---------------------------------------------------------------------------
def _probe_riff(fh) -> FormatInfo:
    """
    走訪 RIFF 頂層 chunk，判斷是標準 WAV 還是 Wwise 容器。

    Wwise 的 .wem 有兩種常見長相：
      A. 標準 WAV 標頭 + `vorb` chunk  → 內含 Ogg Vorbis（最常見）
      B. 標準 WAV 標頭 + Wwise 專有 codec（如 `fmt ` 的 wFormatTag = 0xFFFF）
    """
    try:
        header = fh.read(12)
        if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            return _make("unknown", "RIFF 標頭不完整")
        riff_size = struct.unpack("<I", header[4:8])[0]

        seen: list[str] = []
        fmt_tag: int | None = None
        pos = 12
        # 走訪所有頂層 chunk，最多看 64 個或 1 MiB（避免壞檔無限迴圈）
        for _ in range(64):
            fh.seek(pos)
            hdr = fh.read(8)
            if len(hdr) < 8:
                break
            cid = hdr[0:4]
            csize = struct.unpack("<I", hdr[4:8])[0]
            seen.append(cid.decode("latin-1", "replace").strip())

            if cid == b"fmt ":
                fh.seek(pos + 8)
                tag = fh.read(2)
                if len(tag) == 2:
                    fmt_tag = struct.unpack("<H", tag)[0]
            if cid == b"vorb":
                return _make("wem_vorbis", f"RIFF size={riff_size}, chunks={'/'.join(seen)}")

            pos += 8 + csize + (csize & 1)  # RIFF chunk 需 2-byte 對齊
            if pos > 1_048_576:
                break

        if fmt_tag is not None:
            # 0x0001 = PCM, 0x0003 = IEEE float, 0x0002 = MS ADPCM → 標準 WAV
            if fmt_tag in (0x0001, 0x0002, 0x0003, 0x0011, 0x0055):
                return _make("wav_riff", f"fmt_tag=0x{fmt_tag:04X}, chunks={'/'.join(seen)}")
            return _make(
                "wem_other",
                f"Wwise 專有編碼 fmt_tag=0x{fmt_tag:04X}, chunks={'/'.join(seen)}",
            )
        return _make("unknown", f"RIFF 無 fmt chunk, chunks={'/'.join(seen)}")
    except (OSError, struct.error) as exc:
        return _make("unknown", f"RIFF 解析失敗: {exc}")


# ---------------------------------------------------------------------------
# 主嗅探函式
# ---------------------------------------------------------------------------
def sniff_format(path: str | Path) -> FormatInfo:
    """
    嗅探單一檔案的真實格式。永不拋例外（讀不到就回 unknown 並附原因），
    方便直接餵進批次流程。
    """
    p = Path(path)
    try:
        with p.open("rb") as fh:
            head = fh.read(SNIFF_BYTES)
            if len(head) < 4:
                return _make("unknown", f"檔案過小 ({len(head)} bytes)")

            if head[0:4] == b"RIFF":
                fh.seek(0)
                return _probe_riff(fh)

            if head[0:4] == b"OggS":
                # Ogg 的編碼種類要看第一個頁面的 codec magic
                fh.seek(0)
                blob = fh.read(256)
                if b"OpusHead" in blob:
                    return _make("ogg_opus")
                if b"\x01vorbis" in blob:
                    return _make("ogg_vorbis")
                return _make("ogg_vorbis", "Ogg 容器但未辨識 codec magic")

            if head[0:4] == b"fLaC":
                return _make("flac")

            if head[0:3] == b"ADX":
                return _make("adx", f"version={head[3]:02X}")

            # CRI HCA：magic 可能是 0x48434100 ("HCA\0") 或加密後的 0xFFFFFFFF + 'HCA'
            if head[0:3] == b"HCA" or head[4:7] == b"HCA":
                return _make("hca")

            if head[0:3] == b"FSB":
                version = chr(head[3]) if 48 <= head[3] <= 57 else "?"
                return _make("fsb", f"FSB{version}")

            if head[0:4] == b"AKPK":
                return _make("wwise_akpk")

            if head[0:4] == b"@UTF":
                return _make("cri_acb", "ACB 中的 UTF 表（cue sheet）")

            if head[0:4] == b"AFS2":
                return _make("cri_awb", "AFS2 音訊庫")

            # 少數 AWB 直接以 AWB 標頭開頭（非 AFS2）
            if head[0:3] == b"AWB":
                return _make("cri_awb")

            # Wwise PCK：以 "AKPK" 開頭者已處理；舊版 PCK 開頭為 'PCK' 或含 BKHD 區塊
            if head[0:4] == b"PCK\x00" or head[0:3] == b"PCK":
                return _make("wwise_pck", "Wwise PCK 舊版標頭")

            if head[0:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
                return _make("mp3")

            if head[4:8] == b"ftyp":
                return _make("m4a_mp4", f"brand={head[8:12].decode('latin-1', 'replace')}")

            if head[0:4] == b"FORM":
                return _make("aiff")

            # 以 'CRI' 開頭但非上述者，多半是 CPK
            if head[0:3] == b"CRI":
                return _make("cri_cpk", f"magic={head[0:4]!r}")

            return _make("unknown", f"magic={head[0:4]!r}")
    except OSError as exc:
        return _make("unknown", f"無法讀取: {exc}")


def build_test_riff(*, chunks: list[tuple[bytes, bytes]] | None = None) -> bytes:
    """
    產生合成 RIFF 位元組，供單元測試使用（不需要真的音訊檔）。

    build_test_riff()                                   -> 標準 PCM WAV
    build_test_riff(chunks=[(b"fmt ", pcm_fmt), (b"vorb", b"x" * 8)])  -> Wwise Vorbis
    """
    if chunks is None:
        fmt = struct.pack("<HHIIHH", 0x0001, 1, 48_000, 48_000 * 2, 2, 16)
        chunks = [(b"fmt ", fmt), (b"data", b"\x00" * 16)]

    body = b""
    for cid, payload in chunks:
        body += cid + struct.pack("<I", len(payload)) + payload
        if len(payload) & 1:
            body += b"\x00"  # 對齊位元組
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def describe(path: str | Path) -> str:
    """產生人類可讀的單行說明（給 CLI 用）。"""
    info = sniff_format(path)
    tag = "需解包" if info.needs_unpack else "可直接解碼"
    detail = f"  [{info.detail}]" if info.detail else ""
    return f"{Path(path).name}: {info.label} ({tag}, decoder={info.decoder}){detail}"


__all__ = ["FORMAT_TABLE", "FormatInfo", "build_test_riff", "describe", "sniff_format"]
