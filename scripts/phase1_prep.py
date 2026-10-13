#!/usr/bin/env python3
"""
vOS-VirtualVoice — Phase 1 主線腳本：解包 + 格式標準化 + 時長篩選
=================================================================

一次把遊戲原始語音處理成 GPT-SoVITS 可以直接吃的訓練素材：

    data/00_raw/  ──(1 解包)──▶  01_unpacked/
                  ──(2 解碼+重採樣+單聲道)──▶
                  ──(3 時長篩選 1.5s~15s + 品質關卡)──▶  02_wav24_48k/   ✅ 合格
                                                        99_rejected/     ❌ 淘汰
                                    並輸出 data/_reports/manifest_phase1.csv

跨平台
------
* 只用 pathlib / os.path 處理路徑，Windows / macOS / Linux 皆可
* 外部工具（vgmstream-cli / ffmpeg）用 shutil.which 自動尋找，不寫死路徑
* 支援 .exe / 無副檔名；避開 shell，沒有引號地獄

常用指令
--------
    # 0) 先看環境與素材概況（不寫任何檔案）
    python scripts/phase1_prep.py doctor
    python scripts/phase1_prep.py scan

    # 1) 解包遊戲容器（.wem / .awb / .fsb / .acb / .cpk / .pck）
    python scripts/phase1_prep.py unpack

    # 2) 標準化：24bit/48kHz 單聲道 WAV + 時長篩選 + 品質關卡
    python scripts/phase1_prep.py convert

    # 3) 一條龍（scan → unpack → convert）
    python scripts/phase1_prep.py all

    # 常用選項
    --config configs/pipeline.local.json   改用本機設定
    --dry-run                              只報告、不寫檔
    --limit 50                             只處理前 50 個（先驗證流程）
    --workers 8                            平行解碼（I/O 密集，8~CPU 核心數）
    --overwrite                            已存在的輸出也重做
    --no-quality-gate                      只做時長篩選
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

# 讓 `python scripts/phase1_prep.py` 直接執行時也能匯入同層模組
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# 主控台編碼防護（跨平台痛點）
# ---------------------------------------------------------------------------
# Windows 的舊主控台預設是 cp950 / cp936 / cp932 等非 UTF-8 編碼。
# 只要輸出裡有一個 '✔'，整個批次就會以 UnicodeEncodeError 收場——
# 而且是在跑了幾千個檔案之後才炸。這裡一次處理掉：
#   1. 盡量把 stdout/stderr 切成 UTF-8（Python 3.7+ 的 reconfigure）
#   2. 若切不過去（例如被重導向到不支援的 pipe），退回純 ASCII 符號
def _configure_console() -> bool:
    """回傳是否可以使用 Unicode 符號。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    enc = (getattr(sys.stdout, "encoding", None) or "ascii").lower()
    return "utf" in enc


UNICODE_OK = _configure_console()
MARK = (
    {"ok": "✔", "rejected": "✂", "skipped": "·", "error": "✘"}
    if UNICODE_OK
    else {"ok": "[OK]", "rejected": "[CUT]", "skipped": "[--]", "error": "[ERR]"}
)
OK, BAD, WARN, INFO = ("✔", "✘", "⚠", "ℹ") if UNICODE_OK else ("[OK]", "[X]", "[!]", "[i]")

from _audioio import (  # noqa: E402
    decode,
    measure,
    probe_backends,
    resample,
    run_tool,
    safe_db,
    to_mono,
    write_wav,
)
import config  # noqa: E402  （repo_root 的環境變數覆寫需要整個模組）
from config import PipelineConfig, find_external_bin, load_config  # noqa: E402
from formats import FORMAT_TABLE, FormatInfo, sniff_format  # noqa: E402
from unpack import (  # noqa: E402
    UnpackOptions,
    estimate_total_subsongs,
    probe_with_vgmstream,
    unpack_one,
)
from audition import build_reel, filesystem_safe, list_candidates, plan_samples  # noqa: E402
from tune import analyze, read_manifest, render_report, report_as_dict  # noqa: E402
from report import collect, render, render_json  # noqa: E402
from selftest import run_selftest  # noqa: E402
from space import estimate_from_source_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# Manifest schema（欄位順序固定，方便 Phase 4 用 pandas 讀）
# ---------------------------------------------------------------------------
MANIFEST_FIELDS = [
    "status",            # ok / rejected / skipped / error
    "reason",            # 可讀原因（中文），供人工複核
    "group",             # 群組（多角色專案 = 角色名；單一角色 = default）
    "src_path",          # 相對 repo 根目錄的來源路徑
    "src_format",        # sniff 出的格式代號
    "src_format_label",  # 人類可讀格式
    "src_bytes",         # 來源檔大小
    "src_sr",            # 來源取樣率
    "src_channels",      # 來源聲道數（原始）
    "duration_sec",      # 時長（秒）
    "out_path",          # 產物路徑（未產出則空）
    "out_sr",            # 產物取樣率
    "out_channels",      # 產物聲道數
    "out_bit_depth",     # 產物位元深度
    "peak_dbfs",
    "rms_dbfs",
    "decoder",           # 實際使用的解碼後端
    "error",             # 失敗細節
]


@dataclass
class FileResult:
    """單一檔案的處理結果（= manifest 一列）。"""

    status: str = "error"
    reason: str = ""
    group: str = "default"
    src_path: str = ""
    src_format: str = ""
    src_format_label: str = ""
    src_bytes: int = 0
    src_sr: int = 0
    src_channels: int = 0
    duration_sec: str = ""
    out_path: str = ""
    out_sr: str = ""
    out_channels: str = ""
    out_bit_depth: str = ""
    peak_dbfs: str = ""
    rms_dbfs: str = ""
    decoder: str = ""
    error: str = ""

    def row(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    """
    repo 根目錄。

    可由環境變數 VOS_REPO_ROOT 覆寫，用途：
      * 自動化測試在獨立沙箱內跑完整 CLI，完全不碰真正的 data/
      * 素材放在外接硬碟、想把產物寫到別處
    """
    return config.repo_root()


def rel(path: Path, root: Path) -> str:
    """路徑轉相對字串；超出 repo 範圍時回傳原路徑（跨磁碟時 relative_to 會拋錯）。"""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def human_bytes(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PiB"


def display_width(text: str) -> int:
    """
    估算字串在等寬終端上的顯示寬度。

    中日韓字元（East Asian Wide/Fullwidth）在終端佔兩格，
    len() 只算一個，直接拿來排版會歪掉。這裡用 unicodedata 判斷。
    """
    import unicodedata

    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def pad_display(text: str, width: int) -> str:
    """把字串補到指定的顯示寬度（給含中文的表格用）。"""
    if display_width(text) > width:
        # 太長就截斷，並補上省略號維持寬度
        out = ""
        for ch in text:
            if display_width(out + ch) > width - 1:
                break
            out += ch
        return out + "…" + " " * max(0, width - display_width(out) - 1)
    return text + " " * (width - display_width(text))


def scan_sources(cfg: PipelineConfig, root: Path) -> list[Path]:
    """
    掃描 data/00_raw（含子資料夾）內所有目標副檔名的檔案。

    注意：這裡刻意「不」跳過 .wav——很多遊戲語音解包出來就是 wav，
    它們一樣要走重採樣/位元深度統一。
    """
    exts = set(cfg.scan_extensions)
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(cfg.paths.raw_dir):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                found.append(p)
    found.sort(key=lambda x: rel(x, root))
    return found


def duration_of(samples, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(len(samples)) / float(sample_rate)


# ---------------------------------------------------------------------------
# 子命令：doctor
# ---------------------------------------------------------------------------
def cmd_doctor(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    """環境自我檢查：Python / 依賴 / 外部工具 / 資料夾。"""
    root = repo_root()
    print("=" * 66)
    print(" vOS-VirtualVoice — 環境檢查 (Phase 1)")
    print("=" * 66)

    print(f"\n[Python] {sys.version.split()[0]}  ({sys.platform})")
    print(f"[執行檔] {sys.executable}")
    print(f"[設定檔] {rel(cfg.source_file, root)}")

    print("\n[後端可用性]")
    backends = probe_backends()
    critical_missing: list[str] = []
    for b in backends:
        mark = OK if b.available else BAD
        line = f"  {mark} {b.name:<22} {b.version}"
        if b.note:
            line += f"    {b.note}"
        print(line)

    if not backends[0].available:
        critical_missing.append("soundfile（pip install soundfile）")
    if not any(b.available for b in backends if b.name in ("soxr (重採樣)", "scipy (後備重採樣)")):
        critical_missing.append("soxr 或 scipy（重採樣必需）")

    print("\n[外部工具]")
    for tool, why in (
        ("ffmpeg", "通用解碼後備（m4a/aac/wma）"),
        ("ffprobe", "讀取取樣率"),
        ("vgmstream-cli", "Wwise/CriWare/FMOD 遊戲容器解包"),
    ):
        found = find_external_bin(tool)
        print(f"  {OK if found else BAD} {tool:<16} {found or '（未找到）':<44} {why}")

    print("\n[資料夾]")
    for label, p in (
        ("raw", cfg.paths.raw_dir),
        ("unpacked", cfg.paths.unpacked_dir),
        ("standardized", cfg.paths.standardized_dir),
        ("rejected", cfg.paths.rejected_dir),
        ("reports", cfg.paths.reports_dir),
    ):
        exists = "存在" if p.is_dir() else "尚未建立"
        print(f"  {label:<14} {rel(p, root):<34} [{exists}]")

    print("\n[目標格式]")
    t = cfg.target
    print(f"  {t.bit_depth} bit / {t.sample_rate} Hz / {'單聲道' if t.channels == 1 else '雙聲道'} PCM WAV")
    print(f"  時長門檻：{cfg.duration_filter.min_seconds}s ~ {cfg.duration_filter.max_seconds}s")
    q = cfg.quality_gate
    print(
        f"  品質關卡：{'啟用' if q.enabled else '停用'}"
        f"（peak ≤ {q.peak_dbfs_max} dBFS, RMS ≥ {q.min_rms_dbfs} dBFS）"
    )

    if critical_missing:
        print(f"\n{WARN} 缺少必要依賴：")
        for m in critical_missing:
            print(f"   - {m}")
        print("   修正：pip install -r requirements.txt")
        return 1

    if not find_external_bin("vgmstream-cli"):
        print(
            f"\n{INFO} 未偵測到 vgmstream-cli。若你的素材是 .wem/.awb/.fsb 等遊戲容器，"
            "請先安裝（見 docs/PHASE1-INGEST.md『工具鏈安裝』）；\n"
            "  若素材已是 .wav/.ogg/.flac 等標準格式，可直接執行 convert。"
        )
    print(f"\n{OK} 核心依賴就緒。")
    return 0


def sniff_with_fallback(path: Path, vgm_exe: str | None) -> tuple[FormatInfo, str]:
    """
    嗅探格式；自己的 magic 表認不出來時，改問 vgmstream。

    為什麼要這樣做：我們的 magic 表只認得二十幾種格式，而 vgmstream 認得
    幾百種。遊戲素材千奇百怪，遇到「未知格式」時繼續手寫 magic 判斷
    是事倍功半 —— 直接問 vgmstream 快得多，也更可靠。

    判定成功時回傳 `vgmstream_generic`，並把 decoder 設成 vgmstream，
    這樣後續的 convert 也會用正確的後端處理它（而不是當成 unknown 跳過）。

    Returns:
        (FormatInfo, 額外說明)。說明為空字串代表是用自己的表判定出來的。
    """
    info = sniff_format(path)
    if info.code != "unknown" or not vgm_exe:
        return info, ""

    probe = probe_with_vgmstream(path, vgm_exe)
    if not probe.readable:
        return info, probe.detail
    return _make_vgmstream_info(probe), probe.summary


def _make_vgmstream_info(probe) -> FormatInfo:
    """把 vgmstream 的判定包成 FormatInfo（decoder 固定為 vgmstream）。"""
    meta = FORMAT_TABLE["vgmstream_generic"]
    return FormatInfo(
        code="vgmstream_generic",
        label=str(meta["label"]),
        family=str(meta["family"]),
        needs_unpack=bool(meta["needs_unpack"]),
        decoder="vgmstream",
        detail=probe.summary,
    )


# ---------------------------------------------------------------------------
# 子命令：scan
# ---------------------------------------------------------------------------
def cmd_scan(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    """只嗅探不處理：統計原始素材的『真實格式』分布。這是最重要的第一步。"""
    root = repo_root()
    sources = scan_sources(cfg, root)
    if not sources:
        print(f"{WARN} {rel(cfg.paths.raw_dir, root)} 內找不到任何目標副檔名：")
        print(f"   {', '.join(cfg.scan_extensions)}")
        print("   請把遊戲語音放進該資料夾（可含子資料夾）。")
        return 1

    by_format: dict[str, list[Path]] = {}
    total_bytes = 0
    ext_mismatch: list[tuple[Path, str]] = []
    fallback_notes: dict[str, str] = {}
    unknown_leftover: list[Path] = []

    # 有 vgmstream 就當作「未知格式」的後備判定工具
    vgm_exe = None if args.no_fallback else find_external_bin(
        "vgmstream-cli", cfg.unpack_vgmstream_bin
    )

    for p in sources:
        info, note = sniff_with_fallback(p, vgm_exe)
        by_format.setdefault(info.code, []).append(p)
        if note:
            fallback_notes.setdefault(info.code, note)
        elif info.code == "unknown":
            unknown_leftover.append(p)
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass
        if _ext_mismatch(p, info):
            ext_mismatch.append((p, info.label))

    print("=" * 78)
    print(f" 素材掃描：{len(sources)} 個檔案，{human_bytes(total_bytes)}")
    print(f" 來源：{rel(cfg.paths.raw_dir, root)}")
    print("=" * 78)
    print(f"{'格式':<42}{'數量':>7}{'需解包':>8}")
    print("-" * 78)
    for code, files in sorted(by_format.items(), key=lambda kv: -len(kv[1])):
        # 用實際嗅探結果（含 vgmstream 後備判定），不要重新 sniff —— 那會
        # 再跑一次外部程序，而且會把後備判定出來的格式顯示成「未知」。
        if fallback_notes.get(code):
            label = str(FORMAT_TABLE["vgmstream_generic"]["label"])
            needs = "否"  # 後備判定代表 vgmstream 能直接解，不需先解包
            note = f"   ← {fallback_notes[code]}"
        else:
            info_first = sniff_format(files[0])
            label = info_first.label
            needs = "是" if info_first.needs_unpack else "否"
            note = ""
        print(f"{label:<42}{len(files):>7}{needs:>8}{note}")

    # 需不需要解包：以「是否走 vgmstream 且為容器」為準
    def _needs_unpack_code(code: str) -> bool:
        if code == "vgmstream_generic":
            return False  # 後備判定代表 vgmstream 能直接解
        return sniff_format(by_format[code][0]).needs_unpack

    need_unpack = [
        f for c, fs in by_format.items() if _needs_unpack_code(c) for f in fs
    ]
    direct = len(sources) - len(need_unpack)

    print("-" * 78)
    print(f"  可直接解碼：{direct} 個   需先解包：{len(need_unpack)} 個")

    if fallback_notes:
        n_fallback = sum(len(by_format[c]) for c in fallback_notes)
        print(
            f"\n{INFO} 有 {n_fallback} 個檔案是「vgmstream 可解碼但格式未辨識」"
            f"—— 這不影響流程，會直接用 vgmstream 解碼。"
        )

    if ext_mismatch:
        print(f"\n{WARN} 副檔名與實際格式不符 {len(ext_mismatch)} 個（前 8 筆）：")
        for p, label in ext_mismatch[:8]:
            print(f"   {rel(p, root)}  → 實際為 {label}")
        print("   （這不影響流程：本工具一律以內容為準，只提示你別手動改名）")

    if need_unpack:
        print("\n下一步：python scripts/phase1_prep.py unpack")
    else:
        print("\n下一步：python scripts/phase1_prep.py convert")
    return 0


def _ext_mismatch(p: Path, info: FormatInfo) -> bool:
    """副檔名與嗅探結果明顯衝突時回 True。"""
    ext = p.suffix.lower()
    if ext in (".wem", ".wav") and info.code in ("ogg_vorbis", "ogg_opus"):
        return True
    if ext == ".ogg" and info.code in ("wem_vorbis", "wav_riff"):
        return True
    return ext in (".ogg", ".wav") and info.code == "unknown"


# ---------------------------------------------------------------------------
# 子命令：unpack
# ---------------------------------------------------------------------------
def cmd_unpack(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    """
    用 vgmstream-cli 把遊戲容器解成獨立音訊檔。

    為什麼不自己寫 parser：Wwise 的 codec（Vorbis/Opus/ADPCM/PCM/PTADPCM）、
    CriWare 的 HCA 加密金鑰、FMOD 各版本 FSB 格式差異極大，
    vgmstream 是社群維護十年以上的事實標準，重寫不划算。

    實際的解包邏輯在 scripts/unpack.py（可單元測試），這裡只負責 CLI 互動。
    """
    root = repo_root()
    vgm = find_external_bin("vgmstream-cli", cfg.unpack_vgmstream_bin)
    if not vgm:
        print(f"{BAD} 找不到 vgmstream-cli，無法解包。")
        print("  安裝方式見 docs/PHASE1-INGEST.md（三平台各自一行指令，或放進 tools/bin/）")
        print("  若素材已是 .wav/.ogg/.flac，請直接執行：python scripts/phase1_prep.py convert")
        return 1

    sources = [p for p in scan_sources(cfg, root) if sniff_format(p).needs_unpack]
    if not sources:
        print(f"{INFO} 沒有需要解包的容器檔，略過。直接跑 convert 即可。")
        return 0

    options = UnpackOptions(
        max_subsongs=cfg.unpack_max_subsongs,
        probe_subsongs=not args.no_probe,
        extra_args=cfg.unpack_extra_args,
    )
    cfg.paths.unpacked_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" Phase 1 容器解包")
    print("=" * 72)
    print(f" 容器數：{len(sources)} 個 → {rel(cfg.paths.unpacked_dir, root)}")
    print(f" 工具：{vgm}")
    print(f" 單容器 subsong 上限：{options.max_subsongs}")

    # 先抽樣探測，對「一次解出幾萬個檔案」發出事前警告
    if options.probe_subsongs and not args.dry_run:
        sampled, probed = estimate_total_subsongs(sources, vgm, options)
        if probed:
            avg = sampled / probed
            estimate = int(avg * len(sources))
            print(f" 抽樣 {probed} 個容器平均 {avg:.0f} 個 subsong")
            if estimate > 20000:
                print(
                    f"\n {WARN} 依抽樣推算，本次可能解出約 {estimate:,} 個檔案。"
                    f"請確認磁碟空間（每個檔案數十 KB 到數 MB）。"
                )
                print("   建議先用 --limit 5 小量驗證。")
    print("=" * 72)

    # 磁碟空間預檢。
    #
    # 為什麼一定要做：解包會把壓縮音訊變成 PCM，資料量放大 8~18 倍。
    # 寫爆磁碟的後果不是「失敗」而是「部分成功」—— 容器只解出一部分 subsong，
    # 卻沒有明顯錯誤訊息，Phase 2 會把截斷的素材當成正常資料餵進模型。
    src_bytes = 0
    for p in sources:
        try:
            src_bytes += p.stat().st_size
        except OSError:
            pass
    est = estimate_from_source_bytes(
        src_bytes, cfg.paths.unpacked_dir, label="解包產物（01_unpacked）"
    )
    print()
    print(est.render())
    if not est.sufficient and not args.force:
        print(
            f"\n {BAD} 空間不足，已中止（避免寫出殘缺的產物）。\n"
            f"    三個選項：\n"
            f"      1. 清出空間後重跑\n"
            f"      2. 先用 --limit 5 小量驗證流程\n"
            f"      3. 確定要硬跑：加 --force"
        )
        return 1
    if not est.sufficient:
        print(f"\n {WARN} 空間不足但你指定了 --force，繼續執行。")
    ok = fail = skipped = already = 0
    produced: list[Path] = []

    for i, src in enumerate(sources, 1):
        info = sniff_format(src)
        raw_relative = src.relative_to(cfg.paths.raw_dir)
        dest_dir = cfg.paths.unpacked_dir / raw_relative.parent / src.stem
        prefix = f"[{i}/{len(sources)}] {raw_relative.as_posix()} ({info.label})"

        if args.dry_run:
            print(f"  {prefix}  → dry-run，略過")
            continue

        res = unpack_one(src, dest_dir, vgm, options, force=args.overwrite)

        if res.skipped:
            if res.out_paths:  # 已解包過（冪等略過），不是被上限擋掉
                already += 1
                produced.extend(res.out_paths)
                if args.verbose:
                    print(f"  {INFO} {prefix} → {res.error}")
            else:
                skipped += 1
                print(f"  {WARN} {prefix}\n      {res.error}")
            continue
        if not res.ok:
            fail += 1
            print(f"  {BAD} {prefix}\n      {res.error}")
            continue

        ok += 1
        produced.extend(res.out_paths)
        label = f"（{res.subsongs} 個 subsong）" if res.subsongs > 1 else ""
        print(f"  {OK} {prefix} → {len(res.out_paths)} 個檔案{label}")

        # 每 25 個印一次進度，避免幾千行洗版
        if not args.verbose and i % 25 == 0:
            print(f"  ... 進度 {i}/{len(sources)}  {OK}{ok} {WARN}{skipped} {BAD}{fail}")

    if not args.dry_run:
        total = sum(p.stat().st_size for p in produced if p.is_file())
        print("\n" + "=" * 72)
        print(
            f" 解包完成：新解 {ok}，已存在 {already}，跳過 {skipped}，失敗 {fail}"
            f"，可用產物共 {len(produced)} 個（{human_bytes(total)}）"
        )
        if already:
            print(
                f" {INFO} 有 {already} 個容器之前就解包過（未重做）。"
                f"要強制重解請加 --overwrite。"
            )
        if skipped:
            print(
                f" {INFO} 有 {skipped} 個容器因 subsong 過多被跳過。"
                f"要全部解出請調高 configs 的 unpack.max_subsongs。"
            )
        if fail:
            print(f" {WARN} 失敗的容器多半是缺少 companion 檔（金鑰／索引表）或格式不支援；")
            print("   錯誤訊息已附在上方，請對照 docs/PHASE1-INGEST.md 的狀況排除表。")
        print("\n 下一步：python scripts/phase1_prep.py convert")
    return 0 if fail == 0 else 2


# ---------------------------------------------------------------------------
# 子命令：convert（核心）
# ---------------------------------------------------------------------------
def process_one(src: Path, cfg: PipelineConfig, args: argparse.Namespace) -> FileResult:
    """
    處理單一檔案：解碼 → 單聲道 → 重採樣 → 時長篩選 → 品質關卡 → 寫 24bit WAV。

    刻意先「解碼 + 量測」再決定去留：
      * 時長必須從真實音訊算，不能靠檔頭（很多 wem 的檔頭長度欄位不可信）
      * 品質關卡（peak/RMS）也只有解碼後才算得出來
    """
    root = repo_root()
    res = FileResult(src_path=rel(src, root), group=cfg.group_of(src))
    info = sniff_format(src)
    res.src_format = info.code
    res.src_format_label = info.label
    try:
        res.src_bytes = src.stat().st_size
    except OSError:
        pass

    if info.needs_unpack:
        res.status = "skipped"
        res.reason = "容器檔，需先解包（執行 unpack）"
        return res

    dec = decode(src, sample_rate=None)
    res.decoder = dec.backend
    if not dec.ok:
        res.status = "error"
        res.reason = "解碼失敗"
        res.error = dec.error
        return res

    mono = to_mono(dec.samples)
    res.src_sr = dec.sample_rate
    res.src_channels = 1 if dec.samples.ndim == 1 else int(dec.samples.shape[1])

    dur = duration_of(mono, dec.sample_rate)
    res.duration_sec = f"{dur:.3f}"

    # ---- 規則 1：時長 ----
    lo, hi = cfg.duration_filter.min_seconds, cfg.duration_filter.max_seconds
    if dur < lo:
        res.status, res.reason = "rejected", f"過短 {dur:.2f}s < {lo}s（喘氣/單字哼聲）"
        return _maybe_write_rejected(res, cfg, args, src)
    if dur > hi:
        res.status, res.reason = "rejected", f"過長 {dur:.2f}s > {hi}s"
        return _maybe_write_rejected(res, cfg, args, src)

    # ---- 規則 2：品質關卡 ----
    peak, rms = measure(mono)
    res.peak_dbfs = safe_db(peak)
    res.rms_dbfs = safe_db(rms)
    if cfg.quality_gate.enabled:
        if peak == float("-inf"):
            res.status, res.reason = "rejected", "全靜音（無有效訊號）"
            return _maybe_write_rejected(res, cfg, args, src)
        if peak > cfg.quality_gate.peak_dbfs_max:
            res.status = "rejected"
            # 措辭刻意精準：這裡擋的是「已經貼頂、有削波風險」的素材。
            # 真正的波形削平必須由 RMS 一起判斷（削平的方波 RMS 會逼近 Peak），
            # 所以不要寫成斷言式的「已削波」，免得使用者誤以為門檻可以救。
            res.reason = (
                f"削波風險 peak {peak:.2f} > {cfg.quality_gate.peak_dbfs_max} dBFS"
                f"（門檻可調，但波形已受損者無法還原）"
            )
            return _maybe_write_rejected(res, cfg, args, src)
        if rms < cfg.quality_gate.min_rms_dbfs:
            res.status = "rejected"
            res.reason = f"訊號過弱 RMS {rms:.1f} < {cfg.quality_gate.min_rms_dbfs} dBFS"
            return _maybe_write_rejected(res, cfg, args, src)

    # ---- 通過：重採樣 → 寫檔 ----
    try:
        audio = resample(mono, dec.sample_rate, cfg.target.sample_rate)
    except Exception as exc:
        res.status, res.reason, res.error = "error", "重採樣失敗", str(exc)
        return res

    out_path = cfg.resolve_output_for(src, cfg.paths.standardized_dir)
    res.out_path = rel(out_path, root)
    res.out_sr = cfg.target.sample_rate
    res.out_channels = cfg.target.channels
    res.out_bit_depth = cfg.target.bit_depth

    if args.dry_run:
        res.status, res.reason = "ok", "dry-run（通過所有檢查，未寫檔）"
        return res

    if out_path.exists() and not args.overwrite:
        res.status, res.reason = "ok", "已存在（未覆寫）"
        return res

    try:
        write_wav(out_path, audio, cfg.target.sample_rate, subtype=cfg.target.subtype)
    except Exception as exc:
        res.status, res.reason, res.error = "error", "寫檔失敗", str(exc)
        return res

    # 寫檔後回讀驗證：確保 24-bit / 48kHz 真的落地（防止靜默降級）
    actual = _verify_written(out_path)
    if actual and actual != (cfg.target.sample_rate, cfg.target.subtype):
        res.error = f"寫檔後規格不符：{actual}"
        res.status, res.reason = "error", "輸出規格驗證失敗"
        return res

    res.status, res.reason = "ok", "通過"
    return res


def _maybe_write_rejected(
    res: FileResult, cfg: PipelineConfig, args: argparse.Namespace, src: Path
) -> FileResult:
    """
    把被淘汰的檔案「複製」到 99_rejected 供人工複核。

    兩個刻意的設計決定：
      1. 用 copy2 而非 move —— 原始素材永遠是唯一的真實來源。調參過程中反覆
         調整門檻時，素材若被搬走會非常痛苦；寧可多佔一點硬碟。
      2. 保持原副檔名（不做任何轉檔）—— 有些「壞檔」其實只是格式特殊，
         保持原樣才能用別的播放器複查。
    """
    if args.dry_run:
        return res
    try:
        dest = cfg.paths.rejected_dir / _rejected_relpath(src, cfg)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or args.overwrite:
            shutil.copy2(src, dest)
        res.out_path = rel(dest, repo_root())
    except (OSError, ValueError) as exc:
        res.error = f"複製淘汰檔失敗：{exc}"
    return res


def _rejected_relpath(src: Path, cfg: PipelineConfig) -> Path:
    """
    決定淘汰檔在 99_rejected/ 下的相對路徑。

    ⚠ 這裡有一個真實踩過的坑：不同角色常常有**同名檔案**（例如
    凛/voice_001.wav 與 桜/voice_001.wav）。若只取檔名，
    後處理的檔案會直接覆蓋前一個，你會以為淘汰區只有一個檔。

    因此優先序為：
      1. 群組（角色名）+ 相對結構 → 99_rejected/<角色>/...
      2. 相對 00_raw 的結構
      3. 相對 01_unpacked 的結構
      4. 最後才退回單一檔名
    """
    group = cfg.group_of(src)
    for base in (cfg.paths.raw_dir, cfg.paths.unpacked_dir):
        try:
            rel = src.relative_to(base)
        except ValueError:
            continue
        # 群組名已經是 rel 的第一段時不重複加一層
        if group != "default" and rel.parts and rel.parts[0] == group:
            return rel
        return (Path(group) / rel) if group != "default" else rel
    return Path(src.name)


def _verify_written(path: Path) -> tuple[int, str] | None:
    """回讀 WAV 檔頭，確認取樣率與 subtype。失敗回 None。"""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return int(info.samplerate), str(info.subtype)
    except Exception:
        return None


def cmd_convert(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    root = repo_root()
    sources = scan_sources(cfg, root)

    # convert 預設處理『已解包 + 已是標準格式』的檔案：
    #   01_unpacked/ ∪ (00_raw/ 內非容器的檔案)
    targets: list[Path] = []
    if cfg.paths.unpacked_dir.is_dir():
        exts = set(cfg.scan_extensions)
        for dirpath, _d, files in os.walk(cfg.paths.unpacked_dir):
            for name in files:
                p = Path(dirpath) / name
                if p.suffix.lower() in exts:
                    targets.append(p)
    direct = [p for p in sources if not sniff_format(p).needs_unpack]
    targets.extend(direct)
    # 去重（同一路徑可能同時被兩邊掃到）
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in sorted(targets, key=lambda x: rel(x, root)):
        key = str(p.resolve()).lower() if os.name == "nt" else str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    targets = uniq

    if not targets:
        print(f"{WARN} 找不到可轉換的檔案。")
        print(f"   請把素材放進 {rel(cfg.paths.raw_dir, root)}（或先執行 unpack）。")
        return 1

    if args.limit:
        targets = targets[: args.limit]

    if args.workers <= 0:
        args.workers = min(8, os.cpu_count() or 4)

    containers = sum(1 for p in sources if sniff_format(p).needs_unpack)
    print("=" * 72)
    print(" Phase 1 格式標準化")
    print("=" * 72)
    print(f" 待處理：{len(targets)} 個檔案" + (f"（已限制前 {args.limit} 個）" if args.limit else ""))
    if containers:
        print(f" {INFO} 另有 {containers} 個容器檔尚未解包（先跑 unpack 才會納入本階段）")
    t = cfg.target
    print(f" 目標格式：{t.bit_depth}bit / {t.sample_rate}Hz / {'mono' if t.channels == 1 else 'stereo'} PCM WAV")
    print(
        f" 篩選規則：{cfg.duration_filter.min_seconds}s ≤ 長度 ≤ {cfg.duration_filter.max_seconds}s"
        + ("  + 品質關卡" if cfg.quality_gate.enabled and not args.no_quality_gate else "")
    )
    print(f" 輸出：{rel(cfg.paths.standardized_dir, root)} ／ 淘汰：{rel(cfg.paths.rejected_dir, root)}")
    print(f" 模式：{'DRY-RUN（不寫檔）' if args.dry_run else '實際寫檔'}"
          f"{'  ｜ 平行 ' + str(args.workers) if args.workers > 1 else ''}")
    print("=" * 72)

    # 磁碟空間預檢（convert 有兩份輸出，兩個都要算）：
    #   1. 合格區：24-bit/48kHz 單聲道，144 KB/s（可精確預估）
    #   2. 淘汰區：來源檔的**完整複本**（我們刻意用複製而非搬移，
    #      讓原始素材永遠留著）—— 漏算這項是很容易犯的錯，
    #      因為被淘汰的檔案往往佔多數。
    if not args.dry_run:
        src_bytes = 0
        for p in targets:
            try:
                src_bytes += p.stat().st_size
            except OSError:
                pass
        est_out = estimate_from_source_bytes(
            src_bytes, cfg.paths.standardized_dir, label="標準化產物（02_wav24_48k）"
        )
        est_rej = estimate_from_source_bytes(
            src_bytes,
            cfg.paths.rejected_dir,
            label="淘汰區複本（99_rejected）",
            already_pcm=True,
        )
        combined_free = min(est_out.free_bytes, est_rej.free_bytes)
        needed = est_out.needed_bytes + est_rej.needed_bytes
        print()
        print(est_out.render())
        print(f" 淘汰區另需約 {est_out.human(est_rej.high_bytes) if hasattr(est_out,'human') else est_rej.low_bytes}（來源檔的完整複本）")
        if combined_free < needed and not args.force:
            print(
                f"\n {BAD} 空間不足（合計需約 {needed / 1024**3:.1f} GiB，"
                f"可用 {combined_free / 1024**3:.1f} GiB），已中止。\n"
                f"    選項：1) 清出空間  2) 先跑 --limit 50  3) 加 --force 硬跑"
            )
            return 1
        if combined_free < needed:
            print(f"\n {WARN} 空間不足但你指定了 --force，繼續執行。")
    if not args.dry_run:
        cfg.paths.ensure()

    results: list[FileResult] = []
    counters = {"ok": 0, "rejected": 0, "skipped": 0, "error": 0}
    show_group_tag = cfg.group_by == "folder"

    def _tick(res: FileResult, idx: int) -> None:
        counters[res.status] = counters.get(res.status, 0) + 1
        mark = MARK.get(res.status, "?")
        if res.status == "ok" and not args.verbose:
            if idx % 25 == 0 or idx == len(targets):
                print(
                    f"  ... {idx}/{len(targets)}  "
                    f"{MARK['ok']}{counters['ok']} {MARK['rejected']}{counters['rejected']}"
                    f" {MARK['error']}{counters['error']}"
                )
            return
        name = Path(res.src_path).name
        dur = f"{res.duration_sec}s" if res.duration_sec else "-"
        # 多角色專案一定要顯示群組，否則你看到 "noise_short.wav 被淘汰"
        # 會不知道是哪个角色被淘汰
        tag = f"[{res.group}] " if show_group_tag else ""
        print(f"  {mark} [{idx}/{len(targets)}] {tag}{name}  {dur}  {res.reason}")

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_one, src, cfg, args): src for src in targets
            }
            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    res = fut.result()
                except Exception as exc:  # 單檔爆掉不拖垮整批
                    res = FileResult(
                        status="error",
                        reason="未預期例外",
                        src_path=rel(futures[fut], root),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results.append(res)
                _tick(res, idx)
    else:
        for idx, src in enumerate(targets, 1):
            try:
                res = process_one(src, cfg, args)
            except Exception as exc:
                res = FileResult(
                    status="error",
                    reason="未預期例外",
                    src_path=rel(src, root),
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(res)
            _tick(res, idx)

    # ---- 統計 + 報表 ----
    results.sort(key=lambda r: r.src_path)
    _write_manifest(results, cfg, args)

    total_dur = 0.0
    for r in results:
        if r.status == "ok" and r.duration_sec:
            try:
                total_dur += float(r.duration_sec)
            except ValueError:
                pass

    print("\n" + "=" * 72)
    print(" 結果統計")
    print("=" * 72)
    print(f"  {MARK['ok']} 合格      {counters.get('ok', 0):>6}")
    print(f"  {MARK['rejected']} 淘汰      {counters.get('rejected', 0):>6}")
    print(f"  {MARK['skipped']} 略過      {counters.get('skipped', 0):>6}")
    print(f"  {MARK['error']} 錯誤      {counters.get('error', 0):>6}")
    print(f"  合格素材總長 {total_dur / 60:.1f} 分鐘")

    # 多角色專案：逐角色列出素材量。單一角色時這張表只會有一列 default，故省略。
    per_group = _per_group_stats(results)
    show_groups = cfg.group_by == "folder" and len(per_group) >= 1
    if show_groups and (len(per_group) > 1 or next(iter(per_group), "default") != "default"):
        print("\n 逐角色素材量（Phase 4 會依此分開打標籤、Phase 5 分開選參考音訊）：")
        print(
            f"   {pad_display('群組', 22)}{'合格':>6}{'淘汰':>7}{'錯誤':>7}{'分鐘':>9}"
        )
        for name, st in sorted(per_group.items(), key=lambda kv: -kv[1]["seconds"]):
            print(
                f"   {pad_display(name, 22)}{st['ok']:>6.0f}{st['rejected']:>7.0f}"
                f"{st['error']:>7.0f}{st['seconds'] / 60:>9.1f}"
            )
        thin = [n for n, st in per_group.items() if st["seconds"] < 180]
        if thin:
            print(
                f"\n   {WARN} 以下群組素材 < 3 分鐘，建議補齊或考慮合併角色："
                f"{'、'.join(thin[:6])}"
            )

    # Quality Over Quantity 原則的即時提醒
    minutes = total_dur / 60.0
    print()
    if minutes < 5:
        print(f" {WARN} 目前只有 {minutes:.1f} 分鐘合格素材。GPT-SoVITS 微調建議 ≥ 10 分鐘，")
        print("   最佳區間 15~60 分鐘「極乾淨」音訊；請優先補素材或放寬來源，而非硬練。")
    elif minutes <= 60:
        print(f" {OK} {minutes:.1f} 分鐘落在建議區間（15~60 分鐘），是微調的最佳區段。")
    else:
        print(f" {INFO} {minutes:.1f} 分鐘偏多。建議先挑 15~60 分鐘『最乾淨、情感最豐富』的子集，")
        print("   雜音素材寧可不用——15 分鐘神級乾淨音訊勝過 2 小時雜音數據。")

    if counters.get("error"):
        print(f"\n {BAD} 錯誤明細（前 10 筆）：")
        for r in results:
            if r.status == "error":
                print(f"   {r.src_path}：{r.error or r.reason}")
    print(f"\n 明細報表：{rel(cfg.manifest_path, root)}")
    return 0 if counters.get("error", 0) == 0 else 2


def _per_group_stats(results: list[FileResult]) -> dict[str, dict[str, float]]:
    """依群組（角色）彙總：合格 / 淘汰 / 錯誤 數量與合格秒數。"""
    stats: dict[str, dict[str, float]] = {}
    for r in results:
        st = stats.setdefault(r.group or "default", {"ok": 0, "rejected": 0, "skipped": 0, "error": 0, "seconds": 0.0})
        if r.status in st:
            st[r.status] += 1
        if r.status == "ok" and r.duration_sec:
            try:
                st["seconds"] += float(r.duration_sec)
            except ValueError:
                pass
    return stats


def _write_manifest(results: list[FileResult], cfg: PipelineConfig, args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in results if cfg.manifest_include_rejected or r.status == "ok"]
    # utf-8-sig：Windows Excel 直接雙擊開啟不亂碼
    with cfg.manifest_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.row())

    # 另外輸出一份 JSON 摘要，方便 Phase 2 直接讀（人讀 CSV、機器讀 JSON）
    summary = {
        "phase": 1,
        "config_file": rel(cfg.source_file, repo_root()),
        "target": asdict(cfg.target) | {"subtype": cfg.target.subtype},
        "duration_filter": asdict(cfg.duration_filter),
        "quality_gate": asdict(cfg.quality_gate),
        "group_by": cfg.group_by,
        "counts": {
            s: sum(1 for r in results if r.status == s)
            for s in ("ok", "rejected", "skipped", "error")
        },
        "total_ok_seconds": round(
            sum(float(r.duration_sec) for r in results if r.status == "ok" and r.duration_sec), 3
        ),
        "groups": _per_group_stats(results),
    }
    (cfg.paths.reports_dir / "phase1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 子命令：audition（人工驗收輔助）
# ---------------------------------------------------------------------------
def cmd_audition(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    """
    抽取合格素材樣本，接成單一「試聽帶」，並輸出對照清單。

    這是 Phase 1 驗收清單裡「人工試聽隨機 10 個檔案」那一步的工具化版本。
    自動化能抓時長／電平／削波，但「這句話有沒有感情、有沒有被截斷、
    是不是同一個人在講」只有耳朵知道。
    """
    root = repo_root()
    groups = list_candidates(cfg.paths.standardized_dir)

    if not groups:
        print(f"{WARN} {rel(cfg.paths.standardized_dir, root)} 內找不到任何 .wav。")
        print("   請先執行：python scripts/phase1_prep.py convert")
        return 1

    total_files = sum(len(v) for v in groups.values())
    include = [g for g in args.groups.split(",") if g] if args.groups else None
    if include:
        unknown = [g for g in include if g not in groups]
        if unknown:
            print(f"{WARN} 指定的群組不存在：{'、'.join(unknown)}")
            print(f"   現有群組：{'、'.join(sorted(groups))}")
            return 1

    picked = plan_samples(
        groups, per_group=args.per_group, seed=args.seed, include_groups=include
    )
    if not picked:
        print(f"{WARN} 沒有抽到任何樣本（per_group={args.per_group}）。")
        return 1

    print("=" * 72)
    print(" Phase 1 人工試聽抽樣")
    print("=" * 72)
    print(f" 合格素材：{total_files} 個檔案，{len(groups)} 個群組")
    for name in sorted(groups):
        mark = "✔" if not include or name in include else "·"
        print(f"   {mark} {pad_display(name, 24)} {len(groups[name]):>6} 個")
    print(f"\n 抽樣：每群組 {args.per_group} 個，共 {len(picked)} 個（亂數種子 {args.seed}，可重現）")
    print(f" 響度對齊：{'開（試聽帶 RMS → ' + str(args.target_rms) + ' dBFS）' if not args.no_align else '關'}")
    print("=" * 72)

    out_name = filesystem_safe(f"audition_{len(picked)}samples_seed{args.seed}.wav")
    out_path = cfg.paths.reports_dir / out_name
    list_path = cfg.paths.reports_dir / out_name.replace(".wav", ".md")

    if args.dry_run:
        print(" DRY-RUN：以下是被抽中的樣本（未產生試聽帶）")
        for idx, (group, path) in enumerate(picked, 1):
            print(f"   {idx:>3}. [{group}] {rel(path, root)}")
        return 0

    plan = build_reel(
        picked,
        out_path,
        sample_rate=cfg.target.sample_rate,
        gap_seconds=args.gap,
        align_rms=not args.no_align,
        target_dbfs=args.target_rms,
    )
    plan.write_list(list_path)

    failed = [s for s in plan.samples if s.error]
    readable = plan.count - len(failed)

    print(f"\n {OK} 試聽帶：{rel(out_path, root)}")
    print(f"   {readable} 段，總長 {plan.total_seconds / 60:.1f} 分鐘，每段之間靜音 {args.gap}s")
    print(f" {OK} 對照清單：{rel(list_path, root)}")
    if failed:
        print(f" {WARN} 有 {len(failed)} 個樣本無法讀取（清單中已標示）：")
        for s in failed[:5]:
            print(f"   {s.path.name}：{s.error}")

    print("\n 建議試聽流程（Phase 1 驗收第 3 項）：")
    print("   1. 從頭順聽一次，聽到問題就記下『第幾段』")
    print("   2. 用對照清單回查檔名，並在 02_wav24_48k/ 內找到它")
    print("   3. 逐項確認：① 底噪／電流聲 ② 截頭截尾 ③ 同一人 ④ 情感多樣性")
    print("   4. 若某類問題集中在特定群組 → 該角色的素材需要重挑或重解包")
    return 0


# ---------------------------------------------------------------------------
# 子命令：tune（門檻調校分析）
# ---------------------------------------------------------------------------
def cmd_tune(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    """
    讀 manifest 分析篩選結果，給出「該不該動門檻、往哪個方向動」的具體建議。

    淘汰率本身沒有意義 —— 必須知道淘汰的是什麼。這個子命令把
    「1.2 秒的短台詞被丟掉」和「0.3 秒的喘氣聲被丟掉」區分開來。
    """
    root = repo_root()
    manifest = cfg.manifest_path

    if not manifest.is_file():
        print(f"{WARN} 找不到 manifest：{rel(manifest, root)}")
        print("   請先執行：python scripts/phase1_prep.py convert")
        return 1

    rows = read_manifest(manifest)
    if not rows:
        print(f"{WARN} manifest 是空的：{rel(manifest, root)}")
        return 1

    report = analyze(
        rows,
        min_seconds=cfg.duration_filter.min_seconds,
        max_seconds=cfg.duration_filter.max_seconds,
        peak_dbfs_max=cfg.quality_gate.peak_dbfs_max,
        min_rms_dbfs=cfg.quality_gate.min_rms_dbfs,
    )

    print(render_report(
        report,
        min_seconds=cfg.duration_filter.min_seconds,
        max_seconds=cfg.duration_filter.max_seconds,
        peak_dbfs_max=cfg.quality_gate.peak_dbfs_max,
        min_rms_dbfs=cfg.quality_gate.min_rms_dbfs,
    ))

    if not args.dry_run:
        out = cfg.paths.reports_dir / "phase1_tune.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report_as_dict(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n 機器可讀版本：{rel(out, root)}")

    return 0


# ---------------------------------------------------------------------------
# 子命令：report（Phase 1 綜合報告）
# ---------------------------------------------------------------------------
def cmd_report(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    """
    把 manifest / tune / audition / scan / doctor 的資訊收斂成一份 Markdown。

    用途有兩個：
      1. 你隔幾天回來時，不用重新拼湊「這批素材到底能不能用」
      2. Phase 1 → Phase 2 的交接文件：驗收條件逐項打勾
    """
    root = repo_root()

    rep = collect(
        paths=cfg.paths,
        duration_filter=cfg.duration_filter,
        quality_gate=cfg.quality_gate,
        scan_extensions=cfg.scan_extensions,
        do_scan=not args.no_scan,
    )

    md = render(
        rep,
        paths=cfg.paths,
        duration_filter=cfg.duration_filter,
        quality_gate=cfg.quality_gate,
        target=cfg.target,
        repo_root=root,
        ascii_only=not UNICODE_OK,
    )

    out_path = cfg.paths.reports_dir / "PHASE1_REPORT.md"
    json_path = cfg.paths.reports_dir / "PHASE1_REPORT.json"

    # 主控台只印摘要 —— 完整內容在檔案裡，避免洗版
    print("=" * 72)
    print(" Phase 1 綜合報告")
    print("=" * 72)

    if rep.has_manifest:
        print(f" 合格素材：{rep.ok_count} 個檔案，{rep.ok_minutes:.1f} 分鐘")
    else:
        print(f" {WARN} 尚未產生 manifest（convert 還沒跑）")

    if rep.scan is not None and rep.scan.total_files:
        print(
            f" 原始素材：{rep.scan.total_files} 個檔案"
            f"（可直接解碼 {rep.scan.direct}，需解包 {rep.scan.need_unpack}）"
        )
    if rep.audition_reels:
        print(f" 試聽帶：{len(rep.audition_reels)} 份")

    passed = sum(1 for c in rep.checks if c.passed)
    print(f"\n 驗收條件：{passed}/{len(rep.checks)} 項達標")
    for check in rep.checks:
        print(f"   {OK if check.passed else '·'} {check.label} — {check.detail}")

    if args.dry_run:
        print(f"\n DRY-RUN：未寫出報告檔案（完整內容如下）\n")
        print(md)
        return 0

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    json_path.write_text(render_json(rep), encoding="utf-8")

    print(f"\n {OK} 報告：{rel(out_path, root)}")
    print(f" {OK} 機器可讀：{rel(json_path, root)}")

    if rep.all_checks_passed:
        print(f"\n {OK} 全部驗收條件達標 —— 可以進入 Phase 2")
        print("    但請先確認：你真的從頭聽完試聽帶了嗎？（工具無法代替你的耳朵）")
    else:
        pending = [c for c in rep.checks if not c.passed]
        print(f"\n {WARN} 還有 {len(pending)} 項未達標，詳見報告 §5：")
        for check in pending:
            print(f"   · {check.label} — {check.detail}")
    return 0


# ---------------------------------------------------------------------------
# 子命令：selftest（用合成素材驗證工具鏈本身能不能動）
# ---------------------------------------------------------------------------
def cmd_selftest(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    """
    用合成音訊跑一次真正的流程，回答「這套工具在我的機器上能不能動」。

    不需要任何素材、不需要外部工具、也不會碰你真正的 data/ ——
    它在暫存區建立臨時 repo，跑完就刪。
    """
    print("=" * 72)
    print(" Phase 1 自我測試（合成素材，不需要你的素材）")
    print("=" * 72)
    print(" 流程：產生合成素材 -> convert -> 驗證規格與篩選 -> audition/tune/report")
    print(" 隔離：臨時工作區，跑完刪除；不會碰 data/")
    print("=" * 72)

    report = run_selftest(keep_workspace=args.keep_workspace)

    print("\n" + "=" * 72)
    print(" 檢查結果")
    print("=" * 72)
    for check in report.checks:
        mark = OK if check.passed else BAD
        detail = f"  -- {check.detail}" if check.detail else ""
        print(f"  {mark} {check.label}{detail}")
        if not check.passed and check.hint:
            print(f"      -> {check.hint}")

    if report.notes:
        print(f"\n {INFO} 提醒（不影響自我測試結果）：")
        for note in report.notes[:6]:
            print(f"   - {note}")

    passed = sum(1 for c in report.checks if c.passed)
    print(f"\n 通過 {passed}/{len(report.checks)} 項")

    if report.passed:
        print(f"\n {OK} 工具鏈在你的機器上可以正常運作。")
        print("    下一步：把遊戲語音放進 data/00_raw/，然後跑 scan")
    else:
        print(f"\n {BAD} 有 {len(report.failed)} 項失敗 -- 請先修正再處理真實素材。")
        for check in report.failed:
            print(f"   - {check.label}：{check.detail}")
        if args.keep_workspace and report.workspace:
            print(f"\n   臨時工作區保留在：{report.workspace}")
    return 0 if report.passed else 1

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase1_prep.py",
        description="vOS-VirtualVoice Phase 1：遊戲語音解包 + 24bit/48kHz 標準化 + 時長篩選",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="設定檔路徑（預設 configs/pipeline.local.json，不存在則用 pipeline.default.json）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dry-run", action="store_true", help="只報告、不寫任何檔案")
        p.add_argument("--limit", type=int, default=0, help="最多處理 N 個檔案（先小量驗證）")
        p.add_argument("--workers", type=int, default=0, help="平行解碼執行緒數（0=自動，預設）")
        p.add_argument("--overwrite", action="store_true", help="輸出已存在時仍重做")
        p.add_argument("--no-quality-gate", action="store_true", help="停用品質關卡，只做時長篩選")
        p.add_argument(
            "--max-subsongs",
            type=int,
            default=0,
            help="覆寫單一容器的 subsong 上限（0=用設定檔值；unpack 用）",
        )
        p.add_argument(
            "--no-probe", action="store_true", help="解包前不探測 subsong 數量（較快但無安全上限）"
        )
        p.add_argument("-v", "--verbose", action="store_true", help="逐檔輸出（預設每 25 檔一摘要）")

        p.add_argument(
            "--force",
            action="store_true",
            help="略過磁碟空間預檢（空間不足時仍強制執行）",
        )
        p.add_argument(
            "--no-fallback",
            action="store_true",
            help="scan 時不用 vgmstream 判定未知格式（較快，適合檔案數上萬時）",
        )

    for name, help_text in (
        ("doctor", "檢查 Python 依賴 / 外部工具 / 資料夾 / 目標格式"),
        ("scan", "嗅探素材真實格式分布（不寫檔，最重要的一步）"),
        ("unpack", "用 vgmstream-cli 解包遊戲容器到 01_unpacked/"),
        ("convert", "標準化為 24bit/48kHz WAV + 時長篩選 + 品質關卡"),
        ("audition", "抽取合格素材樣本 → 產生試聽帶與對照清單（人工驗收用）"),
        ("tune", "分析 manifest，建議該不該調整篩選門檻（淘汰率異常時用）"),
        ("report", "產生 Phase 1 綜合報告（驗收清單 + 下一步指令，交接用）"),
        ("selftest", "用合成素材驗證工具鏈本身能不能動（不需要你的素材）"),
        ("all", "依序執行 scan → unpack → convert"),
    ):
        sp = sub.add_parser(name, help=help_text)
        add_common(sp)
        if name == "audition":
            sp.add_argument("--per-group", type=int, default=5, help="每個群組抽幾個（預設 5）")
            sp.add_argument("--seed", type=int, default=20240601, help="亂數種子（固定才能重現）")
            sp.add_argument("--gap", type=float, default=0.9, help="每段之間的靜音秒數")
            sp.add_argument("--groups", default="", help="只抽指定群組，逗號分隔（例：凛,桜）")
            sp.add_argument("--no-align", action="store_true", help="不做響度對齊")
            sp.add_argument(
                "--target-rms", type=float, default=-20.0, help="試聽帶的目標 RMS（dBFS）"
            )
        if name == "report":
            sp.add_argument(
                "--no-scan",
                action="store_true",
                help="跳過 00_raw 的格式掃描（大目錄時較快）",
            )
        if name == "selftest":
            sp.add_argument(
                "--keep-workspace",
                action="store_true",
                help="保留臨時工作區供事後檢查（預設跑完刪除）",
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"{BAD} 設定檔錯誤：{exc}", file=sys.stderr)
        return 2

    # --no-quality-gate / --max-subsongs 以 CLI 覆寫設定檔（dataclass frozen，用 replace）
    if getattr(args, "no_quality_gate", False):
        from dataclasses import replace

        cfg = replace(cfg, quality_gate=replace(cfg.quality_gate, enabled=False))
    if getattr(args, "max_subsongs", 0):
        from dataclasses import replace

        cfg = replace(cfg, unpack_max_subsongs=args.max_subsongs)

    handlers = {
        "doctor": cmd_doctor,
        "scan": cmd_scan,
        "unpack": cmd_unpack,
        "convert": cmd_convert,
        "audition": cmd_audition,
        "tune": cmd_tune,
        "report": cmd_report,
        "selftest": cmd_selftest,
        "all": lambda c, a: _cmd_all(c, a, parser),
    }
    try:
        return handlers[args.command](cfg, args)
    except KeyboardInterrupt:
        print(f"\n{WARN} 使用者中斷。已完成的檔案不受影響，可直接重跑（具冪等性）。")
        return 130


def _cmd_all(cfg: PipelineConfig, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    rc = cmd_doctor(cfg, args)
    if rc != 0:
        print(f"\n{BAD} 環境未就緒，中止 all。請先修正上述問題。")
        return rc
    print()
    rc = cmd_scan(cfg, args)
    if rc != 0:
        return rc
    print()
    rc = cmd_unpack(cfg, args)
    if rc != 0:
        print(f"\n{WARN} 解包有部分失敗，仍繼續進行已解出檔案的標準化。")
    print()
    return cmd_convert(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
