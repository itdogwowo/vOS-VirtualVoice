"""
vOS-VirtualVoice — 自我測試 (Phase 1)

為什麼需要這個
--------------
這套工具鏈依賴 Python 套件（soundfile / soxr / scipy）、外部工具
（vgmstream-cli / ffmpeg）與一堆跨平台細節（路徑、編碼、行尾、暫存目錄）。
任何一環在你的機器上不對，就會在下載素材、整理資料夾之後才發現 ——
那是最不划算的發現時機。

`selftest` 用**合成音訊**跑一次真正的流程（convert / audition / tune / report），
在幾秒鐘內回答一個問題：**這套工具在我的機器上能不能動？**

三個刻意的設計
--------------
1. **完全隔離**：在系統暫存區建立臨時 repo，用 VOS_REPO_ROOT 把 CLI 關進去，
   絕不碰你真正的 data/。跑完就刪。
2. **涵蓋每一條篩選規則**：合成的素材刻意包含合格、過短、過長、訊號過弱四種，
   所以它同時驗證「該過的過、該擋的擋」—— 不是只驗證 happy path。
3. **不依賴外部工具**：vgmstream / ffmpeg 缺席也能跑完（Phase 1 對純 WAV 素材
   本來就不需要它們）。缺少的外部工具會列成「提醒」而不是「失敗」。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# 讓本模組無論從哪裡被匯入都能找到同層的 _audioio / config / diag。
#
# 為什麼需要：CLI（phase1_prep.py）執行時已經把自己的目錄放進 sys.path，
# 所以直接 import 沒問題；但被 pytest 或其他工具直接匯入時就沒有那層保障，
# 會炸在 `ModuleNotFoundError: No module named '_audioio'` ——
# 一個「只有在測試裡才會發生」的失敗，最容易被誤判成環境問題。
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: 自我測試工作區的目錄名前綴。cleanup 會**據此拒絕刪除任何不符的目錄** ——
#: 見 cleanup_workspace 的說明。
WORKSPACE_PREFIX = "vos_selftest_"

#: 合成素材的規格：{檔名: (時長秒, 取樣率, 振幅, 聲道數, 期望結果)}
#:
#: 期望結果刻意涵蓋每一條規則，這樣 selftest 同時驗證「該過的過、該擋的擋」。
FIXTURES: dict[str, tuple[float, int, float, int, str]] = {
    # 合格：需升採樣（44.1k → 48k）
    "ok_short_44k.wav": (2.5, 44_100, 0.5, 1, "ok"),
    # 合格：立體聲需轉單聲道
    "ok_stereo.wav": (4.0, 48_000, 0.5, 2, "ok"),
    # 淘汰：過短（喘氣／單字哼聲）
    "reject_too_short.wav": (0.7, 48_000, 0.5, 1, "rejected"),
    # 淘汰：過長
    "reject_too_long.wav": (16.5, 48_000, 0.5, 1, "rejected"),
    # 淘汰：訊號過弱（耳語級，預設 RMS 門檻會擋下）
    "reject_too_quiet.wav": (3.0, 48_000, 0.003, 1, "rejected"),
}


@dataclass
class Check:
    """單一檢查項。"""

    label: str
    passed: bool
    detail: str = ""
    hint: str = ""


@dataclass
class SelfTestReport:
    """自我測試結果。"""

    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    workspace: Path | None = None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def add(self, label: str, passed: bool, detail: str = "", hint: str = "") -> None:
        self.checks.append(Check(label=label, passed=passed, detail=detail, hint=hint))


# ---------------------------------------------------------------------------
# 合成素材
# ---------------------------------------------------------------------------
def synth_voice(
    seconds: float, sample_rate: int, amp: float, channels: int
) -> list[list[float]]:
    """
    合成一段「像人聲」的測試訊號（基頻 + 諧波 + 淡入淡出）。

    需要 numpy —— 它是必要依賴，缺席時 selftest 應該要能明確說出來，
    而不是丟一個看不懂的 ImportError。
    """
    import numpy as np

    frames = max(1, int(seconds * sample_rate))
    t = np.arange(frames, dtype=np.float64) / sample_rate
    mono = np.zeros(frames, dtype=np.float64)
    for k, gain in enumerate((1.0, 0.5, 0.33, 0.25, 0.2), start=1):
        mono += gain * np.sin(2 * np.pi * 220.0 * k * t)
    peak = float(np.max(np.abs(mono))) or 1.0
    mono = mono / peak * amp

    # 5 ms 淡入淡出，避免邊緣爆音
    fade = min(int(0.005 * sample_rate), frames // 4)
    if fade > 1:
        mono[:fade] *= np.linspace(0, 1, fade)
        mono[-fade:] *= np.linspace(1, 0, fade)

    if channels == 1:
        return [mono.tolist()]
    return [mono.tolist(), (mono * 0.8).tolist()]


def write_fixtures(raw_dir: Path) -> int:
    """把合成素材寫成真正的 WAV 檔。回傳寫入的檔案數。"""
    import numpy as np
    import soundfile as sf

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (seconds, sr, amp, channels, _expected) in FIXTURES.items():
        data = synth_voice(seconds, sr, amp, channels)
        if channels == 1:
            array = np.asarray(data[0], dtype=np.float32)
        else:
            array = np.stack([np.asarray(ch, dtype=np.float32) for ch in data], axis=1)
        sf.write(str(raw_dir / name), array, sr, subtype="PCM_16", format="WAV")
    return len(FIXTURES)


# ---------------------------------------------------------------------------
# 沙箱
# ---------------------------------------------------------------------------
def make_workspace(base: Path | None = None) -> Path:
    """
    建立臨時 repo 骨架（configs + data 五個資料夾）。

    刻意不用 mkdtemp：它建立的目錄帶 0700 權限，在受限環境會讓
    子程序寫不進去（詳見 scripts/_audioio.py 的說明）。
    """
    if base is None:
        base = Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)

    root = base / f"{WORKSPACE_PREFIX}{uuid.uuid4().hex[:10]}"
    root.mkdir()
    (root / "configs").mkdir()
    for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
        (root / "data" / sub).mkdir(parents=True)

    # 用 repo 的預設設定當基礎（確保測到的是真實門檻，不是我另外寫死的值）
    from config import default_config_path

    source = default_config_path()
    data = json.loads(source.read_text(encoding="utf-8"))
    data["grouping"] = {"by": "none"}
    (root / "configs" / "pipeline.local.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def is_workspace(path: Path) -> bool:
    """
    判斷一個路徑「像不像」自我測試工作區。

    這是破壞性操作的安全閘門：目錄名必須帶 WORKSPACE_PREFIX。
    """
    try:
        return path.is_dir() and path.name.startswith(WORKSPACE_PREFIX)
    except OSError:
        return False


def cleanup_workspace(root: Path) -> None:
    """
    清理臨時 repo。**只刪除長得像自我測試工作區的目錄。**

    為什麼要這麼嚴格：這個函式會執行 `shutil.rmtree` —— 一旦參數指錯地方，
    後果是整棵目錄樹消失，而且不會有任何錯誤訊息（`ignore_errors=True`）。
    本專案真的發生過一次工作目錄被清空的事故，所以這裡加上兩道閘門：

      1. 目錄名必須以 `vos_selftest_` 開頭
      2. 必須真的是目錄

    不符就**什麼都不做**。寧可留下垃圾目錄，也不要冒刪錯的風險。
    """
    try:
        root = Path(root)
    except (TypeError, ValueError):
        return

    if not is_workspace(root):
        return  # 不是工作區 → 一律不碰

    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 執行
# ---------------------------------------------------------------------------
def run_cli(args: list[str], repo_root: Path, timeout: int = 300):
    """在沙箱 repo 內執行 CLI（與其他測試共用同一套隔離機制）。"""
    from config import ENV_REPO_ROOT
    from config import repo_root as real_repo_root

    env = dict(os.environ)
    env[ENV_REPO_ROOT] = str(repo_root)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    script = real_repo_root() / "scripts" / "phase1_prep.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def run_selftest(*, keep_workspace: bool = False, quiet: bool = False) -> SelfTestReport:
    """
    跑完整自我測試。

    Args:
        keep_workspace: 保留臨時 repo 供事後檢查（預設跑完就刪）。
        quiet: 不即時輸出進度（給測試用）。
    """
    report = SelfTestReport()

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    # ---- 1. 依賴檢查 ----
    try:
        import numpy  # noqa: F401
        import soundfile as sf

        report.add(
            "音訊 I/O 依賴（soundfile + numpy）",
            True,
            f"libsndfile {sf.__libsndfile_version__}",
        )
    except Exception as exc:
        report.add(
            "音訊 I/O 依賴（soundfile + numpy）",
            False,
            f"{type(exc).__name__}: {exc}",
            "pip install -r requirements.txt",
        )
        return report  # 沒有這兩個什麼都做不了

    from _audioio import HAS_SCIPY, HAS_SOXR

    report.add(
        "重採樣後端（soxr 或 scipy）",
        HAS_SOXR or HAS_SCIPY,
        "soxr（VHQ）" if HAS_SOXR else ("scipy polyphase" if HAS_SCIPY else "兩者皆無"),
        "pip install soxr scipy",
    )
    if not (HAS_SOXR or HAS_SCIPY):
        return report  # 沒有重採樣就無法標準化

    # ---- 2. 建立沙箱 ----
    say("\n[1/5] 建立臨時工作區…")
    try:
        root = make_workspace()
    except OSError as exc:
        report.add("建立臨時工作區", False, str(exc), "確認系統暫存目錄可寫")
        return report
    report.workspace = root
    say(f"      {root}")

    try:
        raw = root / "data" / "00_raw"

        # ---- 3. 合成素材 ----
        say("[2/5] 產生合成素材…")
        try:
            count = write_fixtures(raw)
            report.add(
                "產生合成素材",
                count == len(FIXTURES),
                f"{count} 個檔案（涵蓋合格／過短／過長／過弱）",
            )
        except Exception as exc:
            report.add("產生合成素材", False, f"{type(exc).__name__}: {exc}")
            return report

        # ---- 4. convert ----
        say("[3/5] 執行 convert（標準化 + 篩選）…")
        proc = run_cli(["convert", "--overwrite"], root)
        report.add(
            "convert 執行成功",
            proc.returncode == 0,
            f"exit {proc.returncode}",
            "看下方輸出，或先跑 doctor 檢查依賴",
        )
        if proc.returncode != 0:
            report.notes.append(proc.stdout[-800:])
            report.notes.append(proc.stderr[-400:])
            return report

        # ---- 5. 驗證產物 ----
        say("[4/5] 驗證產物規格與篩選規則…")
        manifest = root / "data" / "_reports" / "manifest_phase1.csv"
        rows: list[dict[str, str]] = []
        if manifest.is_file():
            import csv

            with manifest.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
        by_name = {Path(r.get("src_path", "")).name: r for r in rows}

        # 每個素材的預期結果都要對上
        mismatches: list[str] = []
        for name, (_s, _sr, _a, _c, expected) in FIXTURES.items():
            row = by_name.get(name)
            if row is None:
                mismatches.append(f"{name} 不在 manifest 裡")
                continue
            if row["status"] != expected:
                mismatches.append(
                    f"{name} 預期 {expected}，實際 {row['status']}（{row['reason']}）"
                )
        report.add(
            "篩選規則正確（該過的過、該擋的擋）",
            not mismatches,
            f"{len(rows)} 筆紀錄、{len(mismatches)} 筆不符",
            "；".join(mismatches[:3]),
        )

        # 產物規格
        import soundfile as sf2

        out_dir = root / "data" / "02_wav24_48k"
        wavs = sorted(out_dir.rglob("*.wav"))
        expected_ok = sum(1 for f in FIXTURES.values() if f[4] == "ok")
        report.add(
            "合格區檔案數正確",
            len(wavs) == expected_ok,
            f"{len(wavs)} 個（預期 {expected_ok}）",
        )

        bad_spec: list[str] = []
        for wav in wavs:
            info = sf2.info(str(wav))
            if (info.samplerate, info.subtype, info.channels) != (48_000, "PCM_24", 1):
                bad_spec.append(f"{wav.name}: {info.samplerate}/{info.subtype}/{info.channels}")
        report.add(
            "產物規格為 48 kHz / 24-bit / 單聲道",
            not bad_spec and bool(wavs),
            f"{len(wavs)} 個檔案" if not bad_spec else "；".join(bad_spec[:3]),
        )

        # 淘汰區必須是複本（原檔還在）
        rejected = list((root / "data" / "99_rejected").rglob("*.wav"))
        originals_intact = all((raw / f.name).is_file() for f in rejected)
        report.add(
            "淘汰檔複製到 99_rejected 且原檔保留",
            len(rejected) == 3 and originals_intact,
            f"淘汰區 {len(rejected)} 個、原檔完整={originals_intact}",
            "原始素材永不搬移是刻意的設計",
        )

        # ---- 6. audition / tune / report ----
        say("[5/5] 執行 audition / tune / report…")
        aud = run_cli(["audition", "--per-group", "2"], root)
        reels = list((root / "data" / "_reports").glob("audition_*.wav"))
        report.add(
            "audition 產生試聽帶",
            aud.returncode == 0 and len(reels) == 1,
            f"exit {aud.returncode}、{len(reels)} 份試聽帶",
        )

        tune = run_cli(["tune"], root)
        tune_json = root / "data" / "_reports" / "phase1_tune.json"
        report.add(
            "tune 產生分析結果",
            tune.returncode == 0 and tune_json.is_file(),
            f"exit {tune.returncode}",
        )

        rep = run_cli(["report", "--no-scan"], root)
        rep_md = root / "data" / "_reports" / "PHASE1_REPORT.md"
        report.add(
            "report 產生綜合報告",
            rep.returncode == 0 and rep_md.is_file(),
            f"exit {rep.returncode}",
        )

        # ---- 7. 外部工具（只提醒，不算失敗）----
        from diag import diagnose_all

        statuses = diagnose_all(None, probe=False) if _has_diag() else []
        for status in statuses:
            if not status.available:
                report.notes.append(
                    f"{status.name} 未安裝 —— Phase 1 對純 WAV 素材不需要它，"
                    f"但處理解包與冷門格式時會用到（見 docs/PHASE1-INGEST.md §1）"
                )
        return report
    finally:
        if not keep_workspace and report.workspace is not None:
            cleanup_workspace(report.workspace)


def _has_diag() -> bool:
    """diag 模組是否可用（它是選用的診斷增強，缺席不該讓 selftest 失敗）。"""
    try:
        import diag  # noqa: F401

        return True
    except Exception:
        return False


__all__ = [
    "FIXTURES",
    "WORKSPACE_PREFIX",
    "Check",
    "SelfTestReport",
    "cleanup_workspace",
    "is_workspace",
    "make_workspace",
    "run_cli",
    "run_selftest",
    "synth_voice",
    "write_fixtures",
]
