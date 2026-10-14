"""
vOS-VirtualVoice — Phase 1 綜合報告 (收尾產物)

為什麼需要這份報告
------------------
Phase 1 跑完後，資訊散在六個地方：manifest CSV、summary JSON、tune JSON、
audition 清單、掃描結果、doctor 輸出。等到 Phase 2 要接手時（或你隔幾天回來），
你得重新拼湊「這批素材到底能不能用」。

這份報告把六個來源收斂成**一份可讀的 Markdown**，回答三個問題：

  1. 我現在有什麼？（素材量、格式、篩選結果、逐角色分布）
  2. 還缺什麼才能進 Phase 2？（驗收條件逐項打勾）
  3. 下一步該打什麼指令？

它同時是 Phase 1 → Phase 2 的交接文件：Phase 2 的輸入需求寫在這裡，
你只要對照就知道能不能往下走。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import config
from _audioio import probe_backends
from audition import list_candidates
from formats import sniff_format
from tune import ManifestRow, TuneReport, analyze, read_manifest

#: Phase 1 驗收標準（與 docs/PHASE1-INGEST.md §7 一致）
MIN_TOTAL_MINUTES = 15.0
MIN_PER_GROUP_MINUTES = 3.0
RECOMMENDED_MIN_MINUTES = 10.0
RECOMMENDED_MAX_MINUTES = 60.0


@dataclass
class ScanSummary:
    """掃描結果摘要（不寫檔，只做統計）。"""

    total_files: int = 0
    total_bytes: int = 0
    by_format: dict[str, int] = field(default_factory=dict)
    need_unpack: int = 0
    direct: int = 0
    ext_mismatch: int = 0
    #: 已解包完成的容器數（01_unpacked/<相對路徑>/<容器名>/ 內有 .wav）
    already_unpacked: int = 0
    #: 仍未解包的容器數 = need_unpack - already_unpacked
    pending_unpack: int = 0


@dataclass
class CheckItem:
    """一項驗收條件的檢查結果。"""

    label: str
    passed: bool
    detail: str


@dataclass
class Phase1Report:
    """Phase 1 的完整狀態。"""

    manifest_rows: list[ManifestRow] = field(default_factory=list)
    tune: TuneReport | None = None
    scan: ScanSummary | None = None
    audition_reels: list[Path] = field(default_factory=list)
    audition_lists: list[Path] = field(default_factory=list)
    backends: list[tuple[str, bool, str]] = field(default_factory=list)
    checks: list[CheckItem] = field(default_factory=list)
    generated_at: str = ""

    # -- 便利屬性 ---------------------------------------------------------
    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.manifest_rows if r.status == "ok")

    @property
    def ok_minutes(self) -> float:
        seconds = sum(
            r.duration for r in self.manifest_rows if r.status == "ok" and r.duration
        )
        return seconds / 60.0

    @property
    def has_manifest(self) -> bool:
        return bool(self.manifest_rows)

    @property
    def all_checks_passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def group_minutes(self) -> dict[str, float]:
        """各群組的合格時長（分鐘）。"""
        out: dict[str, float] = {}
        for row in self.manifest_rows:
            if row.status == "ok" and row.duration:
                out[row.group] = out.get(row.group, 0.0) + row.duration / 60.0
        return out

    def group_counts(self) -> dict[str, dict[str, int]]:
        """
        各群組的合格／淘汰／錯誤數量。

        ⚠ 為什麼不能只看 group_minutes：某個角色可能**全數被淘汰**，
        此時它不會出現在 group_minutes 裡 —— 只看時長會讓「這個角色完全沒素材」
        這件事從報告上消失，而那正是最需要被看到的情況。
        """
        out: dict[str, dict[str, int]] = {}
        for row in self.manifest_rows:
            bucket = out.setdefault(
                row.group, {"ok": 0, "rejected": 0, "skipped": 0, "error": 0}
            )
            if row.status in bucket:
                bucket[row.status] += 1
        return out


# ---------------------------------------------------------------------------
# 收集
# ---------------------------------------------------------------------------
def collect_scan(
    raw_dir: Path,
    extensions: tuple[str, ...],
    *,
    unpacked_dir: Path | None = None,
) -> ScanSummary:
    """
    掃描 00_raw 的格式分布（唯讀）。

    刻意重用 sniff_format 而不是呼叫 CLI 的 cmd_scan —— 報告應該能在任何時候
    產生，不該因為「找不到檔案」而中止。

    ⚠ 關於 `need_unpack` 與 `pending_unpack` 的差別（很容易搞錯）：
    `need_unpack` 是「**格式上**需要解包的容器數」，解包後來源容器仍留在
    00_raw（我們從不搬移或刪除原始素材），所以這個數字不會下降。
    要判斷「還有沒有事要做」，必須看 `pending_unpack` —— 也就是
    01_unpacked 底下是否已經有對應的產物目錄。
    """
    summary = ScanSummary()
    if not raw_dir.is_dir():
        return summary

    exts = set(extensions)
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        summary.total_files += 1
        try:
            summary.total_bytes += path.stat().st_size
        except OSError:
            pass
        info = sniff_format(path)
        summary.by_format[info.label] = summary.by_format.get(info.label, 0) + 1
        if info.needs_unpack:
            summary.need_unpack += 1
            if unpacked_dir is not None and _has_unpacked_output(path, raw_dir, unpacked_dir):
                summary.already_unpacked += 1
        else:
            summary.direct += 1
        if path.suffix.lower() in (".wem", ".wav") and info.code in (
            "ogg_vorbis",
            "ogg_opus",
            "unknown",
        ):
            summary.ext_mismatch += 1

    summary.pending_unpack = summary.need_unpack - summary.already_unpacked
    return summary


def _has_unpacked_output(src: Path, raw_dir: Path, unpacked_dir: Path) -> bool:
    """檢查某個容器是否已經解包過（對應 cmd_unpack 的產物目錄結構）。"""
    try:
        rel_parent = src.relative_to(raw_dir).parent
    except ValueError:
        rel_parent = Path()
    out_dir = unpacked_dir / rel_parent / src.stem
    if not out_dir.is_dir():
        return False
    try:
        return any(out_dir.glob("*.wav"))
    except OSError:
        return False


def build_checks(report: Phase1Report) -> list[CheckItem]:
    """
    逐項檢查 Phase 1 驗收條件（對應 docs/PHASE1-INGEST.md §7）。

    刻意把「有跑過 audition」也算一項 —— 人工試聽是本階段唯一無法自動化的
    驗收步驟，沒有工具痕跡就代表它很可能沒發生。
    """
    checks: list[CheckItem] = []

    checks.append(
        CheckItem(
            label="已產生 manifest（convert 跑過）",
            passed=report.has_manifest,
            detail=(
                f"{len(report.manifest_rows)} 筆紀錄"
                if report.has_manifest
                else "尚未執行 convert"
            ),
        )
    )

    minutes = report.ok_minutes
    checks.append(
        CheckItem(
            label=f"合格素材 ≥ {RECOMMENDED_MIN_MINUTES:.0f} 分鐘（建議 15~60）",
            passed=minutes >= RECOMMENDED_MIN_MINUTES,
            detail=f"目前 {minutes:.1f} 分鐘" + ("" if minutes else "（尚無合格素材）"),
        )
    )

    # 「每個角色 ≥ N 分鐘」只在多角色專案才有意義。
    # ⚠ 這裡必須用 group_counts（manifest 裡出現過的所有群組），
    # 不能用 group_minutes —— 某角色全數被淘汰時它不會出現在後者，
    # 會讓 1 個有效角色 + 1 個全滅角色被誤判成「單一角色專案」。
    if len(report.group_counts()) > 1:
        groups = report.group_minutes()
        thin = {g: m for g, m in groups.items() if m < MIN_PER_GROUP_MINUTES}
        for group, c in report.group_counts().items():
            if c["ok"] == 0:
                thin.setdefault(group, 0.0)
        checks.append(
            CheckItem(
                label=f"每個角色 ≥ {MIN_PER_GROUP_MINUTES:.0f} 分鐘",
                passed=not thin,
                detail=(
                    "全部達標"
                    if not thin
                    else "不足：" + "、".join(f"{g}({m:.1f}分)" for g, m in sorted(thin.items()))
                ),
            )
        )

    checks.append(
        CheckItem(
            label="人工試聽已執行（audition）",
            passed=bool(report.audition_reels),
            detail=(
                f"已產生 {len(report.audition_reels)} 份試聽帶"
                if report.audition_reels
                else "尚未執行 audition —— 這是唯一無法自動化的驗收步驟"
            ),
        )
    )

    checks.append(
        CheckItem(
            label="門檻已檢視（tune）",
            passed=report.tune is not None,
            detail=(
                f"{len(report.tune.suggestions)} 則建議待評估"
                if report.tune and report.tune.suggestions
                else ("無異常" if report.tune else "尚未執行 tune")
            ),
        )
    )

    if report.scan is not None:
        checks.append(
            CheckItem(
                label="原始素材已全部解包（無待處理容器）",
                passed=report.scan.pending_unpack == 0,
                detail=(
                    "無容器待處理"
                    if report.scan.pending_unpack == 0
                    else f"仍有 {report.scan.pending_unpack} 個容器未解包（執行 unpack）"
                ),
            )
        )

    return checks


def collect(
    *,
    paths,
    duration_filter,
    quality_gate,
    scan_extensions: tuple[str, ...],
    do_scan: bool = True,
    cfg=None,
) -> Phase1Report:
    """收集所有 Phase 1 產物與狀態。"""
    report = Phase1Report(generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # manifest → tune 分析
    manifest_path = paths.reports_dir / "manifest_phase1.csv"
    report.manifest_rows = read_manifest(manifest_path)
    if report.manifest_rows:
        report.tune = analyze(
            report.manifest_rows,
            min_seconds=duration_filter.min_seconds,
            max_seconds=duration_filter.max_seconds,
            peak_dbfs_max=quality_gate.peak_dbfs_max,
            min_rms_dbfs=quality_gate.min_rms_dbfs,
        )

    # audition 產物
    if paths.reports_dir.is_dir():
        report.audition_reels = sorted(paths.reports_dir.glob("audition_*.wav"))
        report.audition_lists = sorted(paths.reports_dir.glob("audition_*.md"))

    # 環境後端
    report.backends = [
        (b.name, b.available, b.version) for b in probe_backends(cfg)
    ]

    # 掃描 00_raw（唯讀）
    if do_scan:
        report.scan = collect_scan(
            paths.raw_dir, scan_extensions, unpacked_dir=paths.unpacked_dir
        )

    report.checks = build_checks(report)
    return report


# ---------------------------------------------------------------------------
# 呈現
# ---------------------------------------------------------------------------
def _tick(passed: bool, ascii_only: bool = False) -> str:
    if ascii_only:
        return "[OK]" if passed else "[--]"
    return "✅" if passed else "⬜"


def render(
    report: Phase1Report,
    *,
    paths=None,
    duration_filter=None,
    quality_gate=None,
    target=None,
    repo_root: Path | None = None,
    ascii_only: bool = False,
) -> str:
    """把報告渲染成 Markdown。"""
    root = repo_root or config.repo_root()

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return p.resolve().as_posix()

    L: list[str] = []
    L.append("# Phase 1 綜合報告 — 數據解包與格式標準化")
    L.append("")
    L.append(f"產生時間：{report.generated_at}")
    if duration_filter is not None and target is not None:
        L.append(
            f"目標格式：{target.bit_depth} bit / {target.sample_rate} Hz / "
            f"{'單聲道' if target.channels == 1 else '雙聲道'} PCM WAV"
        )
        L.append(
            f"篩選門檻：{duration_filter.min_seconds}s ~ {duration_filter.max_seconds}s"
            + (
                f"　｜　Peak ≤ {quality_gate.peak_dbfs_max} dBFS"
                f"　｜　RMS ≥ {quality_gate.min_rms_dbfs} dBFS"
                if quality_gate is not None and quality_gate.enabled
                else "　｜　（品質關卡已停用）"
            )
        )
    L.append("")

    # ---- 1. 總覽 ----
    L.append("## 1. 總覽")
    L.append("")
    if report.has_manifest:
        total = len(report.manifest_rows)
        counts: dict[str, int] = {}
        for row in report.manifest_rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        L.append("| 項目 | 數量 |")
        L.append("|------|-----:|")
        L.append(f"| 處理過的檔案 | {total} |")
        L.append(f"| ✅ 合格 | {counts.get('ok', 0)} |")
        L.append(f"| ✂ 淘汰 | {counts.get('rejected', 0)} |")
        L.append(f"| · 略過（容器/已存在） | {counts.get('skipped', 0)} |")
        L.append(f"| ✘ 錯誤 | {counts.get('error', 0)} |")
        L.append(f"| **合格素材總長** | **{report.ok_minutes:.1f} 分鐘** |")
    else:
        L.append("> ⚠ 尚未產生 manifest。先執行 `python scripts/phase1_prep.py convert`。")
    L.append("")

    # ---- 2. 原始素材格式分布 ----
    if report.scan is not None:
        L.append("## 2. 原始素材（data/00_raw）")
        L.append("")
        if report.scan.total_files == 0:
            L.append("> ⚠ 找不到任何目標副檔名的檔案。請把遊戲語音放進 `data/00_raw/`")
            L.append("> （多角色專案用 `data/00_raw/<角色名>/`）。")
        else:
            L.append(
                f"共 **{report.scan.total_files}** 個檔案，"
                f"{report.scan.total_bytes / 1024 / 1024:.1f} MiB。"
                f"可直接解碼 {report.scan.direct} 個，需先解包 {report.scan.need_unpack} 個。"
            )
            L.append("")
            L.append("| 偵測到的格式 | 數量 |")
            L.append("|--------------|-----:|")
            for label, count in sorted(
                report.scan.by_format.items(), key=lambda kv: -kv[1]
            ):
                L.append(f"| {label} | {count} |")
            if report.scan.need_unpack:
                pending = report.scan.pending_unpack
                if pending:
                    L.append("")
                    L.append(
                        f"> ⚠ 還有 **{pending}** 個容器未解包"
                        f"（需要 vgmstream-cli，見 [格式手冊](FORMAT-PLAYBOOK.md) §8）。"
                    )
                else:
                    L.append("")
                    L.append(
                        f"> ✅ {report.scan.already_unpacked} 個容器已全部解包完成"
                        f"（來源容器仍保留在 00_raw，這是刻意的 —— 原始素材永不搬移）。"
                    )
        L.append("")

    # ---- 3. 逐角色分布 ----
    # 只要 manifest 有出現過任何群組名就該顯示 —— 包含「某個角色全數被淘汰」
    # 的情況（那時它不會出現在 group_minutes，但正是最該被看到的警訊）。
    group_counts = report.group_counts()
    if group_counts:
        minutes_by_group = report.group_minutes()
        multi = len(group_counts) > 1
        L.append("## 3. 逐角色素材量" + ("（多角色專案）" if multi else ""))
        L.append("")
        L.append("| 角色 | 合格 | 淘汰 | 合格時長 | 狀態 |")
        L.append("|------|-----:|-----:|--------:|------|")
        for group in sorted(group_counts, key=lambda g: -minutes_by_group.get(g, 0.0)):
            counts = group_counts[group]
            minutes = minutes_by_group.get(group, 0.0)
            if counts["ok"] == 0:
                flag = "❌ 無合格素材"
            elif multi and minutes < MIN_PER_GROUP_MINUTES:
                flag = "⚠ 不足 3 分鐘"
            else:
                flag = "✅"
            L.append(
                f"| {group} | {counts['ok']} | {counts['rejected']} | "
                f"{minutes:.1f} 分鐘 | {flag} |"
            )
        L.append("")

    # ---- 4. 淘汰分析 + 建議 ----
    if report.tune is not None and report.tune.categories:
        L.append("## 4. 篩選結果分析")
        L.append("")
        L.append("| 淘汰原因 | 數量 | 緊貼門檻 |")
        L.append("|----------|-----:|--------:|")
        for key, cat in sorted(
            report.tune.categories.items(), key=lambda kv: -kv[1].count
        ):
            L.append(f"| {cat.label} | {cat.count} | {len(cat.boundary)} |")
        L.append("")
        if report.tune.suggestions:
            L.append("### 建議")
            L.append("")
            for s in report.tune.suggestions:
                icon = "⚠" if s.severity == "warn" else "ℹ"
                L.append(f"- {icon} **{s.title}**")
                for line in s.detail.splitlines():
                    L.append(f"  {line.strip()}")
                if s.config_hint:
                    L.append(f"  - 設定片段（`configs/pipeline.local.json`）：`{s.config_hint}`")
            L.append("")
        else:
            L.append("> 沒有需要調整的地方。")
            L.append("")

    # ---- 5. 驗收清單 ----
    L.append("## 5. 進入 Phase 2 的驗收條件")
    L.append("")
    L.append("| 狀態 | 條件 | 現況 |")
    L.append("|:----:|------|------|")
    for check in report.checks:
        L.append(f"| {_tick(check.passed, ascii_only)} | {check.label} | {check.detail} |")
    L.append("")
    if report.all_checks_passed:
        L.append("### ✅ 全部達標 —— 可以進入 Phase 2")
        L.append("")
        L.append(
            "> 提醒：`audition` 只是「產生試聽帶」。**你必須真的從頭聽完**，"
            "確認沒有底噪、沒有截頭截尾、沒有換人、情感種類有多樣。"
            "報告只能記錄你跑過工具，不能代替你的耳朵。"
        )
    else:
        pending = [c for c in report.checks if not c.passed]
        L.append(f"### ⬜ 還有 {len(pending)} 項未達標")
        L.append("")
        for check in pending:
            L.append(f"- **{check.label}** — {check.detail}")
    L.append("")

    # ---- 6. 環境 ----
    if report.backends:
        L.append("## 6. 環境狀態")
        L.append("")
        L.append("| 後端 | 可用 | 版本／路徑 |")
        L.append("|------|:----:|-----------|")
        for name, available, version in report.backends:
            L.append(f"| {name} | {'✅' if available else '❌'} | {version} |")
        L.append("")

    # ---- 7. 產物追溯 ----
    L.append("## 7. 產物位置")
    L.append("")
    if paths is not None:
        items = [
            ("合格素材（Phase 2 的輸入）", paths.standardized_dir),
            ("容器解包產物", paths.unpacked_dir),
            ("被淘汰的檔案（供人工複核）", paths.rejected_dir),
            ("原始素材", paths.raw_dir),
            ("報表", paths.reports_dir),
        ]
        L.append("| 用途 | 路徑 |")
        L.append("|------|------|")
        for label, path in items:
            exists = "✔" if path.is_dir() else "（尚未建立）"
            L.append(f"| {label} | `{rel(path)}` {exists} |")
    if report.audition_reels:
        L.append("")
        L.append("試聽帶：")
        for reel in report.audition_reels:
            L.append(f"- `{rel(reel)}`")
    L.append("")

    # ---- 8. 下一步 ----
    L.append("## 8. 下一步指令")
    L.append("")
    L.extend(_next_steps(report))
    L.append("")

    return "\n".join(L)


def _next_steps(report: Phase1Report) -> list[str]:
    """依目前狀態給出「照著打就會動」的指令序列。"""
    steps: list[str] = []
    if not report.has_manifest:
        if report.scan is not None and report.scan.total_files == 0:
            steps.append(
                "1. 把遊戲語音放進 `data/00_raw/`（多角色用 `data/00_raw/<角色名>/`）"
            )
            steps.append("2. `python scripts/phase1_prep.py scan` — 先確認格式")
        else:
            steps.append("1. `python scripts/phase1_prep.py scan` — 確認格式分布")
            if report.scan is not None and report.scan.pending_unpack:
                steps.append("2. `python scripts/phase1_prep.py unpack` — 解開容器")
            steps.append("3. `python scripts/phase1_prep.py convert` — 標準化 + 篩選")
        return steps

    if report.scan is not None and report.scan.pending_unpack:
        steps.append(
            f"1. `python scripts/phase1_prep.py unpack` — "
            f"還有 {report.scan.pending_unpack} 個容器未處理"
        )

    if report.tune is not None:
        warns = [s for s in report.tune.suggestions if s.severity == "warn"]
        if warns:
            steps.append(
                "2. 檢視上面的建議，必要時改 `configs/pipeline.local.json` 後重跑 convert"
            )

    if not report.audition_reels:
        steps.append(
            "3. `python scripts/phase1_prep.py audition --per-group 5` — 產生試聽帶"
        )
        steps.append("4. **從頭聽完試聽帶**，對照清單逐項確認品質")
    else:
        steps.append(f"3. 再聽一次試聽帶（已有 {len(report.audition_reels)} 份）")

    if not report.all_checks_passed:
        steps.append("5. 修正上面未達標的項目後，重新產生這份報告")
    else:
        steps.append("5. ✅ 驗收完成 —— 可以請我進入 Phase 2（UVR5 人聲淨化 + 響度標準化）")

    return steps


def render_json(report: Phase1Report) -> str:
    """機器可讀版本（供 Phase 2 直接讀）。"""
    from tune import report_as_dict

    payload = {
        "phase": 1,
        "generated_at": report.generated_at,
        "has_manifest": report.has_manifest,
        "ok_count": report.ok_count,
        "ok_minutes": round(report.ok_minutes, 3),
        "group_minutes": {g: round(m, 3) for g, m in report.group_minutes().items()},
        "group_counts": report.group_counts(),
        "checks": [
            {"label": c.label, "passed": c.passed, "detail": c.detail} for c in report.checks
        ],
        "all_checks_passed": report.all_checks_passed,
        "scan": (
            {
                "total_files": report.scan.total_files,
                "total_bytes": report.scan.total_bytes,
                "need_unpack": report.scan.need_unpack,
                "already_unpacked": report.scan.already_unpacked,
                "pending_unpack": report.scan.pending_unpack,
                "direct": report.scan.direct,
                "by_format": report.scan.by_format,
            }
            if report.scan is not None
            else None
        ),
        "audition_reels": [p.name for p in report.audition_reels],
        "tune": report_as_dict(report.tune) if report.tune is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "MIN_PER_GROUP_MINUTES",
    "MIN_TOTAL_MINUTES",
    "RECOMMENDED_MAX_MINUTES",
    "RECOMMENDED_MIN_MINUTES",
    "CheckItem",
    "Phase1Report",
    "ScanSummary",
    "build_checks",
    "collect",
    "collect_scan",
    "render",
    "render_json",
]
