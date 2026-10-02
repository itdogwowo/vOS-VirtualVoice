"""
vOS-VirtualVoice — 篩選門檻調校建議 (Phase 1)

為什麼需要這個模組
------------------
Phase 1 的門檻（1.5~15 秒、peak ≤ -0.5 dBFS、RMS ≥ -50 dBFS）是**起點，不是聖旨**。
但「淘汰率 40%」這個數字本身沒有意義 —— 你必須知道**淘汰的是什麼**才能判斷：

  * 如果淘汰的都是 0.3~0.8 秒的喘氣聲  → 門檻正確，素材就是這樣
  * 如果淘汰的是 1.2~1.5 秒的短台詞    → 門檻太嚴，你在丟掉可用的情感素材
  * 如果淘汰的是 RMS -53 dBFS 的耳語    → 門檻太嚴，而且丟掉的正是最珍貴的素材

這個模組把 manifest 變成「該不該動門檻、往哪個方向動」的具體建議。

設計原則
--------
* **只建議、不改檔**：門檻是藝術判斷，工具負責給數據，你負責決定。
* **樣本不足就不建議**：只有 2 個檔案被淘汰時，任何統計都是噪音。
* **考慮邊界群聚**：緊貼門檻被淘汰的檔案，才是真正值得重新考慮的對象；
  離門檻很遠的被淘汰者是門檻正確運作的證據。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

#: 被淘汰的檔案裡，落在門檻這個比例區間內才值得重新考慮。
#: 例如 min=1.5s、margin=0.4 時，1.5~2.1 秒的被淘汰者會被點出來。
DEFAULT_MARGIN = 0.4
#: 要有多少筆「邊界被淘汰者」才值得提出建議（樣本太少 = 噪音）
MIN_BOUNDARY_SAMPLES = 3
#: 要有多少筆同類淘汰才值得提出整體性建議
MIN_CATEGORY_SAMPLES = 5


# ---------------------------------------------------------------------------
# manifest 讀取
# ---------------------------------------------------------------------------
@dataclass
class ManifestRow:
    """manifest 的一列（只保留分析需要的欄位）。"""

    status: str
    reason: str
    group: str
    src_path: str
    duration: float | None
    peak_dbfs: float | None
    rms_dbfs: float | None


def _to_float(text: str | None) -> float | None:
    """把 manifest 的字串轉成 float；'-inf' / 空字串 / 壞值回 None。"""
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned or cleaned.lower() in ("-inf", "inf", "nan", "none"):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_manifest(path: str | Path) -> list[ManifestRow]:
    """
    讀取 manifest_phase1.csv（utf-8-sig，Windows Excel 相容）。

    欄位缺漏時不炸掉 —— manifest 的 schema 會隨階段演進，
    分析工具應該容忍舊格式，只要能算的就算。
    """
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[ManifestRow] = []
    with p.open(encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                ManifestRow(
                    status=(raw.get("status") or "").strip(),
                    reason=(raw.get("reason") or "").strip(),
                    group=(raw.get("group") or "default").strip() or "default",
                    src_path=(raw.get("src_path") or "").strip(),
                    duration=_to_float(raw.get("duration_sec")),
                    peak_dbfs=_to_float(raw.get("peak_dbfs")),
                    rms_dbfs=_to_float(raw.get("rms_dbfs")),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def percentile(values: list[float], pct: float) -> float | None:
    """線性插值百分位數（不依賴 numpy，方便單獨測試）。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


@dataclass
class CategoryStats:
    """一類淘汰原因的統計。"""

    key: str            # 內部代號：too_short / too_long / clipped / too_quiet / other
    label: str          # 顯示名稱
    rows: list[ManifestRow] = field(default_factory=list)
    #: 緊貼門檻、「差一點就合格」的檔案
    boundary: list[ManifestRow] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)


def categorize(rows: list[ManifestRow]) -> dict[str, CategoryStats]:
    """
    依 reason 的關鍵字分類被淘汰的檔案。

    為什麼用關鍵字而不是狀態碼：manifest 是給人看的，
    reason 是可讀中文；額外加一個機器碼欄位會讓 schema 更複雜但收益有限。
    """
    specs = {
        "too_short": "過短",
        "too_long": "過長",
        "clipped": "削波",
        "too_quiet": "訊號過弱",
        "silent": "靜音",
    }
    stats: dict[str, CategoryStats] = {}
    for key, label in specs.items():
        stats[key] = CategoryStats(key=key, label=label)
    stats["other"] = CategoryStats(key="other", label="其他")

    for row in rows:
        if row.status != "rejected":
            continue
        matched = False
        for key, keyword in specs.items():
            if keyword in row.reason:
                stats[key].rows.append(row)
                matched = True
                break
        if not matched:
            stats["other"].rows.append(row)

    # 只回傳「真的有東西」的分類，避免下游一直處理一堆空殼
    return {key: cat for key, cat in stats.items() if cat.count > 0}


# ---------------------------------------------------------------------------
# 建議
# ---------------------------------------------------------------------------
@dataclass
class Suggestion:
    """一則具體、可行動的建議。"""

    key: str
    title: str
    detail: str
    config_hint: str = ""   # 可以直接貼進 configs/pipeline.local.json 的片段
    severity: str = "info"  # info / warn


@dataclass
class TuneReport:
    ok_count: int = 0
    rejected_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    ok_seconds: float = 0.0
    categories: dict[str, CategoryStats] = field(default_factory=dict)
    duration_stats: dict[str, float | None] = field(default_factory=dict)
    suggestions: list[Suggestion] = field(default_factory=list)
    per_group_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.ok_count + self.rejected_count + self.skipped_count + self.error_count

    @property
    def ok_ratio(self) -> float:
        return self.ok_count / self.total if self.total else 0.0


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def analyze(
    rows: list[ManifestRow],
    *,
    min_seconds: float,
    max_seconds: float,
    peak_dbfs_max: float,
    min_rms_dbfs: float,
    margin: float = DEFAULT_MARGIN,
) -> TuneReport:
    """
    分析 manifest 並產生門檻調校建議。

    這是純函式（不讀檔、不寫檔），方便測試任意門檻組合。
    """
    report = TuneReport()
    for row in rows:
        if row.status == "ok":
            report.ok_count += 1
            if row.duration:
                report.ok_seconds += row.duration
                report.per_group_seconds[row.group] = (
                    report.per_group_seconds.get(row.group, 0.0) + row.duration
                )
        elif row.status == "rejected":
            report.rejected_count += 1
        elif row.status == "skipped":
            report.skipped_count += 1
        elif row.status == "error":
            report.error_count += 1

    report.categories = categorize(rows)

    ok_durations = [
        r.duration for r in rows if r.status == "ok" and r.duration is not None
    ]
    report.duration_stats = {
        "min": min(ok_durations) if ok_durations else None,
        "p05": percentile(ok_durations, 5),
        "median": percentile(ok_durations, 50),
        "p95": percentile(ok_durations, 95),
        "max": max(ok_durations) if ok_durations else None,
    }

    # ---- 標出「緊貼門檻被淘汰」的檔案 ----
    #
    # ⚠ 方向很容易寫反：被「過短」淘汰的檔案，其時長必定 **小於** min_seconds。
    # 所以「差一點就合格」的區間是 [min*(1-margin), min)，不是 [min, min*(1+margin))。
    # 寫反的症狀是永遠找不到邊界樣本 —— 分析工具會靜默地什麼都不建議。
    short = report.categories.get("too_short")
    if short:
        floor = min_seconds * (1 - margin)
        short.boundary = [
            r for r in short.rows if r.duration is not None and floor <= r.duration < min_seconds
        ]
    long = report.categories.get("too_long")
    if long:
        ceiling = max_seconds * (1 + margin)
        long.boundary = [
            r for r in long.rows if r.duration is not None and max_seconds < r.duration <= ceiling
        ]
    quiet = report.categories.get("too_quiet")
    if quiet:
        # RMS 是負值且門檻是「下限」：被淘汰者必定 **小於** min_rms_dbfs。
        # 「差一點」的區間是 (min - span, min)。
        span = abs(min_rms_dbfs) * margin
        quiet.boundary = [
            r
            for r in quiet.rows
            if r.rms_dbfs is not None and min_rms_dbfs - span < r.rms_dbfs < min_rms_dbfs
        ]

    report.suggestions = _build_suggestions(
        report,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        peak_dbfs_max=peak_dbfs_max,
        min_rms_dbfs=min_rms_dbfs,
    )

    # manifest 全空時，「資料不足」本身就是最重要的結論 —— 必須明確講出來，
    # 否則使用者只會看到一份什麼都沒說的報告。
    if report.total == 0:
        report.suggestions.insert(
            0,
            Suggestion(
                key="no_data",
                title="manifest 內沒有任何資料",
                detail=(
                    "可能是 convert 還沒跑，或 manifest 被清空／格式不符。\n"
                    "    請先執行：python scripts/phase1_prep.py convert"
                ),
                severity="warn",
            ),
        )
    return report


def _build_suggestions(
    report: TuneReport,
    *,
    min_seconds: float,
    max_seconds: float,
    peak_dbfs_max: float,
    min_rms_dbfs: float,
) -> list[Suggestion]:
    out: list[Suggestion] = []

    # ---- 過短：邊界群聚 = 可能誤殺 ----
    short = report.categories.get("too_short")
    if short and len(short.boundary) >= MIN_BOUNDARY_SAMPLES:
        durations = [r.duration for r in short.boundary if r.duration is not None]
        lo, hi = min(durations), max(durations)
        suggested = max(0.8, round(lo - 0.1, 2))
        out.append(
            Suggestion(
                key="lower_min_seconds",
                title=f"考慮把最短時長從 {min_seconds}s 降到約 {suggested}s",
                detail=(
                    f"有 {len(short.boundary)} 個檔案落在 {lo:.2f}~{hi:.2f}s 之間被淘汰"
                    f"（門檻是 {min_seconds}s）。\n"
                    f"    緊貼門檻的失敗者通常不是垃圾，而是差一點的短台詞 —— "
                    f"它們對情感模型比標準長度的旁白更有價值。\n"
                    f"    動手前先聽：python scripts/phase1_prep.py audition --groups <該群組>，"
                    f"再手動開 99_rejected/ 內這幾個檔案對照。"
                ),
                config_hint=f'{{ "duration_filter": {{ "min_seconds": {suggested} }} }}',
                severity="warn",
            )
        )
    elif short and short.count >= MIN_CATEGORY_SAMPLES:
        out.append(
            Suggestion(
                key="min_seconds_ok",
                title=f"最短時長 {min_seconds}s 看起來是對的",
                detail=(
                    f"{short.count} 個檔案因過短被淘汰，且沒有群聚在門檻附近"
                    f"（多半是真正的喘氣／單字哼聲）。維持現值。"
                ),
            )
        )

    # ---- 過長 ----
    long = report.categories.get("too_long")
    if long and len(long.boundary) >= MIN_BOUNDARY_SAMPLES:
        durations = [r.duration for r in long.boundary if r.duration is not None]
        lo, hi = min(durations), max(durations)
        suggested = min(30.0, round(hi + 0.5, 1))
        out.append(
            Suggestion(
                key="raise_max_seconds",
                title=f"考慮把最長時長從 {max_seconds}s 提高到約 {suggested}s",
                detail=(
                    f"有 {len(long.boundary)} 個檔案落在 {lo:.1f}~{hi:.1f}s 之間被淘汰"
                    f"（門檻是 {max_seconds}s）。\n"
                    f"    ⚠ 但要注意代價：Phase 4 的 ASR 對齊在 20 秒以上會明顯變差，"
                    f"Phase 5 的參考音訊建議 3~10 秒。\n"
                    f"    除非這些長檔案的情感價值特別高，否則不建議為了多收幾條而拉長門檻。"
                ),
                config_hint=f'{{ "duration_filter": {{ "max_seconds": {suggested} }} }}',
                severity="warn",
            )
        )
    elif long and long.count >= MIN_CATEGORY_SAMPLES:
        out.append(
            Suggestion(
                key="max_seconds_ok",
                title=f"最長時長 {max_seconds}s 看起來是對的",
                detail=f"{long.count} 個檔案因過長被淘汰，沒有明顯的邊界群聚。",
            )
        )

    # ---- 訊號過弱：最珍貴的誤殺類型 ----
    quiet = report.categories.get("too_quiet")
    if quiet and len(quiet.boundary) >= MIN_BOUNDARY_SAMPLES:
        lowest = min(r.rms_dbfs for r in quiet.boundary if r.rms_dbfs is not None)
        suggested = round(lowest - 3, 1)
        out.append(
            Suggestion(
                key="lower_min_rms",
                title=f"考慮把 RMS 下限從 {min_rms_dbfs} 降到約 {suggested} dBFS",
                detail=(
                    f"有 {len(quiet.boundary)} 個檔案落在 {min_rms_dbfs} 附近被淘汰，"
                    f"最低的是 {lowest:.1f} dBFS。\n"
                    f"    ⚠ 遊戲刻意錄小聲的耳語／內心戲台詞常落在這裡 —— "
                    f"它們對情感模型的價值特別高，被丟掉很可惜。\n"
                    f"    這些檔案不需要在 Phase 1 解決音量問題，Phase 2 的響度標準化會處理。"
                ),
                config_hint=f'{{ "quality_gate": {{ "min_rms_dbfs": {suggested} }} }}',
                severity="warn",
            )
        )

    # ---- 削波：通常是素材本身的問題，不是門檻 ----
    clipped = report.categories.get("clipped")
    if clipped and clipped.count >= MIN_CATEGORY_SAMPLES:
        out.append(
            Suggestion(
                key="clipping_source",
                title=f"{clipped.count} 個檔案削波（peak > {peak_dbfs_max} dBFS）",
                detail=(
                    "削波是波形已經被切平的物理損壞，降門檻救不回來。\n"
                    "    兩種可能：① 素材本身在遊戲內混音時就爆了（無解，放棄）\n"
                    "    ② 你用了解包工具做了音量增益（檢查工具參數，別做增益）\n"
                    "    建議先試聽 99_rejected/ 內其中 2~3 個確認是哪一種。"
                ),
                severity="info",
            )
        )

    # ---- 素材量體檢 ----
    minutes = report.ok_seconds / 60.0
    if minutes < 5:
        out.append(
            Suggestion(
                key="too_little_material",
                title=f"合格素材只有 {minutes:.1f} 分鐘，遠低於微調需求",
                detail=(
                    "GPT-SoVITS 微調建議 ≥ 10 分鐘，最佳 15~60 分鐘。\n"
                    "    三個方向：① 補更多遊戲語音 ② 放寬門檻（見上方建議）"
                    " ③ 重新解包（可能有容器被 subsong 上限擋掉了）"
                ),
                severity="warn",
            )
        )
    elif minutes > 60:
        out.append(
            Suggestion(
                key="too_much_material",
                title=f"合格素材 {minutes:.1f} 分鐘偏多",
                detail=(
                    "依你的 Quality Over Quantity 原則：15 分鐘神級乾淨音訊勝過 2 小時雜音數據。\n"
                    "    建議先挑 15~60 分鐘『最乾淨、情感最豐富』的子集，"
                    "並用 audition 確認這批子集的品質。"
                ),
                severity="info",
            )
        )

    # ---- 群組素材不足 ----
    thin = {g: s for g, s in report.per_group_seconds.items() if s < 180}
    if len(report.per_group_seconds) > 1 and thin:
        names = "、".join(f"{g}({s / 60:.1f}分)" for g, s in sorted(thin.items())[:6])
        out.append(
            Suggestion(
                key="thin_groups",
                title=f"{len(thin)} 個角色的素材不足 3 分鐘",
                detail=(
                    f"{names}\n"
                    "    多角色專案若某角色素材太少，Phase 5 生成該角色的音色會不穩定。\n"
                    "    選項：① 補素材 ② 考慮把相近角色合併成一個群組"
                    "（改資料夾結構後重跑 convert）"
                ),
                severity="warn",
            )
        )

    # ---- 解碼錯誤 ----
    if report.error_count >= MIN_CATEGORY_SAMPLES:
        out.append(
            Suggestion(
                key="decode_errors",
                title=f"{report.error_count} 個檔案解碼失敗",
                detail=(
                    "多半是缺少 ffmpeg 或該格式太冷門。\n"
                    "    先跑 doctor 確認後端，再對照 docs/FORMAT-PLAYBOOK.md 的排查流程。"
                ),
                severity="warn",
            )
        )

    return out


def render_report(report: TuneReport, *, min_seconds: float, max_seconds: float,
                  peak_dbfs_max: float, min_rms_dbfs: float) -> str:
    """把報告變成可讀文字（給 CLI 印出）。"""
    lines: list[str] = []
    total = report.total

    lines.append("=" * 72)
    lines.append(" Phase 1 門檻調校分析")
    lines.append("=" * 72)
    lines.append(
        f" manifest：{total} 個檔案  "
        f"合格 {report.ok_count}（{report.ok_ratio * 100:.1f}%）"
        f" 淘汰 {report.rejected_count}"
        f" 略過 {report.skipped_count}"
        f" 錯誤 {report.error_count}"
    )
    lines.append(f" 合格素材總長：{report.ok_seconds / 60:.1f} 分鐘")

    lines.append("\n 現行門檻")
    lines.append(f"   時長      {min_seconds}s ~ {max_seconds}s")
    lines.append(f"   Peak      ≤ {peak_dbfs_max} dBFS")
    lines.append(f"   RMS       ≥ {min_rms_dbfs} dBFS")

    ds = report.duration_stats
    if ds.get("median") is not None:
        lines.append("\n 合格素材時長分布（秒）")
        lines.append(
            f"   最短 {_fmt(ds.get('min'))}   P5 {_fmt(ds.get('p05'))}"
            f"   中位數 {_fmt(ds.get('median'))}"
            f"   P95 {_fmt(ds.get('p95'))}   最長 {_fmt(ds.get('max'))}"
        )
    else:
        lines.append("\n 合格素材時長分布：尚無合格樣本（全部被淘汰或還沒跑 convert）")

    if report.categories:
        lines.append("\n 淘汰原因分布")
        for key, cat in sorted(
            report.categories.items(), key=lambda kv: -kv[1].count
        ):
            if not cat.count:
                continue
            boundary = f"（其中 {len(cat.boundary)} 個緊貼門檻）" if cat.boundary else ""
            lines.append(f"   {cat.label:<10} {cat.count:>6} 個{boundary}")

    if report.per_group_seconds and len(report.per_group_seconds) > 1:
        lines.append("\n 逐角色合格時長")
        for group, seconds in sorted(
            report.per_group_seconds.items(), key=lambda kv: -kv[1]
        ):
            flag = "  ← 不足 3 分鐘" if seconds < 180 else ""
            lines.append(f"   {group:<22} {seconds / 60:>6.1f} 分鐘{flag}")

    lines.append("\n" + "=" * 72)
    lines.append(" 建議")
    lines.append("=" * 72)
    if not report.suggestions:
        lines.append(" 資料量不足以提出建議。先跑 convert 產生 manifest 再說。")
    for i, s in enumerate(report.suggestions, 1):
        mark = {"warn": "⚠", "info": "ℹ"}.get(s.severity, "ℹ")
        lines.append(f"\n {mark} [{i}] {s.title}")
        lines.append(f"    {s.detail}")
        if s.config_hint:
            lines.append(f"    設定片段 → configs/pipeline.local.json")
            lines.append(f"      {s.config_hint}")

    lines.append(
        "\n 提醒：這些是**建議**。門檻是藝術判斷 —— 動手改之前，"
        "先用 audition 聽聽 99_rejected/ 裡的邊界檔案。"
    )
    return "\n".join(lines)


def report_as_dict(report: TuneReport) -> dict:
    """機器可讀版本（寫進 _reports/ 供後續階段參考）。"""
    return {
        "phase": 1,
        "counts": {
            "ok": report.ok_count,
            "rejected": report.rejected_count,
            "skipped": report.skipped_count,
            "error": report.error_count,
        },
        "ok_ratio": round(report.ok_ratio, 4),
        "ok_seconds": round(report.ok_seconds, 3),
        "duration_stats": {k: (round(v, 3) if v is not None else None) for k, v in report.duration_stats.items()},
        "reject_categories": {
            key: {"label": cat.label, "count": cat.count, "boundary_count": len(cat.boundary)}
            for key, cat in report.categories.items()
            if cat.count
        },
        "per_group_seconds": {g: round(s, 3) for g, s in report.per_group_seconds.items()},
        "suggestions": [
            {"key": s.key, "severity": s.severity, "title": s.title, "config_hint": s.config_hint}
            for s in report.suggestions
        ],
    }


__all__ = [
    "DEFAULT_MARGIN",
    "MIN_BOUNDARY_SAMPLES",
    "MIN_CATEGORY_SAMPLES",
    "CategoryStats",
    "ManifestRow",
    "Suggestion",
    "TuneReport",
    "analyze",
    "categorize",
    "percentile",
    "read_manifest",
    "render_report",
    "report_as_dict",
]
