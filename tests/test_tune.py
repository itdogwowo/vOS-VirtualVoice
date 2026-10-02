"""
vOS-VirtualVoice — 門檻調校分析測試 (Phase 1)

核心觀念：**淘汰率本身沒有意義，淘汰的內容才有意義。**
這些測試守著「工具能不能分辨以下兩種情況」：

  A. 淘汰 0.3~0.8 秒的喘氣聲      → 門檻正確，不該建議改動
  B. 淘汰 1.2~1.5 秒的短台詞      → 門檻太嚴，必須建議放寬

分不出 A 和 B 的分析工具，比沒有分析工具更糟（會誤導你丟掉好素材）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config as cfgmod  # noqa: E402
from tune import (  # noqa: E402
    ManifestRow,
    analyze,
    categorize,
    percentile,
    read_manifest,
    render_report,
    report_as_dict,
)

DEFAULTS = dict(min_seconds=1.5, max_seconds=15.0, peak_dbfs_max=-0.5, min_rms_dbfs=-50.0)


def row(status: str, reason: str = "", duration: float | None = None,
        rms: float | None = None, peak: float | None = None,
        group: str = "default", src: str = "a.wav") -> ManifestRow:
    return ManifestRow(
        status=status, reason=reason, group=group, src_path=src,
        duration=duration, peak_dbfs=peak, rms_dbfs=rms,
    )


# ---------------------------------------------------------------------------
# 1. 基礎工具
# ---------------------------------------------------------------------------
class TestPercentile:
    @pytest.mark.parametrize(
        "values,pct,expected",
        [
            ([], 50, None),
            ([5.0], 50, 5.0),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 50, 3.0),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 0, 1.0),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 100, 5.0),
            ([1.0, 2.0], 50, 1.5),
        ],
    )
    def test_percentile(self, values, pct, expected) -> None:
        got = percentile(values, pct)
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected)

    def test_unsorted_input_is_handled(self) -> None:
        assert percentile([5.0, 1.0, 3.0], 50) == 3.0


class TestReadManifest:
    def test_reads_utf8_sig_and_parses_numbers(self, tmp_dir: Path) -> None:
        p = tmp_dir / "manifest_phase1.csv"
        p.write_text(
            "status,reason,group,src_path,duration_sec,peak_dbfs,rms_dbfs\n"
            "ok,通過,凛,data/00_raw/a.wav,3.500,-6.02,-11.36\n"
            "rejected,過短 0.70s < 1.5s,凛,data/00_raw/b.wav,0.700,-inf,-inf\n"
            "rejected,全靜音,,data/00_raw/c.wav,3.000,-inf,-inf\n",
            encoding="utf-8-sig",
        )
        rows = read_manifest(p)
        assert len(rows) == 3
        assert rows[0].status == "ok"
        assert rows[0].duration == pytest.approx(3.5)
        assert rows[0].rms_dbfs == pytest.approx(-11.36)
        assert rows[1].duration == pytest.approx(0.7)
        assert rows[1].rms_dbfs is None, "-inf 必須轉成 None，否則後續統計會被污染"
        assert rows[2].group == "default", "空 group 應回退為 default"

    def test_missing_file_returns_empty(self, tmp_dir: Path) -> None:
        assert read_manifest(tmp_dir / "nope.csv") == []

    def test_missing_columns_do_not_crash(self, tmp_dir: Path) -> None:
        """manifest schema 會隨階段演進，分析工具必須容忍舊格式。"""
        p = tmp_dir / "old.csv"
        p.write_text("status,reason\nok,通過\n", encoding="utf-8")
        rows = read_manifest(p)
        assert len(rows) == 1
        assert rows[0].duration is None

    def test_malformed_numbers_become_none(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bad.csv"
        p.write_text(
            "status,duration_sec,rms_dbfs\nok,N/A,abc\n", encoding="utf-8"
        )
        rows = read_manifest(p)
        assert rows[0].duration is None and rows[0].rms_dbfs is None


class TestCategorize:
    def test_categorizes_by_reason_keyword(self) -> None:
        rows = [
            row("rejected", "過短 0.70s < 1.5s（喘氣/單字哼聲）"),
            row("rejected", "過長 16.50s > 15.0s"),
            row("rejected", "削波風險 peak 0.00 > -0.5 dBFS"),
            row("rejected", "訊號過弱 RMS -55.0 < -50.0 dBFS"),
            row("rejected", "全靜音（無有效訊號）"),
            row("rejected", "某個沒見過的原因"),
            row("ok", "通過"),
        ]
        cats = categorize(rows)
        assert cats["too_short"].count == 1
        assert cats["too_long"].count == 1
        assert cats["clipped"].count == 1
        assert cats["too_quiet"].count == 1
        assert cats["silent"].count == 1
        assert cats["other"].count == 1
        assert sum(c.count for c in cats.values()) == 6, "ok 不該被算進淘汰分類"

    def test_empty_categories_are_dropped(self) -> None:
        cats = categorize([row("rejected", "過短 0.1s")])
        assert set(cats) == {"too_short"}


# ---------------------------------------------------------------------------
# 2. 關鍵判斷：邊界群聚 vs 遠離門檻
# ---------------------------------------------------------------------------
class TestBoundaryDetection:
    def test_short_utterances_just_below_threshold_trigger_suggestion(self) -> None:
        """
        情境 B：1.2~1.5 秒的短台詞被淘汰 → 必須建議放寬。
        這是最容易誤殺的一類 —— 它們是真正的情感素材。
        """
        rows = [row("ok", "通過", duration=4.0) for _ in range(20)]
        rows += [
            row("rejected", "過短 1.30s < 1.5s", duration=1.30, rms=-14.0),
            row("rejected", "過短 1.35s < 1.5s", duration=1.35, rms=-14.0),
            row("rejected", "過短 1.42s < 1.5s", duration=1.42, rms=-14.0),
            row("rejected", "過短 1.20s < 1.5s", duration=1.20, rms=-14.0),
        ]
        report = analyze(rows, **DEFAULTS)
        keys = {s.key for s in report.suggestions}
        assert "lower_min_seconds" in keys, (
            f"邊界群聚必須觸發放寬建議，實際建議：{keys}"
        )
        suggestion = next(s for s in report.suggestions if s.key == "lower_min_seconds")
        assert "1.1" in suggestion.config_hint, suggestion.config_hint
        assert suggestion.severity == "warn"

    def test_breaths_far_below_threshold_do_not_trigger_suggestion(self) -> None:
        """
        情境 A：0.3~0.8 秒的喘氣聲被淘汰 → 門檻正確，不該建議改動。
        這是最重要的反向測試：分析工具不能「看到淘汰就想放寬」。
        """
        rows = [row("ok", "通過", duration=4.0) for _ in range(20)]
        rows += [
            row("rejected", "過短 0.30s < 1.5s", duration=0.30),
            row("rejected", "過短 0.45s < 1.5s", duration=0.45),
            row("rejected", "過短 0.60s < 1.5s", duration=0.60),
            row("rejected", "過短 0.70s < 1.5s", duration=0.70),
            row("rejected", "過短 0.80s < 1.5s", duration=0.80),
        ]
        report = analyze(rows, **DEFAULTS)
        keys = {s.key for s in report.suggestions}
        assert "lower_min_seconds" not in keys, (
            "遠離門檻的淘汰是門檻正確運作的證據，不該建議放寬"
        )
        assert "min_seconds_ok" in keys, "應該明確回報門檻是對的"

    def test_too_few_boundary_samples_do_not_trigger(self) -> None:
        """只有 2 筆邊界樣本時，任何統計都是噪音。"""
        rows = [row("ok", "通過", duration=4.0) for _ in range(20)]
        rows += [
            row("rejected", "過短 1.40s < 1.5s", duration=1.40),
            row("rejected", "過短 1.45s < 1.5s", duration=1.45),
        ]
        report = analyze(rows, **DEFAULTS)
        assert "lower_min_seconds" not in {s.key for s in report.suggestions}

    def test_long_files_near_ceiling_trigger_raise_suggestion(self) -> None:
        rows = [row("ok", "通過", duration=4.0) for _ in range(10)]
        rows += [
            row("rejected", "過長 16.20s > 15.0s", duration=16.2),
            row("rejected", "過長 15.80s > 15.0s", duration=15.8),
            row("rejected", "過長 15.30s > 15.0s", duration=15.3),
            row("rejected", "過長 15.10s > 15.0s", duration=15.1),
        ]
        report = analyze(rows, **DEFAULTS)
        s = next((x for x in report.suggestions if x.key == "raise_max_seconds"), None)
        assert s is not None
        assert "Phase 4" in s.detail, "應提醒拉長對 ASR 對齊的負面影響"

    def test_quiet_whisper_level_files_trigger_rms_suggestion(self) -> None:
        """
        耳語／內心戲台詞被 RMS 門檻誤殺 —— 這類素材情感價值最高，
        必須被明確點出來（並提醒 Phase 2 會處理音量）。
        """
        rows = [row("ok", "通過", duration=4.0, rms=-14.0) for _ in range(10)]
        rows += [
            row("rejected", f"訊號過弱 RMS {v} < -50.0 dBFS", duration=3.0, rms=v)
            for v in (-52.0, -54.0, -56.0, -51.5)
        ]
        report = analyze(rows, **DEFAULTS)
        s = next((x for x in report.suggestions if x.key == "lower_min_rms"), None)
        assert s is not None, f"應建議放寬 RMS 下限，實際：{[x.key for x in report.suggestions]}"
        assert "Phase 2" in s.detail
        assert "耳語" in s.detail

    def test_clipping_does_not_suggest_threshold_change(self) -> None:
        """削波是物理損壞，降門檻救不回來 —— 不該給出 config_hint。"""
        rows = [row("ok", "通過", duration=4.0) for _ in range(10)]
        rows += [
            row("rejected", "削波風險 peak 0.00 > -0.5 dBFS", duration=3.0, peak=0.0)
            for _ in range(6)
        ]
        report = analyze(rows, **DEFAULTS)
        s = next((x for x in report.suggestions if x.key == "clipping_source"), None)
        assert s is not None
        assert s.config_hint == "", "削波不該給設定片段（沒有門檻能修好它）"
        assert "物理損壞" in s.detail or "救不回來" in s.detail


# ---------------------------------------------------------------------------
# 3. 素材量體檢
# ---------------------------------------------------------------------------
class TestMaterialVolume:
    def test_too_little_material_is_flagged(self) -> None:
        rows = [row("ok", "通過", duration=10.0) for _ in range(6)]  # 60 秒
        report = analyze(rows, **DEFAULTS)
        assert "too_little_material" in {s.key for s in report.suggestions}
        assert report.ok_seconds == pytest.approx(60.0)

    def test_good_volume_range_is_not_flagged(self) -> None:
        rows = [row("ok", "通過", duration=5.0) for _ in range(240)]  # 20 分鐘
        report = analyze(rows, **DEFAULTS)
        keys = {s.key for s in report.suggestions}
        assert "too_little_material" not in keys
        assert "too_much_material" not in keys

    def test_too_much_material_suggests_subset(self) -> None:
        rows = [row("ok", "通過", duration=10.0) for _ in range(500)]  # 83 分鐘
        report = analyze(rows, **DEFAULTS)
        s = next((x for x in report.suggestions if x.key == "too_much_material"), None)
        assert s is not None
        assert "Quality Over Quantity" in s.detail or "15 分鐘" in s.detail

    def test_thin_groups_are_flagged_in_multicharacter_project(self) -> None:
        rows = [row("ok", "通過", duration=10.0, group="凛") for _ in range(60)]
        rows += [row("ok", "通過", duration=5.0, group="桜", src="b.wav") for _ in range(4)]
        report = analyze(rows, **DEFAULTS)
        s = next((x for x in report.suggestions if x.key == "thin_groups"), None)
        assert s is not None
        assert "桜" in s.detail and "凛" not in s.detail.split("\n")[0]

    def test_single_group_does_not_trigger_thin_group_warning(self) -> None:
        """單一角色專案不該收到『角色素材不足』的通知。"""
        rows = [row("ok", "通過", duration=5.0, group="default") for _ in range(30)]
        report = analyze(rows, **DEFAULTS)
        assert "thin_groups" not in {s.key for s in report.suggestions}

    def test_decode_errors_are_flagged(self) -> None:
        rows = [row("ok", "通過", duration=5.0) for _ in range(10)]
        rows += [row("error", "解碼失敗", src=f"e{i}.wav") for i in range(6)]
        report = analyze(rows, **DEFAULTS)
        assert "decode_errors" in {s.key for s in report.suggestions}
        assert report.error_count == 6


# ---------------------------------------------------------------------------
# 4. 邊界情況
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_manifest_does_not_crash(self) -> None:
        report = analyze([], **DEFAULTS)
        assert report.total == 0
        assert report.ok_ratio == 0.0
        text = render_report(report, **DEFAULTS)
        assert "沒有任何資料" in text
        assert "尚無合格樣本" in text

    def test_all_rejected_does_not_crash(self) -> None:
        rows = [row("rejected", "過短 0.5s < 1.5s", duration=0.5) for _ in range(10)]
        report = analyze(rows, **DEFAULTS)
        assert report.ok_count == 0
        assert report.ok_ratio == 0.0
        assert render_report(report, **DEFAULTS)

    def test_skipped_only_does_not_crash(self) -> None:
        rows = [row("skipped", "容器檔，需先解包") for _ in range(5)]
        report = analyze(rows, **DEFAULTS)
        assert report.skipped_count == 5

    def test_report_is_json_serializable(self) -> None:
        rows = [row("ok", "通過", duration=5.0), row("rejected", "過短 0.4s", duration=0.4)]
        payload = report_as_dict(analyze(rows, **DEFAULTS))
        text = json.dumps(payload, ensure_ascii=False)
        assert "counts" in text and "suggestions" in text

    def test_render_includes_all_sections(self) -> None:
        rows = [row("ok", "通過", duration=5.0, group="凛") for _ in range(20)]
        rows += [row("ok", "通過", duration=5.0, group="桜", src="b.wav") for _ in range(4)]
        rows += [
            row("rejected", "過短 1.4s < 1.5s", duration=1.4, group="凛") for _ in range(4)
        ]
        text = render_report(analyze(rows, **DEFAULTS), **DEFAULTS)
        for expected in ("現行門檻", "時長分布", "淘汰原因分布", "逐角色合格時長", "建議"):
            assert expected in text, f"報告缺少區塊：{expected}"

    def test_empty_manifest_render_includes_no_data_suggestion(self) -> None:
        text = render_report(analyze([], **DEFAULTS), **DEFAULTS)
        assert "沒有任何資料" in text
        assert "convert" in text, "應提示下一步該跑什麼"


# ---------------------------------------------------------------------------
# 5. 透過 CLI
# ---------------------------------------------------------------------------
def _scaffold(root: Path) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)
    data = json.loads((REPO / "configs" / "pipeline.default.json").read_text(encoding="utf-8"))
    data["grouping"] = {"by": "folder"}
    (root / "configs" / "pipeline.local.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def _run_cli(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env[cfgmod.ENV_REPO_ROOT] = str(root)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "phase1_prep.py"), *args],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


def _write_manifest(root: Path, body: str) -> None:
    header = (
        "status,reason,group,src_path,src_format,src_format_label,src_bytes,src_sr,"
        "src_channels,duration_sec,out_path,out_sr,out_channels,out_bit_depth,"
        "peak_dbfs,rms_dbfs,decoder,error\n"
    )
    (root / "data" / "_reports" / "manifest_phase1.csv").write_text(
        header + body, encoding="utf-8-sig"
    )


class TestTuneCli:
    def test_tune_without_manifest_gives_actionable_hint(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir)
        proc = _run_cli(["tune"], root)
        assert proc.returncode == 1
        assert "convert" in proc.stdout
        assert "Traceback" not in proc.stderr

    def test_tune_reports_and_writes_json(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir)
        lines = []
        # 20 個合格（每 6 秒）
        for i in range(20):
            lines.append(
                f"ok,通過,凛,data/00_raw/ok{i}.wav,RIFF/WAVE,x,1000,48000,1,6.000,"
                f"data/02_wav24_48k/凛/ok{i}.wav,48000,1,24,-6.0,-14.0,soundfile,"
            )
        # 4 個邊界被淘汰（1.3~1.45 秒）→ 應建議放寬
        for i, dur in enumerate(("1.30", "1.35", "1.40", "1.45")):
            lines.append(
                f"rejected,過短 {dur}s < 1.5s（喘氣/單字哼聲）,凛,data/00_raw/s{i}.wav,"
                f"RIFF/WAVE,x,1000,48000,1,{dur},,,,,-6.0,-14.0,soundfile,"
            )
        _write_manifest(root, "\n".join(lines) + "\n")

        proc = _run_cli(["tune"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert "門檻調校分析" in proc.stdout
        assert "合格 20" in proc.stdout
        assert "lower_min_seconds" not in proc.stdout  # 這是內部代號，不該外洩到畫面
        assert "最短時長" in proc.stdout

        payload = json.loads((root / "data" / "_reports" / "phase1_tune.json").read_text("utf-8"))
        assert payload["counts"]["ok"] == 20
        assert payload["counts"]["rejected"] == 4
        assert payload["ok_ratio"] == pytest.approx(20 / 24, abs=0.001)
        keys = {s["key"] for s in payload["suggestions"]}
        assert "lower_min_seconds" in keys
        assert payload["reject_categories"]["too_short"]["boundary_count"] == 4

    def test_tune_dry_run_does_not_write_json(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir)
        _write_manifest(root, "ok,通過,default,data/00_raw/a.wav,x,x,1,48000,1,5.0,,48000,1,24,-6,-14,sf,\n")
        proc = _run_cli(["tune", "--dry-run"], root)
        assert proc.returncode == 0, proc.stdout
        assert not (root / "data" / "_reports" / "phase1_tune.json").exists()

    def test_tune_works_end_to_end_after_convert(self, tmp_dir: Path) -> None:
        """整合測試：convert 產生 manifest → tune 能讀懂它。"""
        from conftest import synth_voice
        from _audioio import write_wav

        root = _scaffold(tmp_dir)
        raw = root / "data" / "00_raw" / "凛"
        raw.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            write_wav(raw / f"ok{i}.wav", synth_voice(3.0, amp=0.5), 48_000, subtype="PCM_16")
        write_wav(raw / "short.wav", synth_voice(1.0, amp=0.5), 48_000, subtype="PCM_16")

        proc = _run_cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, proc.stdout

        proc = _run_cli(["tune"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert "門檻調校分析" in proc.stdout
        assert "合格 6" in proc.stdout
        payload = json.loads((root / "data" / "_reports" / "phase1_tune.json").read_text("utf-8"))
        assert payload["counts"]["ok"] == 6
        assert payload["counts"]["rejected"] == 1
        assert payload["reject_categories"]["too_short"]["count"] == 1
