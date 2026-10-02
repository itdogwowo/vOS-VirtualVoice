"""
vOS-VirtualVoice — Phase 1 全鏈路整合測試

這個檔案只問一個問題：**六個子命令串起來到底會不會動？**

每個子命令各自都有單元測試，但真正會壞的地方是**交接處**：
  * unpack 產生的檔名，convert 找不找得到？
  * convert 的 manifest，tune / report 讀不讀得懂？
  * audition 抽的是 02_wav24_48k，group_by=folder 的結構它認不認得？
  * 多角色專案的角色名，會不會在某一站掉掉？

所以這個檔案用**假 vgmstream**（tests/fake_vgmstream.py）走完真實流程：
容器 → unpack → convert → tune → audition → report，
並且在每一步驗證產物與下一步的期待一致。

沒有這個測試，Phase 1 的六個零件可能各自完美但拼不起來。
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfgmod  # noqa: E402
from conftest import make_fake_exe, run_cli  # noqa: E402
from fake_vgmstream import count_subsongs  # noqa: E402


def _scaffold(root: Path, *, fake_exe: Path | None = None, grouping: str = "folder") -> Path:
    """建立最小可用的 repo 骨架（configs + 五個 data 子資料夾）。"""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)

    data = json.loads((REPO / "configs" / "pipeline.default.json").read_text(encoding="utf-8"))
    data["grouping"] = {"by": grouping}
    if fake_exe is not None:
        data["unpack"]["vgmstream_bin"] = str(fake_exe)
    (root / "configs" / "pipeline.local.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def _cli(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return run_cli(args, root)


def _container(path: Path, subsongs: int, seconds: float, amp: float = 0.5) -> Path:
    """
    建立假容器。檔名編碼替身要模擬的參數：
      fake_subsongs=N  產生 N 個 subsong
      fake_dur=S       第 1 個 subsong 的長度（之後每個 +0.2 秒）
      fake_amp=A       振幅（模擬耳語或過大）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"AFS2" + b"\x00" * 64)
    assert count_subsongs(path) == subsongs
    return path


@pytest.fixture
def pipeline(tmp_dir: Path) -> tuple[Path, Path]:
    """建立一個含假 vgmstream 的沙箱 repo。"""
    fake = make_fake_exe(tmp_dir / "bin")
    root = _scaffold(tmp_dir / "repo", fake_exe=fake)
    return root, fake


# ===========================================================================
# 完整鏈路
# ===========================================================================
class TestFullPipeline:
    def test_container_to_report_end_to_end(self, pipeline: tuple[Path, Path]) -> None:
        """
        最重要的一條測試：容器 → unpack → convert → tune → audition → report。

        設計素材時刻意讓每一站都有事做：
          * 凜：8 個 subsong，每個 3~4.4 秒 → 全部合格
          * 桜：4 個 subsong，其中含極短（被時長篩選淘汰）
          * 旁白：1 個 subsong，6 秒 → 合格但素材量不足（驗證「逐角色不足」警告）
        """
        root, _fake = pipeline
        raw = root / "data" / "00_raw"

        _container(raw / "凛" / "voice_fake_subsongs=8_fake_dur=3.awb", 8, 3.0)
        _container(raw / "桜" / "voice_fake_subsongs=3_fake_dur=2.awb", 3, 2.0)
        # 第 4 個 subsong 只有 0.4 秒 → 會被時長規則淘汰
        _container(raw / "桜" / "short_fake_subsongs=1_fake_dur=0.4.awb", 1, 0.4)
        _container(raw / "ナレーション" / "narr_fake_subsongs=2_fake_dur=6.awb", 2, 6.0)

        # ---- 站 1：scan 認得所有容器 ----
        proc = _cli(["scan"], root)
        assert proc.returncode == 0, proc.stdout
        assert "需先解包：4 個" in proc.stdout
        assert "可直接解碼：0 個" in proc.stdout

        # ---- 站 2：unpack ----
        proc = _cli(["unpack"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        # 8 + 3 + 1 + 2 = 14 個 subsong
        assert "可用產物共 14 個" in proc.stdout

        unpacked = sorted((root / "data" / "01_unpacked").rglob("*.wav"))
        assert len(unpacked) == 14, [p.name for p in unpacked]
        # 檔名要帶 subsong 編號（證明 ?s 萬用字元生效）
        assert any("#2" in p.name for p in unpacked)

        # ---- 站 3：convert（跨目錄結構 + 篩選）----
        proc = _cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

        out = root / "data" / "02_wav24_48k"
        for group in ("凛", "桜", "ナレーション"):
            assert (out / group).is_dir(), f"群組目錄缺失：{group}"
        # 凜 8 + 桜 3 + 旁白 2 = 13 合格；0.4 秒那個被淘汰
        assert len(list((out / "凛").glob("*.wav"))) == 8
        assert len(list((out / "桜").glob("*.wav"))) == 3
        assert len(list((out / "ナレーション").glob("*.wav"))) == 2

        rejected = list((root / "data" / "99_rejected").rglob("*.wav"))
        assert len(rejected) == 1, [p.name for p in rejected]
        # 淘汰檔必須保留「角色 → 容器 → subsong」的完整結構，
        # 否則多角色專案的同名檔案會互相覆蓋
        rejected_rel = rejected[0].relative_to(root / "data" / "99_rejected")
        assert rejected_rel.parts[0] == "桜", rejected_rel
        assert "short" in rejected_rel.parts[1]

        # 規格驗證
        import soundfile as sf

        info = sf.info(str(next((out / "凛").glob("*.wav"))))
        assert (info.samplerate, info.subtype, info.channels) == (48_000, "PCM_24", 1)

        # ---- 站 4：convert 的 manifest 能被 tune 讀懂 ----
        manifest = root / "data" / "_reports" / "manifest_phase1.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 14
        groups_in_manifest = {r["group"] for r in rows if r["status"] == "ok"}
        assert groups_in_manifest == {"凛", "桜", "ナレーション"}

        proc = _cli(["tune"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert "合格 13" in proc.stdout
        payload = json.loads((root / "data" / "_reports" / "phase1_tune.json").read_text("utf-8"))
        assert payload["counts"]["ok"] == 13
        assert payload["reject_categories"]["too_short"]["count"] == 1

        # ---- 站 5：audition 認得 group_by=folder 的產物結構 ----
        proc = _cli(["audition", "--per-group", "2", "--seed", "7"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert "3 個群組" in proc.stdout, proc.stdout
        # 每組抽 2 個 → 共 6 段（桜 只有 3 個合格檔，仍抽得到 2 個）
        assert "共 6 個" in proc.stdout
        for group in ("凛", "桜", "ナレーション"):
            assert group in proc.stdout

        reels = list((root / "data" / "_reports").glob("audition_*.wav"))
        lists = list((root / "data" / "_reports").glob("audition_*.md"))
        assert len(reels) == 1 and len(lists) == 1
        assert reels[0].stat().st_size > 10_000, "試聽帶應該有實際音訊內容"

        # 對照清單必須涵蓋所有抽中樣本
        list_text = lists[0].read_text(encoding="utf-8")
        for group in ("凛", "桜", "ナレーション"):
            assert group in list_text

        # ---- 站 6：report 收斂全部資訊 ----
        proc = _cli(["report"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

        report_md = (root / "data" / "_reports" / "PHASE1_REPORT.md").read_text("utf-8")
        report_json = json.loads(
            (root / "data" / "_reports" / "PHASE1_REPORT.json").read_text("utf-8")
        )

        assert "Phase 1 綜合報告" in report_md
        assert report_json["ok_count"] == 13
        assert set(report_json["group_minutes"]) == {"凛", "桜", "ナレーション"}
        assert report_json["scan"]["pending_unpack"] == 0, "完整跑完 unpack 後不該還有待處理容器"
        assert report_json["scan"]["already_unpacked"] == 4, "4 個容器都該標為已解包"
        assert len(report_json["audition_reels"]) == 1
        assert report_json["tune"] is not None

        # 報告必須包含所有章節
        for section in (
            "## 1. 總覽",
            "## 2. 原始素材",
            "## 3. 逐角色素材量",
            "## 5. 進入 Phase 2 的驗收條件",
            "## 7. 產物位置",
            "## 8. 下一步指令",
        ):
            assert section in report_md, f"報告缺少章節：{section}"

        # 旁白只有 12 秒（< 3 分鐘）→ 應該被標為不足
        assert "ナレーション" in report_md
        assert "不足" in report_md

    def test_pipeline_is_idempotent(self, pipeline: tuple[Path, Path]) -> None:
        """
        重跑整條鏈路不該產生重複的合格產物。

        設計細節：`convert` 預設不覆寫既有產物，所以重跑不會爆量；
        `unpack` 則會產生 `__1` 序號的新檔（安全優先），但那些新檔在
        convert 階段會因為「目標已存在」而被略過 —— 所以合格區數量不變。
        """
        root, _fake = pipeline
        _container(root / "data" / "00_raw" / "hero" / "v_fake_subsongs=3_fake_dur=3.awb", 3, 3.0)

        first = _cli(["all"], root)
        assert first.returncode == 0, f"{first.stdout}\n{first.stderr}"
        out = root / "data" / "02_wav24_48k"
        count_after_first = len(list(out.rglob("*.wav")))
        assert count_after_first == 3

        second = _cli(["all"], root)
        assert second.returncode == 0, f"{second.stdout}\n{second.stderr}"
        assert len(list(out.rglob("*.wav"))) == count_after_first, (
            "重跑不該讓合格區數量增加"
        )

        # manifest 是重新產生的，數字應該一致
        with (root / "data" / "_reports" / "manifest_phase1.csv").open(
            encoding="utf-8-sig", newline=""
        ) as fh:
            rows = list(csv.DictReader(fh))
        assert sum(1 for r in rows if r["status"] == "ok") == 3

    def test_dry_run_chain_writes_nothing(self, pipeline: tuple[Path, Path]) -> None:
        """
        --dry-run 走完 unpack + convert，不該有任何產物。

        注意：unpack --dry-run **不會**產生 01_unpacked 的檔案，所以後面的
        convert --dry-run 會正常回報「找不到可轉換的檔案」（exit 1）——
        那是正確行為，不是錯誤。這裡只驗證「沒有東西被寫出來」。
        """
        root, _fake = pipeline
        _container(root / "data" / "00_raw" / "hero" / "v_fake_subsongs=3_fake_dur=3.awb", 3, 3.0)

        unpack_proc = _cli(["unpack", "--dry-run"], root)
        assert unpack_proc.returncode == 0, unpack_proc.stdout
        assert "dry-run" in unpack_proc.stdout

        convert_proc = _cli(["convert", "--dry-run"], root)
        assert "Traceback" not in convert_proc.stderr

        assert not list((root / "data" / "01_unpacked").rglob("*.wav"))
        assert not list((root / "data" / "02_wav24_48k").rglob("*.wav"))
        assert not (root / "data" / "_reports" / "manifest_phase1.csv").exists()

    def test_unpack_then_convert_reports_rejections(self, pipeline: tuple[Path, Path]) -> None:
        """
        真實 unpack（非 dry-run）→ convert 必須看到解包產物並做出篩選判斷。
        這是「解包 → 篩選」的交接測試：替身產生的音訊約 0.05 秒，
        必然被時長規則淘汰 —— 但那證明它真的被處理了。
        """
        root, _fake = pipeline
        _container(root / "data" / "00_raw" / "hero" / "v_fake_subsongs=2.awb", 2, 0.05)

        assert _cli(["unpack"], root).returncode == 0
        assert len(list((root / "data" / "01_unpacked").rglob("*.wav"))) == 2

        proc = _cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert "過短" in proc.stdout

        with (root / "data" / "_reports" / "manifest_phase1.csv").open(
            encoding="utf-8-sig", newline=""
        ) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert all(r["status"] == "rejected" for r in rows)

    def test_single_character_chain_keeps_flat_layout(self, pipeline: tuple[Path, Path]) -> None:
        """grouping.by=none 時整條鏈路都不該出現群組子資料夾。"""
        root, fake = pipeline
        # 重建設定為單一角色
        _scaffold(root, fake_exe=fake, grouping="none")
        _container(root / "data" / "00_raw" / "v_fake_subsongs=3_fake_dur=3.awb", 3, 3.0)

        proc = _cli(["all"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

        out = root / "data" / "02_wav24_48k"
        wavs = sorted(out.glob("*.wav"))
        assert len(wavs) == 3, [p.name for p in wavs]
        assert not (out / "default").exists(), "單一角色不該有 default/ 資料夾"

        proc = _cli(["audition", "--per-group", "3"], root)
        assert proc.returncode == 0, proc.stdout
        assert "1 個群組" in proc.stdout

    def test_quiet_whisper_container_needs_rms_gate_tuning(
        self, pipeline: tuple[Path, Path]
    ) -> None:
        """
        耳語級素材（振幅 0.004 ≈ -48 dBFS peak / -51 dBFS RMS）在**預設門檻**
        下會被「訊號過弱」淘汰 —— 而這正是遊戲刻意錄小聲的耳語／內心戲台詞。

        這個測試同時固定兩件事：
          1. 預設門檻確實會誤殺這類素材（工具要有能力指出來）
          2. 把 min_rms_dbfs 放寬到 -60 之後，同一批素材完整通過全鏈路
             （證明「不該在 Phase 1 丟掉它們」是可行動的建議，不是空話）

        音量問題本來就該留給 Phase 2 的響度標準化處理，不在 Phase 1 解決。
        """
        root, _fake = pipeline
        _container(
            root / "data" / "00_raw" / "凛"
            / "whisper_fake_subsongs=3_fake_dur=3_fake_amp=0.004.awb",
            3,
            3.0,
        )
        assert _cli(["unpack"], root).returncode == 0

        # ---- 1. 預設門檻：會被誤殺 ----
        proc = _cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, proc.stdout
        assert "訊號過弱" in proc.stdout, proc.stdout
        with (root / "data" / "_reports" / "manifest_phase1.csv").open(
            encoding="utf-8-sig", newline=""
        ) as fh:
            rows = list(csv.DictReader(fh))
        assert rows and all(r["status"] == "rejected" for r in rows)

        # tune 必須能把這件事講出來（而不是讓你以為素材就是爛）
        tune = _cli(["tune"], root)
        assert tune.returncode == 0, tune.stdout
        payload = json.loads((root / "data" / "_reports" / "phase1_tune.json").read_text("utf-8"))
        keys = {s["key"] for s in payload["suggestions"]}
        assert "lower_min_rms" in keys, f"tune 應該建議放寬 RMS 下限，實際：{keys}"

        # ---- 2. 放寬門檻後：完整通過 ----
        config_path = root / "configs" / "pipeline.local.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data["quality_gate"]["min_rms_dbfs"] = -60.0
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        proc = _cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, proc.stdout
        assert len(list((root / "data" / "02_wav24_48k").rglob("*.wav"))) == 3

        probe = _cli(["report"], root)
        assert probe.returncode == 0, probe.stdout
        report = json.loads((root / "data" / "_reports" / "PHASE1_REPORT.json").read_text("utf-8"))
        assert report["ok_count"] == 3

    def test_clipped_container_is_rejected_with_reason(self, pipeline: tuple[Path, Path]) -> None:
        """波形被削平的素材會被品質關卡擋下，且理由要進 manifest。"""
        root, _fake = pipeline
        # fake_clip：不歸一化，讓波形真的爆掉被削平
        _container(
            root / "data" / "00_raw" / "凛"
            / "loud_fake_clip_fake_subsongs=2_fake_dur=3_fake_amp=2.5.awb",
            2,
            3.0,
        )
        assert _cli(["unpack"], root).returncode == 0
        proc = _cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, proc.stdout

        with (root / "data" / "_reports" / "manifest_phase1.csv").open(
            encoding="utf-8-sig", newline=""
        ) as fh:
            rows = list(csv.DictReader(fh))
        assert rows
        assert any("削波" in r["reason"] for r in rows), [r["reason"] for r in rows]
        assert all(r["status"] == "rejected" for r in rows), [r["status"] for r in rows]
        # 合格區不該收到爆掉的波形
        assert not list((root / "data" / "02_wav24_48k").rglob("*.wav"))


# ===========================================================================
# 報告的驗收邏輯
# ===========================================================================
class TestReportChecks:
    def test_empty_repo_report_says_not_ready(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir / "empty", grouping="none")
        proc = _cli(["report"], root)
        assert proc.returncode == 0, proc.stdout
        assert "尚未產生 manifest" in proc.stdout

        md = (root / "data" / "_reports" / "PHASE1_REPORT.md").read_text("utf-8")
        assert "尚未產生 manifest" in md
        assert "把遊戲語音放進" in md
        assert "達標" in md or "未達標" in md

    def test_report_flags_unpacked_containers(self, tmp_dir: Path) -> None:
        """00_raw 還有容器未解包時，報告必須明確指出。"""
        fake = make_fake_exe(tmp_dir / "bin")
        root = _scaffold(tmp_dir / "repo", fake_exe=fake, grouping="none")
        _container(root / "data" / "00_raw" / "v_fake_subsongs=3_fake_dur=3.awb", 3, 3.0)

        proc = _cli(["report"], root)
        assert proc.returncode == 0, proc.stdout
        payload = json.loads((root / "data" / "_reports" / "PHASE1_REPORT.json").read_text("utf-8"))
        assert payload["scan"]["need_unpack"] == 1
        assert payload["scan"]["pending_unpack"] == 1
        assert payload["scan"]["already_unpacked"] == 0
        check = next(c for c in payload["checks"] if "待處理容器" in c["label"])
        assert check["passed"] is False
        assert "unpack" in check["detail"]

    def test_report_no_scan_flag_skips_raw_scan(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir / "repo", grouping="none")
        proc = _cli(["report", "--no-scan"], root)
        assert proc.returncode == 0, proc.stdout
        payload = json.loads((root / "data" / "_reports" / "PHASE1_REPORT.json").read_text("utf-8"))
        assert payload["scan"] is None
        # 沒有 scan 就不該有「待處理容器」這項檢查
        assert not any("待處理容器" in c["label"] for c in payload["checks"])

    def test_report_shows_group_with_zero_qualified_material(self, tmp_dir: Path) -> None:
        """
        回歸測試：某個角色**全數被淘汰**時，報告必須明確顯示，不能從表上消失。

        這個 bug 真實發生過：逐角色表格以「合格時長」為主鍵，某角色全滅時
        它就不會出現在表上 —— 於是「這個角色完全沒素材」這個最重要的警訊
        在報告裡變成一塊空白。表格必須以 manifest 裡出現過的群組為主。
        """
        fake = make_fake_exe(tmp_dir / "bin")
        root = _scaffold(tmp_dir / "repo", fake_exe=fake, grouping="folder")
        # 凛：正常素材；桜：只有極短素材（全滅）
        _container(root / "data" / "00_raw" / "凛" / "v_fake_subsongs=3_fake_dur=3.awb", 3, 3.0)
        _container(root / "data" / "00_raw" / "桜" / "s_fake_subsongs=2_fake_dur=0.4.awb", 2, 0.4)

        assert _cli(["all"], root).returncode == 0
        proc = _cli(["report"], root)
        assert proc.returncode == 0, proc.stdout

        md = (root / "data" / "_reports" / "PHASE1_REPORT.md").read_text("utf-8")
        payload = json.loads((root / "data" / "_reports" / "PHASE1_REPORT.json").read_text("utf-8"))

        assert "## 3. 逐角色素材量" in md
        assert "桜" in md, "全數被淘汰的角色仍必須出現在報告上"
        assert "無合格素材" in md
        assert payload["group_counts"]["桜"]["ok"] == 0
        assert payload["group_counts"]["桜"]["rejected"] == 2

        # 驗收條件必須把「桜 沒有素材」算成未達標
        check = next(c for c in payload["checks"] if "每個角色" in c["label"])
        assert check["passed"] is False
        assert "桜" in check["detail"]

    def test_report_per_group_check_absent_for_single_character(self, tmp_dir: Path) -> None:
        """單一角色專案不該出現『每個角色 ≥ N 分鐘』這項檢查。"""
        fake = make_fake_exe(tmp_dir / "bin")
        root = _scaffold(tmp_dir / "repo", fake_exe=fake, grouping="none")
        _container(root / "data" / "00_raw" / "v_fake_subsongs=3_fake_dur=3.awb", 3, 3.0)
        assert _cli(["all"], root).returncode == 0
        assert _cli(["report"], root).returncode == 0
        payload = json.loads((root / "data" / "_reports" / "PHASE1_REPORT.json").read_text("utf-8"))
        assert not any("每個角色" in c["label"] for c in payload["checks"])
        # 但逐角色表格仍要出現（只有一列 default），讓你知道素材落在哪一組
        md = (root / "data" / "_reports" / "PHASE1_REPORT.md").read_text("utf-8")
        assert "## 3. 逐角色素材量" in md
