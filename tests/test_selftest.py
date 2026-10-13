"""
vOS-VirtualVoice - 自我測試的測試 (Phase 1)

`selftest` 是「在下載素材之前，先確認工具能動」的那道防線。所以它自己
也必須被驗證 —— 尤其是它最容易犯的兩個錯：

  1. **污染真正的 data/**：若 VOS_REPO_ROOT 沒設好，就會把合成素材寫進
     你真正的素材庫。這個錯誤本專案真的發生過。
  2. **說謊**：檢查項目必須真的在驗證東西。一個永遠回報「通過」的
     selftest 比沒有 selftest 更糟。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import selftest as st  # noqa: E402
from selftest import (  # noqa: E402
    FIXTURES,
    WORKSPACE_PREFIX,
    cleanup_workspace,
    is_workspace,
    make_workspace,
    write_fixtures,
)


class TestFixtures:
    def test_covers_every_filter_rule(self) -> None:
        """
        最重要的一點：合成素材必須涵蓋**每一條篩選規則**。
        只測「合格的會過」等於沒測到篩選 —— 真正會出錯的是「該擋的有沒有擋」。
        """
        expectations = {spec[4] for spec in FIXTURES.values()}
        assert "ok" in expectations
        assert "rejected" in expectations
        assert sum(1 for s in FIXTURES.values() if s[4] == "ok") >= 2
        assert sum(1 for s in FIXTURES.values() if s[4] == "rejected") >= 3

    def test_rejection_reasons_are_distinct(self) -> None:
        """三個淘汰樣本要對應三條不同規則（過短／過長／過弱），不能重複。"""
        specs = [s for s in FIXTURES.values() if s[4] == "rejected"]
        durations = [s[0] for s in specs]
        amps = [s[2] for s in specs]
        assert min(durations) < 1.5, "要有一個過短的樣本（預設門檻 1.5s）"
        assert max(durations) > 15.0, "要有一個過長的樣本（預設門檻 15s）"
        assert min(amps) < 0.01, "要有一個訊號過弱的樣本"

    def test_ok_samples_need_resampling_and_downmixing(self) -> None:
        """合格樣本要涵蓋「升採樣」與「立體聲轉單聲道」兩種處理路徑。"""
        ok_specs = [s for s in FIXTURES.values() if s[4] == "ok"]
        assert 44_100 in {s[1] for s in ok_specs}, "要有一個需要升採樣的樣本"
        assert 2 in {s[3] for s in ok_specs}, "要有一個需要轉單聲道的立體聲樣本"


class TestSynthVoice:
    def test_mono_shape(self) -> None:
        data = st.synth_voice(0.5, 48_000, 0.5, 1)
        assert len(data) == 1
        assert len(data[0]) == 24_000

    def test_stereo_shape(self) -> None:
        data = st.synth_voice(0.5, 48_000, 0.5, 2)
        assert len(data) == 2
        assert len(data[0]) == len(data[1]) == 24_000

    def test_amplitude_is_respected(self) -> None:
        """振幅要真的等於指定值，否則「訊號過弱」的樣本可能根本不會被擋。"""
        data = st.synth_voice(1.0, 48_000, 0.25, 1)
        assert max(abs(v) for v in data[0]) == pytest.approx(0.25, abs=0.01)

    def test_quiet_sample_is_below_default_rms_gate(self) -> None:
        """
        最關鍵的一點：過弱的樣本必須真的低於預設門檻（-50 dBFS RMS），
        否則 selftest 會回報「篩選規則正確」但其實那個樣本根本沒被擋。
        """
        import numpy as np

        data = np.asarray(st.synth_voice(3.0, 48_000, 0.003, 1), dtype=np.float64)
        rms_db = 20 * np.log10(float(np.sqrt(np.mean(data**2))))
        assert rms_db < -50.0, f"過弱樣本的 RMS 應低於 -50 dBFS，實際 {rms_db:.1f}"

    def test_fade_prevents_click_at_edges(self) -> None:
        data = st.synth_voice(1.0, 48_000, 0.5, 1)
        assert abs(data[0][0]) < 1e-6
        assert abs(data[0][-1]) < 1e-6


class TestWorkspace:
    def test_creates_full_repo_skeleton(self, tmp_dir: Path) -> None:
        root = make_workspace(tmp_dir)
        try:
            assert (root / "configs" / "pipeline.local.json").is_file()
            for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
                assert (root / "data" / sub).is_dir(), sub
        finally:
            cleanup_workspace(root)

    def test_uses_real_default_thresholds(self, tmp_dir: Path) -> None:
        """
        臨時 repo 要用**真正的預設門檻**，不能在測試裡另外寫死一組值 ——
        否則 selftest 驗證的是那組寫死的值，不是你實際會用到的設定。
        """
        import json

        root = make_workspace(tmp_dir)
        try:
            data = json.loads(
                (root / "configs" / "pipeline.local.json").read_text(encoding="utf-8")
            )
            reference = json.loads(
                (REPO / "configs" / "pipeline.default.json").read_text(encoding="utf-8")
            )
            assert data["duration_filter"] == reference["duration_filter"]
            assert data["target"] == reference["target"]
            assert data["grouping"]["by"] == "none"
        finally:
            cleanup_workspace(root)

    def test_each_call_makes_a_fresh_workspace(self, tmp_dir: Path) -> None:
        a = make_workspace(tmp_dir)
        b = make_workspace(tmp_dir)
        try:
            assert a != b, "兩次呼叫不可共用同一個工作區（會互相污染）"
        finally:
            cleanup_workspace(a)
            cleanup_workspace(b)


class TestCleanupGuard:
    """
    破壞性操作的安全閘門。

    cleanup_workspace 會執行 shutil.rmtree 且使用 ignore_errors=True ——
    一旦參數指錯地方，後果是整棵樹消失且沒有任何錯誤訊息。
    本專案真的發生過一次工作目錄被清空的事故，所以這裡是最重要的測試。
    """

    def test_refuses_to_delete_non_workspace_dir(self, tmp_dir: Path) -> None:
        danger = tmp_dir / "DO-NOT-DELETE"
        danger.mkdir()
        (danger / "important.txt").write_text("keep me", encoding="utf-8")

        cleanup_workspace(danger)  # 必須什麼都不做

        assert danger.is_dir(), "不符前綴的目錄被刪除了 —— 這是嚴重缺陷"
        assert (danger / "important.txt").is_file()

    def test_deletes_real_workspace(self, tmp_dir: Path) -> None:
        ws = tmp_dir / f"{WORKSPACE_PREFIX}abc123"
        ws.mkdir()
        (ws / "x.txt").write_text("x", encoding="utf-8")
        assert is_workspace(ws)
        cleanup_workspace(ws)
        assert not ws.exists()

    @pytest.mark.parametrize("bad", ["", ".", "..", None])
    def test_never_raises_on_dangerous_input(self, bad, tmp_dir: Path) -> None:
        cleanup_workspace(bad)  # type: ignore[arg-type]

    def test_repo_survives_cleanup_of_repo_path(self) -> None:
        """就算把 repo 根目錄傳進去，也必須因為不符前綴而被拒絕。"""
        cleanup_workspace(REPO)
        assert (REPO / "scripts" / "phase1_prep.py").is_file()

    def test_is_workspace_rejects_files(self, tmp_dir: Path) -> None:
        f = tmp_dir / f"{WORKSPACE_PREFIX}file"
        f.write_text("x", encoding="utf-8")
        assert not is_workspace(f), "檔案不可被當成工作區"


class TestWriteFixtures:
    def test_writes_readable_wavs_with_correct_specs(self, tmp_dir: Path) -> None:
        import soundfile as sf

        count = write_fixtures(tmp_dir)
        assert count == len(FIXTURES)
        for name, (seconds, sr, amp, channels, _expected) in FIXTURES.items():
            info = sf.info(str(tmp_dir / name))
            assert info.samplerate == sr, name
            assert info.channels == channels, name
            assert info.frames == pytest.approx(seconds * sr, rel=0.01), name


class TestRunSelftest:
    def test_selftest_passes_and_cleans_up(self) -> None:
        """核心斷言：必須回報通過，且預設不留下任何東西。"""
        report = st.run_selftest(quiet=True)
        assert report.passed, [(c.label, c.detail, c.hint) for c in report.failed]
        assert len(report.checks) >= 8, "檢查項目太少，覆蓋不足"
        if report.workspace is not None:
            assert not report.workspace.exists(), "預設應刪除臨時工作區"

    def test_selftest_did_not_touch_real_data(self) -> None:
        """
        紅線：selftest 的合成素材**絕不可**出現在真正的 data/。

        判斷方式是檢查「selftest 自己的檔名」（FIXTURES 的 key）——
        而不是「目錄是否為空」。理由是目錄裡可能有其他無關的殘留
        （例如先前手動診斷留下的探針目錄），那不該讓這條測試誤報，
        否則真正的污染反而會被噪音蓋掉。
        """
        report = st.run_selftest(quiet=True)
        assert report.passed, "自我測試本身要先通過"

        fixture_names = set(FIXTURES)
        for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
            target = REPO / "data" / sub
            if not target.is_dir():
                continue
            polluted = [
                p.name for p in target.iterdir() if p.name in fixture_names
            ]
            assert not polluted, f"selftest 污染了真正的 data/{sub}：{polluted}"

    def test_missing_external_tools_are_notes_not_failures(self) -> None:
        """
        vgmstream / ffmpeg 缺席**不該**讓自我測試失敗 ——
        Phase 1 對純 WAV 素材本來就不需要它們。
        """
        report = st.run_selftest(quiet=True)
        assert report.passed
        assert report.notes, "未安裝外部工具時應該要有提醒"