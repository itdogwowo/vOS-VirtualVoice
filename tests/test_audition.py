"""
vOS-VirtualVoice — 人工試聽抽樣測試 (Phase 1, 驗收步驟)

守著的幾個重點：
  1. 分層抽樣：素材量懸殊時，小群組（旁白、配角）不能被大群組淹沒
  2. 可重現：同一組素材固定種子 → 固定樣本，修完再聽才有意義
  3. 響度對齊：不寫回合格區，只在試聽帶上生效，且增益有上下限
  4. 壞檔不中斷：單一樣本讀不到不能讓整條試聽帶失敗
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config as cfgmod  # noqa: E402
from _audioio import measure, write_wav  # noqa: E402
from audition import (  # noqa: E402
    build_reel,
    filesystem_safe,
    list_candidates,
    plan_samples,
)
from conftest import synth_voice  # noqa: E402


def make_tree(root: Path, spec: dict[str, int]) -> None:
    """依 {群組: 檔案數} 建立假的 02_wav24_48k 結構（每檔 2 秒）。"""
    for group, count in spec.items():
        for i in range(count):
            path = root / group / f"{group}_{i:03d}.wav"
            write_wav(path, synth_voice(2.0, amp=0.5), 48_000, subtype="PCM_24")


# ---------------------------------------------------------------------------
# 1. 候選清單
# ---------------------------------------------------------------------------
class TestListCandidates:
    def test_groups_by_top_level_folder(self, tmp_dir: Path) -> None:
        make_tree(tmp_dir, {"heroine_A": 3, "heroine_B": 2})
        groups = list_candidates(tmp_dir)
        assert set(groups) == {"heroine_A", "heroine_B"}
        assert len(groups["heroine_A"]) == 3
        assert len(groups["heroine_B"]) == 2

    def test_loose_files_become_default_group(self, tmp_dir: Path) -> None:
        write_wav(tmp_dir / "loose.wav", synth_voice(2.0), 48_000)
        groups = list_candidates(tmp_dir)
        assert set(groups) == {"default"}

    def test_missing_dir_returns_empty_not_error(self, tmp_dir: Path) -> None:
        assert list_candidates(tmp_dir / "nope") == {}

    def test_ignores_non_wav(self, tmp_dir: Path) -> None:
        make_tree(tmp_dir, {"g": 1})
        (tmp_dir / "g" / "notes.txt").write_text("x", encoding="utf-8")
        assert len(list_candidates(tmp_dir)["g"]) == 1


# ---------------------------------------------------------------------------
# 2. 分層抽樣
# ---------------------------------------------------------------------------
class TestPlanSamples:
    def test_stratified_sampling_protects_small_groups(self, tmp_dir: Path) -> None:
        """
        核心價值：凛有 500 條、旁白只有 3 條時，全域隨機幾乎不會抽到旁白。
        分層抽樣必須讓每個群組都有代表。
        """
        make_tree(tmp_dir, {"heroine_A": 40, "narration": 3})
        groups = list_candidates(tmp_dir)
        picked = plan_samples(groups, per_group=5, seed=1)
        by_group: dict[str, int] = {}
        for group, _ in picked:
            by_group[group] = by_group.get(group, 0) + 1
        assert by_group["heroine_A"] == 5
        assert by_group["narration"] == 3, "小群組不足時應全取，而不是被跳過"

    def test_reproducible_with_same_seed(self, tmp_dir: Path) -> None:
        make_tree(tmp_dir, {"g": 20})
        groups = list_candidates(tmp_dir)
        a = [p.name for _, p in plan_samples(groups, per_group=5, seed=42)]
        b = [p.name for _, p in plan_samples(groups, per_group=5, seed=42)]
        assert a == b, "固定種子必須產生相同樣本，否則『修完再聽』無法比較"

    def test_different_seed_gives_different_sample(self, tmp_dir: Path) -> None:
        make_tree(tmp_dir, {"g": 40})
        groups = list_candidates(tmp_dir)
        a = {p.name for _, p in plan_samples(groups, per_group=5, seed=1)}
        b = {p.name for _, p in plan_samples(groups, per_group=5, seed=2)}
        assert a != b

    def test_group_seed_is_independent(self, tmp_dir: Path) -> None:
        """
        每個群組用自己的種子：加一個新群組不該改變既有群組抽到的樣本，
        否則每次補素材都要重聽全部。
        """
        make_tree(tmp_dir, {"g1": 20, "g2": 20})
        groups_a = list_candidates(tmp_dir)
        picked_a = {p.name for g, p in plan_samples(groups_a, per_group=4, seed=7) if g == "g1"}

        make_tree(tmp_dir, {"g3": 20})  # 新增第三個群組
        groups_b = list_candidates(tmp_dir)
        picked_b = {p.name for g, p in plan_samples(groups_b, per_group=4, seed=7) if g == "g1"}
        assert picked_a == picked_b

    def test_include_groups_filter(self, tmp_dir: Path) -> None:
        make_tree(tmp_dir, {"g1": 5, "g2": 5})
        groups = list_candidates(tmp_dir)
        picked = plan_samples(groups, per_group=3, include_groups=["g2"])
        assert {g for g, _ in picked} == {"g2"}

    def test_empty_groups_returns_empty(self) -> None:
        assert plan_samples({}) == []


# ---------------------------------------------------------------------------
# 3. 試聽帶
# ---------------------------------------------------------------------------
class TestBuildReel:
    def _picked(self, tmp_dir: Path, n: int, amp: float = 0.5):
        files = []
        for i in range(n):
            p = tmp_dir / f"src_{i}.wav"
            write_wav(p, synth_voice(2.0, amp=amp), 48_000, subtype="PCM_24")
            files.append(("g", p))
        return files

    def test_reel_duration_includes_gaps(self, tmp_dir: Path) -> None:
        picked = self._picked(tmp_dir, 3)
        plan = build_reel(picked, tmp_dir / "reel.wav", gap_seconds=1.0)
        assert plan.reel_written
        # 3 段 × (2 秒 + 1 秒靜音) = 9 秒
        assert 8.9 < plan.total_seconds < 9.1, plan.total_seconds
        assert plan.count == 3

    def test_offsets_are_monotonic_and_match_gap(self, tmp_dir: Path) -> None:
        picked = self._picked(tmp_dir, 3)
        plan = build_reel(picked, tmp_dir / "reel.wav", gap_seconds=1.0)
        offsets = [s.offset_sec for s in plan.samples]
        assert offsets == sorted(offsets)
        assert offsets[0] == 0.0
        assert offsets[1] == pytest.approx(3.0, abs=0.01), "第 2 段應在 2s 音訊 + 1s 靜音之後"

    def test_reel_is_readable_and_16bit(self, tmp_dir: Path) -> None:
        import soundfile as sf

        picked = self._picked(tmp_dir, 2)
        plan = build_reel(picked, tmp_dir / "reel.wav", gap_seconds=0.5)
        info = sf.info(str(plan.reel_path))
        assert info.channels == 1
        assert info.subtype == "PCM_16", "試聽帶用 16-bit 即可（用途是聽，不是訓練）"
        assert info.samplerate == 48_000

    def test_rms_alignment_lifts_quiet_samples(self, tmp_dir: Path) -> None:
        """不同電平的素材要對齊到接近的 RMS，否則試聽時要一直調音量。"""
        quiet = tmp_dir / "quiet.wav"
        loud = tmp_dir / "loud.wav"
        write_wav(quiet, synth_voice(2.0, amp=0.02), 48_000, subtype="PCM_24")
        write_wav(loud, synth_voice(2.0, amp=0.9), 48_000, subtype="PCM_24")

        plan = build_reel(
            [("g", quiet), ("g", loud)],
            tmp_dir / "reel.wav",
            gap_seconds=0.2,
            align_rms=True,
            target_dbfs=-20.0,
        )
        assert plan.reel_written
        # 兩段的原始 RMS 差距極大，但對齊後應都接近目標
        import soundfile as sf

        data, sr = sf.read(str(plan.reel_path), dtype="float32")
        first = data[: int(2.0 * sr)]
        _, rms_first = measure(first)
        assert -24.0 < rms_first < -16.0, f"對齊後 RMS 應接近 -20 dBFS，得到 {rms_first}"

    def test_alignment_gain_is_limited(self, tmp_dir: Path) -> None:
        """
        極小聲的樣本不該被無上限放大（那只會把底噪變成噪音）。
        硬上限是 +30 dB，所以 amp=0.0005（約 -66 dBFS）不該變成滿格訊號。
        """
        very_quiet = tmp_dir / "veryquiet.wav"
        write_wav(very_quiet, synth_voice(1.0, amp=0.0005), 48_000, subtype="PCM_24")
        plan = build_reel(
            [("g", very_quiet)], tmp_dir / "reel.wav", gap_seconds=0.1, target_dbfs=-20.0
        )
        import soundfile as sf

        data, _ = sf.read(str(plan.reel_path), dtype="float32")
        peak = float(np.max(np.abs(data)))
        assert peak < 0.1, f"增益硬上限 {30} dB 應被遵守，實際 peak={peak}"

    def test_whisper_level_sample_still_becomes_audible(self, tmp_dir: Path) -> None:
        """
        遊戲的耳語／內心戲台詞常刻意錄得很小聲。這類素材情感價值高，
        必須被對齊到聽得見的程度，否則你在試聽帶上會直接漏掉它們。
        """
        whisper = tmp_dir / "whisper.wav"
        write_wav(whisper, synth_voice(1.0, amp=0.004), 48_000, subtype="PCM_24")
        plan = build_reel(
            [("g", whisper)], tmp_dir / "reel.wav", gap_seconds=0.1, target_dbfs=-20.0
        )
        import soundfile as sf

        data, _ = sf.read(str(plan.reel_path), dtype="float32")
        _, rms = measure(data[: int(1.0 * 48_000)])
        assert rms > -30.0, f"耳語級樣本應被提升到聽得見，實際 RMS={rms:.1f} dBFS"

    def test_broken_sample_does_not_abort_reel(self, tmp_dir: Path) -> None:
        """單一樣本損壞時，其餘樣本仍要產生試聽帶（並標記該筆錯誤）。"""
        good = tmp_dir / "good.wav"
        write_wav(good, synth_voice(2.0), 48_000, subtype="PCM_24")
        broken = tmp_dir / "broken.wav"
        broken.write_bytes(b"not a wav at all")

        plan = build_reel(
            [("g", good), ("g", broken), ("g", good)], tmp_dir / "reel.wav", gap_seconds=0.1
        )
        assert plan.reel_written, "不該因為一個壞檔就放棄整條試聽帶"
        errors = [s for s in plan.samples if s.error]
        assert len(errors) == 1
        assert errors[0].path.name == "broken.wav"

    def test_empty_input_returns_empty_plan(self, tmp_dir: Path) -> None:
        plan = build_reel([], tmp_dir / "reel.wav")
        assert plan.count == 0 and not plan.reel_written

    def test_no_align_keeps_original_levels(self, tmp_dir: Path) -> None:
        quiet = tmp_dir / "quiet.wav"
        write_wav(quiet, synth_voice(1.0, amp=0.02), 48_000, subtype="PCM_24")
        plan = build_reel(
            [("g", quiet)], tmp_dir / "reel.wav", align_rms=False, gap_seconds=0.0
        )
        import soundfile as sf

        data, _ = sf.read(str(plan.reel_path), dtype="float32")
        assert float(np.max(np.abs(data))) < 0.05, "關閉對齊後電平應維持原樣"

    def test_resampler_is_injectable(self, tmp_dir: Path) -> None:
        """可注入重採樣器 → 測試不需要真的跑重採樣，也方便之後換演算法。"""
        p = tmp_dir / "src.wav"
        write_wav(p, synth_voice(1.0, sr=44_100), 44_100, subtype="PCM_24")
        calls: list[tuple[int, int]] = []

        def fake_resample(x, src_sr, dst_sr):
            calls.append((src_sr, dst_sr))
            # 真的做一個簡單的長度調整，模擬重採樣
            target_len = int(len(x) * dst_sr / src_sr)
            return np.resize(x, target_len).astype(np.float32)

        plan = build_reel(
            [("g", p)], tmp_dir / "reel.wav", sample_rate=48_000, resampler=fake_resample
        )
        assert calls == [(44_100, 48_000)]
        assert plan.reel_written


class TestWriteList:
    def test_list_contains_every_sample_and_is_markdown(self, tmp_dir: Path) -> None:
        make_tree(tmp_dir / "out", {"hero": 2})
        groups = list_candidates(tmp_dir / "out")
        picked = plan_samples(groups, per_group=2, seed=3)
        plan = build_reel(picked, tmp_dir / "reel.wav", gap_seconds=0.2)

        md = tmp_dir / "reel.md"
        plan.write_list(md)
        text = md.read_text(encoding="utf-8")
        assert "人工試聽清單" in text
        assert text.count("| ") > plan.count
        for s in plan.samples:
            assert s.path.name in text

    def test_list_marks_unreadable_samples(self, tmp_dir: Path) -> None:
        broken = tmp_dir / "bad.wav"
        broken.write_bytes(b"x")
        plan = build_reel([("g", broken)], tmp_dir / "reel.wav")
        md = tmp_dir / "out.md"
        plan.write_list(md)
        assert "⚠" in md.read_text(encoding="utf-8")


class TestHelpers:
    @pytest.mark.parametrize(
        "raw,expected_fragment",
        [
            ("a/b:c.wav", "_"),
            ('bad"name.wav', "_"),
            ("  dots..  ", "dots"),
            ("", "reel"),
        ],
    )
    def test_filesystem_safe(self, raw: str, expected_fragment: str) -> None:
        out = filesystem_safe(raw)
        assert expected_fragment in out
        assert not any(ch in out for ch in '<>:"/\\|?*')


# ---------------------------------------------------------------------------
# 4. 透過 CLI
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


class TestAuditionCli:
    def test_audition_produces_reel_and_list(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir)
        make_tree(root / "data" / "02_wav24_48k", {"heroine_A": 4, "narration": 2})

        proc = _run_cli(["audition", "--per-group", "2"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

        reels = list((root / "data" / "_reports").glob("audition_*.wav"))
        lists = list((root / "data" / "_reports").glob("audition_*.md"))
        assert len(reels) == 1, "應產生一份試聽帶"
        assert len(lists) == 1, "應產生一份對照清單"
        assert reels[0].stat().st_size > 1000
        assert "heroine_A" in proc.stdout and "narration" in proc.stdout

    def test_audition_dry_run_writes_nothing(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir)
        make_tree(root / "data" / "02_wav24_48k", {"g": 3})
        proc = _run_cli(["audition", "--dry-run"], root)
        assert proc.returncode == 0, proc.stdout
        assert not list((root / "data" / "_reports").glob("audition_*"))

    def test_audition_empty_output_dir_gives_actionable_hint(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir)
        proc = _run_cli(["audition"], root)
        assert proc.returncode == 1
        assert "convert" in proc.stdout, "應提示先跑 convert"
        assert "Traceback" not in proc.stderr

    def test_audition_unknown_group_is_rejected(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir)
        make_tree(root / "data" / "02_wav24_48k", {"g": 3})
        proc = _run_cli(["audition", "--groups", "nope"], root)
        assert proc.returncode == 1
        assert "不存在" in proc.stdout
        assert "g" in proc.stdout, "應列出實際存在的群組"

    def test_audition_is_deterministic_via_cli(self, tmp_dir: Path) -> None:
        root = _scaffold(tmp_dir)
        make_tree(root / "data" / "02_wav24_48k", {"g": 8})
        a = _run_cli(["audition", "--per-group", "3", "--seed", "5"], root)
        b = _run_cli(["audition", "--per-group", "3", "--seed", "5"], root)
        assert a.returncode == 0 and b.returncode == 0
        lists = sorted((root / "data" / "_reports").glob("audition_*5.md"))
        assert len(lists) == 1
        first = lists[0].read_text(encoding="utf-8")
        assert first.count("| g |") == 3
