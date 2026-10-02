"""
vOS-VirtualVoice — Phase 1 自動化測試

特色：**不需要任何真實音訊素材**。
測試用 numpy 合成 WAV / Ogg，用 struct 合成 Wwise RIFF 檔頭，
所以在 CI 或新機器上 clone 下來就能跑：

    python -m pytest tests/ -v
    # 或
    python scripts/phase1_prep.py doctor

覆蓋範圍：
  1. 格式嗅探（WAV / Wwise Vorbis / Ogg / FSB / ACB / 未知）——包含「副檔名騙人」情境
  2. 設定載入與驗證（含錯誤設定的明確報錯）
  3. 音訊 I/O：單聲道轉換、重採樣、量測、24-bit 寫檔規格
  4. 端到端：假素材 → convert → 檢查產物規格 + manifest 內容
"""

from __future__ import annotations

import csv
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config as cfgmod  # noqa: E402
import formats as fmt  # noqa: E402
from _audioio import HAS_SOUNDFILE, measure, resample, safe_db, to_mono, write_wav  # noqa: E402
from conftest import synth_voice  # noqa: E402

sf = __import__("soundfile")


# ---------------------------------------------------------------------------
# 合成素材工具
# ---------------------------------------------------------------------------
@pytest.fixture
def sandbox(tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    建立一個獨立沙箱：把 repo 根目錄換成暫存資料夾，
    這樣設定載入相關的測試永遠不會污染真正的 data/。
    """
    root = tmp_dir
    (root / "configs").mkdir()
    (root / "data" / "00_raw").mkdir(parents=True)
    for sub in ("01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)

    cfg_src = REPO / "configs" / "pipeline.default.json"
    (root / "configs" / "pipeline.local.json").write_text(
        cfg_src.read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.setattr(cfgmod, "repo_root", lambda: root)
    return root


# ---------------------------------------------------------------------------
# 1. 格式嗅探
# ---------------------------------------------------------------------------
class TestSniffFormat:
    def test_plain_pcm_wav_is_detected_as_wav(self, tmp_dir: Path) -> None:
        p = tmp_dir / "a.wav"
        write_wav(p, synth_voice(0.5), 48_000, subtype="PCM_24")
        info = fmt.sniff_format(p)
        assert info.code == "wav_riff"
        assert info.needs_unpack is False
        assert info.decoder == "soundfile"

    def test_wwise_vorbis_detected_from_synthetic_riff(self, tmp_dir: Path) -> None:
        """關鍵測試：同樣是 RIFF，但有 vorb chunk → 必須判為 Wwise，需要解包。"""
        pcm_fmt = struct.pack("<HHIIHH", 0x0001, 1, 48_000, 48_000 * 2, 2, 16)
        blob = fmt.build_test_riff(chunks=[(b"fmt ", pcm_fmt), (b"vorb", b"OggS" + b"\x00" * 40)])
        p = tmp_dir / "b.wem"
        p.write_bytes(blob)
        info = fmt.sniff_format(p)
        assert info.code == "wem_vorbis"
        assert info.needs_unpack is True
        assert info.decoder == "vgmstream"

    def test_wwise_nonstandard_codec_is_not_treated_as_pcm(self, tmp_dir: Path) -> None:
        """wFormatTag = 0xFFFF（Wwise 專有）不可誤判成 PCM。"""
        weird = struct.pack("<HHIIHH", 0xFFFF, 1, 48_000, 48_000 * 2, 2, 16)
        blob = fmt.build_test_riff(chunks=[(b"fmt ", weird), (b"data", b"\x00" * 32)])
        p = tmp_dir / "c.wem"
        p.write_bytes(blob)
        info = fmt.sniff_format(p)
        assert info.code == "wem_other"
        assert info.needs_unpack is True

    def test_ogg_renamed_to_wem_is_marked_as_mismatch(self, tmp_dir: Path) -> None:
        """最常見的遊戲素材陷阱：.wem 其實是純 Ogg，不需解包。"""
        p = tmp_dir / "d.wem"
        sf.write(str(p), synth_voice(0.5), 48_000, format="OGG", subtype="VORBIS")
        info = fmt.sniff_format(p)
        assert info.code == "ogg_vorbis"
        assert info.needs_unpack is False, "純 Ogg 不該被要求解包"
        assert fmt.sniff_format(p).as_row()["needs_unpack"] == "no"

    def test_fsb_and_acb_and_awb_magic(self, tmp_dir: Path) -> None:
        cases = {
            b"FSB5" + b"\x00" * 60: "fsb",
            b"@UTF" + b"\x00" * 60: "cri_acb",
            b"AFS2" + b"\x00" * 60: "cri_awb",
            b"AKPK" + b"\x00" * 60: "wwise_akpk",
            b"fLaC" + b"\x00" * 60: "flac",
        }
        for blob, expected in cases.items():
            p = tmp_dir / f"{expected}.bin"
            p.write_bytes(blob)
            assert fmt.sniff_format(p).code == expected, expected

    def test_unknown_garbage_never_raises(self, tmp_dir: Path) -> None:
        p = tmp_dir / "junk.wem"
        p.write_bytes(b"\xde\xad\xbe\xef" * 32)
        info = fmt.sniff_format(p)
        assert info.code == "unknown"
        assert info.decoder == "none"

    def test_tiny_file_returns_unknown(self, tmp_dir: Path) -> None:
        p = tmp_dir / "tiny.wav"
        p.write_bytes(b"RI")
        assert fmt.sniff_format(p).code == "unknown"

    def test_missing_file_returns_unknown_not_exception(self, tmp_dir: Path) -> None:
        assert fmt.sniff_format(tmp_dir / "nope.wem").code == "unknown"


# ---------------------------------------------------------------------------
# 2. 設定載入
# ---------------------------------------------------------------------------
class TestConfig:
    def test_default_config_loads_and_paths_are_relative_to_root(self, sandbox: Path) -> None:
        cfg = cfgmod.load_config(sandbox / "configs" / "pipeline.local.json")
        assert cfg.target.sample_rate == 48_000
        assert cfg.target.bit_depth == 24
        assert cfg.target.subtype == "PCM_24"
        assert cfg.duration_filter.min_seconds == 1.5
        assert cfg.duration_filter.max_seconds == 15.0
        assert cfg.paths.raw_dir.is_absolute()
        assert cfg.paths.raw_dir.name == "00_raw"

    def test_invalid_duration_range_raises_clear_error(self, sandbox: Path) -> None:
        bad = json.loads((sandbox / "configs" / "pipeline.local.json").read_text(encoding="utf-8"))
        bad["duration_filter"] = {"min_seconds": 20, "max_seconds": 10}
        p = sandbox / "configs" / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(cfgmod.ConfigError, match="min_seconds"):
            cfgmod.load_config(p)

    def test_invalid_bit_depth_raises(self, sandbox: Path) -> None:
        bad = json.loads((sandbox / "configs" / "pipeline.local.json").read_text(encoding="utf-8"))
        bad["target"]["bit_depth"] = 20
        p = sandbox / "configs" / "bad2.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(cfgmod.ConfigError, match="bit_depth"):
            cfgmod.load_config(p)

    def test_missing_config_file_raises(self, tmp_dir: Path) -> None:
        with pytest.raises(cfgmod.ConfigError, match="找不到設定檔"):
            cfgmod.load_config(tmp_dir / "nope.json")

    def test_output_path_uses_wav_suffix(self, sandbox: Path) -> None:
        cfg = cfgmod.load_config(sandbox / "configs" / "pipeline.local.json")
        out = cfg.resolve_output_for(Path("data/00_raw/hero/a.wem"), cfg.paths.standardized_dir)
        assert out.suffix == ".wav"
        assert out.name == "a.wav"

    def test_grouping_defaults_to_none_and_flattens_output(self, sandbox: Path) -> None:
        cfg = cfgmod.load_config(sandbox / "configs" / "pipeline.local.json")
        assert cfg.group_by == "none"
        src = sandbox / "data" / "00_raw" / "hero_a" / "a.wem"
        assert cfg.group_of(src) == "default"
        out = cfg.resolve_output_for(src, cfg.paths.standardized_dir)
        assert out.parent == cfg.paths.standardized_dir / "hero_a"
        assert "default" not in out.parts

    def test_grouping_folder_splits_by_top_level_dir(self, sandbox: Path) -> None:
        import json as _json

        data = _json.loads((sandbox / "configs" / "pipeline.local.json").read_text(encoding="utf-8"))
        data["grouping"] = {"by": "folder"}
        p = sandbox / "configs" / "grouped.json"
        p.write_text(_json.dumps(data), encoding="utf-8")
        cfg = cfgmod.load_config(p)

        assert cfg.group_by == "folder"
        src_a = sandbox / "data" / "00_raw" / "heroine_A" / "sub" / "a.wem"
        src_b = sandbox / "data" / "00_raw" / "heroine_B" / "b.wem"
        src_flat = sandbox / "data" / "00_raw" / "loose.wem"
        assert cfg.group_of(src_a) == "heroine_A"
        assert cfg.group_of(src_b) == "heroine_B"
        assert cfg.group_of(src_flat) == "default", "頂層散檔沒有群組可歸，進 default"

        out_a = cfg.resolve_output_for(src_a, cfg.paths.standardized_dir)
        assert out_a.parts[-3:] == ("heroine_A", "sub", "a.wav")

    def test_invalid_grouping_value_raises(self, sandbox: Path) -> None:
        import json as _json

        data = _json.loads((sandbox / "configs" / "pipeline.local.json").read_text(encoding="utf-8"))
        data["grouping"] = {"by": "character"}
        p = sandbox / "configs" / "badgroup.json"
        p.write_text(_json.dumps(data), encoding="utf-8")
        with pytest.raises(cfgmod.ConfigError, match="grouping.by"):
            cfgmod.load_config(p)

    def test_repo_root_honours_env_override(self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        端到端測試的隔離基礎：VOS_REPO_ROOT 必須能覆寫 repo 根目錄，
        否則測試素材會寫進開發者真正的 data/。
        """
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(tmp_dir))
        assert cfgmod.repo_root() == tmp_dir.resolve()
        monkeypatch.delenv(cfgmod.ENV_REPO_ROOT)
        assert cfgmod.repo_root() == REPO


# ---------------------------------------------------------------------------
# 3. 音訊 I/O
# ---------------------------------------------------------------------------
class TestAudioIO:
    def test_to_mono_averages_stereo(self) -> None:
        left = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        right = np.array([0.0, 1.0, 0.5], dtype=np.float32)
        stereo = np.stack([left, right], axis=1)
        mono = to_mono(stereo)
        assert mono.shape == (3,)
        np.testing.assert_allclose(mono, [0.5, 0.0, 0.5], atol=1e-6)

    def test_to_mono_is_noop_on_mono(self) -> None:
        x = np.array([0.1, 0.2], dtype=np.float32)
        assert to_mono(x) is x

    def test_resample_44100_to_48000_preserves_duration(self) -> None:
        sig = synth_voice(1.0, sr=44_100)
        out = resample(sig, 44_100, 48_000)
        assert abs(len(out) / 48_000 - 1.0) < 0.01, "重採樣後時長必須保持一致"

    def test_resample_same_rate_is_identity(self) -> None:
        sig = synth_voice(0.2)
        assert resample(sig, 48_000, 48_000) is sig

    def test_resample_downsample_48k_to_16k(self) -> None:
        sig = synth_voice(1.0, sr=48_000)
        out = resample(sig, 48_000, 16_000)
        assert abs(len(out) / 16_000 - 1.0) < 0.02

    def test_measure_reports_dbfs(self) -> None:
        peak, rms = measure(synth_voice(0.5, amp=0.5))
        assert -7.0 < peak < -5.0, f"amp=0.5 的 peak 應約 -6 dBFS，得到 {peak}"
        assert rms < peak, "RMS 必然低於 Peak"

    def test_measure_silence_is_negative_infinity(self) -> None:
        peak, rms = measure(np.zeros(1000, dtype=np.float32))
        assert peak == float("-inf") and rms == float("-inf")

    def test_measure_empty_array_does_not_crash(self) -> None:
        assert measure(np.array([], dtype=np.float32)) == (float("-inf"), float("-inf"))

    def test_safe_db_formats_infinity(self) -> None:
        assert safe_db(float("-inf")) == "-inf"
        assert safe_db(-6.0) == "-6.00"

    def test_written_wav_is_24bit_48k_mono(self, tmp_dir: Path) -> None:
        p = tmp_dir / "out.wav"
        write_wav(p, synth_voice(0.3), 48_000, subtype="PCM_24")
        info = sf.info(str(p))
        assert info.samplerate == 48_000
        assert info.subtype == "PCM_24"
        assert info.channels == 1

    def test_written_wav_creates_parent_dirs(self, tmp_dir: Path) -> None:
        p = tmp_dir / "deep" / "nest" / "out.wav"
        write_wav(p, synth_voice(0.1), 48_000)
        assert p.is_file()


# ---------------------------------------------------------------------------
# 4. 端到端（不打網路、不需外部工具、完全不污染真正的 data/）
# ---------------------------------------------------------------------------
def _run_cli(
    args: list[str], repo_root: Path, *, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """
    以子程序執行真正的 CLI，並用 VOS_REPO_ROOT 把它的 repo 根目錄
    指向指定的沙箱資料夾 —— 這是關鍵：沒有這個覆寫，端到端測試就會把
    測試素材寫進開發者真正的 data/ 裡（曾經真的發生過）。
    """
    import os as _os

    env = dict(_os.environ)
    env[cfgmod.ENV_REPO_ROOT] = str(repo_root)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "phase1_prep.py"), *args],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


def _scaffold_repo(root: Path, *, config_patch: dict | None = None) -> Path:
    """
    在 root 建立一個最小可用的 repo 骨架：configs/ 與 data/ 五個資料夾。
    可選擇性覆寫設定檔欄位（例如把 grouping.by 改成 folder）。
    """
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)

    data = json.loads((REPO / "configs" / "pipeline.default.json").read_text(encoding="utf-8"))
    for key, value in (config_patch or {}).items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    (root / "configs" / "pipeline.local.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


#: 端到端用的素材配置（刻意涵蓋每一條規則）
E2E_SPEC: dict[str, tuple[float, int, float]] = {
    "ok_short.wav": (2.0, 44_100, 0.5),      # 合格，需升採樣 + 立體聲→單聲道
    "ok_long.wav": (10.0, 48_000, 0.5),      # 合格
    "too_short.wav": (0.8, 48_000, 0.5),     # 淘汰：< 1.5s
    "too_long.wav": (16.0, 48_000, 0.5),     # 淘汰：> 15s
    "silent.wav": (3.0, 48_000, 0.0),        # 淘汰：全靜音
}


def _make_spec_files(raw_dir: Path, spec: dict[str, tuple[float, int, float]]) -> None:
    for name, (dur, sr, amp) in spec.items():
        path = raw_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sig = synth_voice(dur, sr=sr, amp=amp)
        if name == "ok_short.wav":  # 順便驗證立體聲 → 單聲道
            sig = np.stack([sig, sig * 0.8], axis=1)
        write_wav(path, sig, sr, subtype="PCM_16")


class TestEndToEnd:
    """
    端到端測試：每個測試都在自己的沙箱 repo 內跑真正的 CLI。

    VOS_REPO_ROOT 讓 CLI 把沙箱當成 repo 根目錄，因此
    **真正的 data/ 永遠不會被碰到**，也不需要 finally 清理。
    """

    def test_convert_produces_24bit_48k_and_correct_manifest(self, tmp_dir: Path) -> None:
        root = _scaffold_repo(tmp_dir)
        _make_spec_files(root / "data" / "00_raw", E2E_SPEC)

        proc = _run_cli(["convert", "--overwrite", "--limit", "10"], root)
        assert proc.returncode == 0, f"CLI 失敗：\n{proc.stdout}\n{proc.stderr}"

        # -- 產物規格：24bit / 48kHz / 單聲道 --
        out = root / "data" / "02_wav24_48k"
        for name in ("ok_short.wav", "ok_long.wav"):
            f = out / name
            assert f.is_file(), f"{name} 應該要產出"
            info = sf.info(str(f))
            assert (info.samplerate, info.subtype, info.channels) == (48_000, "PCM_24", 1)

        # -- 淘汰檔應被複製（不是搬走）到 99_rejected --
        raw = root / "data" / "00_raw"
        rejected = root / "data" / "99_rejected"
        for name in ("too_short.wav", "too_long.wav", "silent.wav"):
            assert (rejected / name).is_file(), f"{name} 應被複製到 99_rejected"
            assert (raw / name).is_file(), f"{name} 原檔不可被搬走或刪除"

        # -- manifest 內容 --
        man = root / "data" / "_reports" / "manifest_phase1.csv"
        assert man.is_file()
        with man.open(encoding="utf-8-sig", newline="") as fh:
            rows = {Path(r["src_path"]).name: r for r in csv.DictReader(fh)}

        assert rows["ok_short.wav"]["status"] == "ok"
        assert rows["ok_short.wav"]["src_sr"] == "44100"
        assert rows["ok_short.wav"]["src_channels"] == "2"
        assert rows["ok_short.wav"]["out_sr"] == "48000"
        assert rows["ok_short.wav"]["out_bit_depth"] == "24"
        assert rows["ok_short.wav"]["group"] == "default"
        for name, keyword in (("too_short.wav", "過短"), ("too_long.wav", "過長"), ("silent.wav", "靜音")):
            assert rows[name]["status"] == "rejected", name
            assert keyword in rows[name]["reason"], (name, rows[name]["reason"])

        # -- summary JSON 可被 Phase 2 直接讀 --
        summary = json.loads((root / "data" / "_reports" / "phase1_summary.json").read_text("utf-8"))
        assert summary["counts"]["ok"] == 2
        assert summary["counts"]["rejected"] == 3
        assert summary["target"]["sample_rate"] == 48_000
        assert 11.9 < summary["total_ok_seconds"] < 12.1
        assert "default" in summary["groups"]

    def test_multi_character_grouping_separates_output_and_manifest(self, tmp_dir: Path) -> None:
        """
        多角色專案：data/00_raw/<角色名>/ 的素材必須各自進到
        02_wav24_48k/<角色名>/，且 manifest 的 group 欄位要正確。
        """
        root = _scaffold_repo(tmp_dir, config_patch={"grouping": {"by": "folder"}})
        raw = root / "data" / "00_raw"
        _make_spec_files(raw / "heroine_A", {"a_01.wav": (3.0, 48_000, 0.5)})
        _make_spec_files(raw / "heroine_B", {"b_01.wav": (4.0, 48_000, 0.5), "b_02.wav": (0.5, 48_000, 0.5)})

        proc = _run_cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, proc.stdout

        out = root / "data" / "02_wav24_48k"
        assert (out / "heroine_A" / "a_01.wav").is_file()
        assert (out / "heroine_B" / "b_01.wav").is_file()
        assert not (out / "b_02.wav").exists(), "過短的檔案不該出現在合格區"

        # 逐角色統計要能反映「哪些角色素材不足」
        proc = _run_cli(["convert", "--overwrite"], root)
        assert "逐角色素材量" in proc.stdout, proc.stdout
        assert "heroine_A" in proc.stdout and "heroine_B" in proc.stdout

        with (root / "data" / "_reports" / "manifest_phase1.csv").open(
            encoding="utf-8-sig", newline=""
        ) as fh:
            rows = {Path(r["src_path"]).name: r for r in csv.DictReader(fh)}
        assert rows["a_01.wav"]["group"] == "heroine_A"
        assert rows["b_01.wav"]["group"] == "heroine_B"
        assert rows["b_02.wav"]["status"] == "rejected"
        assert rows["b_02.wav"]["group"] == "heroine_B"

        summary = json.loads((root / "data" / "_reports" / "phase1_summary.json").read_text("utf-8"))
        assert summary["group_by"] == "folder"
        assert summary["groups"]["heroine_A"]["ok"] == 1
        assert summary["groups"]["heroine_B"]["ok"] == 1
        assert summary["groups"]["heroine_B"]["rejected"] == 1

    def test_same_named_files_from_different_characters_do_not_collide(self, tmp_dir: Path) -> None:
        """
        真實地雷：不同角色常有同名檔案（A/voice_001.wav 與 B/voice_001.wav）。
        淘汰區與合格區都必須分開存放，否則後處理的檔案會覆蓋前一個，
        你會以為淘汰區只有一個檔。
        """
        root = _scaffold_repo(tmp_dir, config_patch={"grouping": {"by": "folder"}})
        raw = root / "data" / "00_raw"
        # 同名檔案，但一位合格、一位過短 → 必須一個進合格區、一個進淘汰區
        _make_spec_files(raw / "heroine_A", {"voice_001.wav": (3.0, 48_000, 0.5)})
        _make_spec_files(raw / "heroine_B", {"voice_001.wav": (0.6, 48_000, 0.5)})

        proc = _run_cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, proc.stdout

        assert (root / "data" / "02_wav24_48k" / "heroine_A" / "voice_001.wav").is_file()
        assert not (root / "data" / "02_wav24_48k" / "heroine_B" / "voice_001.wav").exists()
        assert (root / "data" / "99_rejected" / "heroine_B" / "voice_001.wav").is_file(), (
            "淘汰檔必須帶上角色子資料夾，否則會被其他角色的同名檔覆蓋"
        )
        assert not (root / "data" / "99_rejected" / "voice_001.wav").exists()

        # 原檔全部保留
        assert (raw / "heroine_A" / "voice_001.wav").is_file()
        assert (raw / "heroine_B" / "voice_001.wav").is_file()

    def test_single_character_mode_keeps_output_flat(self, tmp_dir: Path) -> None:
        """grouping.by=none 時，產物不該多出一層 default/ 資料夾。"""
        root = _scaffold_repo(tmp_dir)
        _make_spec_files(root / "data" / "00_raw", {"single.wav": (3.0, 48_000, 0.5)})
        proc = _run_cli(["convert", "--overwrite"], root)
        assert proc.returncode == 0, proc.stdout
        out = root / "data" / "02_wav24_48k"
        assert (out / "single.wav").is_file()
        assert not (out / "default").exists()

    def test_dry_run_writes_nothing(self, tmp_dir: Path) -> None:
        root = _scaffold_repo(tmp_dir)
        _make_spec_files(root / "data" / "00_raw", {"dryrun_probe.wav": (3.0, 48_000, 0.5)})

        proc = _run_cli(["convert", "--dry-run"], root)
        assert proc.returncode == 0, proc.stdout
        assert not (root / "data" / "02_wav24_48k" / "dryrun_probe.wav").exists()
        # dry-run 也不該寫出報表
        assert not (root / "data" / "_reports" / "manifest_phase1.csv").exists()

    def test_unpack_without_vgmstream_is_actionable(self, tmp_dir: Path) -> None:
        """沒有 vgmstream-cli 時，unpack 要給出可操作指引而不是 traceback。"""
        root = _scaffold_repo(tmp_dir)
        (root / "data" / "00_raw" / "voice.wem").write_bytes(
            fmt.build_test_riff(
                chunks=[
                    (b"fmt ", struct.pack("<HHIIHH", 0xFFFF, 1, 48_000, 96_000, 2, 16)),
                    (b"data", b"\x00" * 32),
                ]
            )
        )
        proc = _run_cli(["unpack"], root)
        out = proc.stdout
        if "找不到 vgmstream-cli" in out:
            assert "PHASE1-INGEST" in out or "convert" in out
        else:  # 真的有裝 vgmstream：合成檔會解包失敗，但不該是 traceback
            assert "Traceback" not in proc.stderr

    def test_doctor_exits_zero_when_deps_present(self, tmp_dir: Path) -> None:
        root = _scaffold_repo(tmp_dir)
        proc = _run_cli(["doctor"], root)
        assert proc.returncode == 0, proc.stdout
        assert "soundfile" in proc.stdout

    def test_scan_on_empty_raw_dir_gives_actionable_hint(self, tmp_dir: Path) -> None:
        """空的 00_raw 應該給出可操作的提示，而不是丟例外。"""
        root = _scaffold_repo(tmp_dir)
        proc = _run_cli(["scan"], root)
        assert proc.returncode == 1
        assert "00_raw" in proc.stdout
        assert "Traceback" not in proc.stderr

    def test_scan_detects_renamed_ogg_and_container(self, tmp_dir: Path) -> None:
        """scan 必須依內容而非副檔名判斷：.wem 其實是 Ogg → 不需解包。"""
        root = _scaffold_repo(tmp_dir)
        raw = root / "data" / "00_raw"
        raw.mkdir(parents=True, exist_ok=True)
        sf.write(str(raw / "actually_ogg.wem"), synth_voice(2.0), 48_000, format="OGG", subtype="VORBIS")
        (raw / "actually_wwise.wem").write_bytes(
            fmt.build_test_riff(
                chunks=[
                    (b"fmt ", struct.pack("<HHIIHH", 0x0001, 1, 48_000, 96_000, 2, 16)),
                    (b"vorb", b"OggS" + b"\x00" * 40),
                ]
            )
        )
        proc = _run_cli(["scan"], root)
        assert "Ogg Vorbis" in proc.stdout
        assert "Wwise Vorbis" in proc.stdout
        assert "可直接解碼：1 個   需先解包：1 個" in proc.stdout


# ---------------------------------------------------------------------------
# 5. 跨平台路徑衛生
# ---------------------------------------------------------------------------
class TestRepoHygiene:
    def test_no_hardcoded_absolute_user_paths_in_scripts(self) -> None:
        """
        紅線檢查：程式碼與文件中不得出現真實使用者名稱或本機絕對路徑。
        這條測試會在 CI 上直接擋下誤 commit 的個資。
        """
        import re

        patterns = [
            # C:\Users\<name> 或 C:/Users/<name>（但排除文件中的 C:\Users\<你的帳號> 樣板）
            re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}(?!<)[A-Za-z0-9._-]+"),
            re.compile(r"/Users/(?!<)[A-Za-z0-9._-]+/"),   # /Users/<name>/
            re.compile(r"/home/(?!<)[A-Za-z0-9._-]+/"),    # /home/<name>/
        ]
        offenders: list[str] = []
        for folder in ("scripts", "tests", "configs", "docs"):
            for f in (REPO / folder).rglob("*"):
                if f.is_file() and f.suffix in (".py", ".json", ".md", ".sh", ".ps1"):
                    text = f.read_text(encoding="utf-8", errors="replace")
                    for pat in patterns:
                        for m in pat.finditer(text):
                            offenders.append(f"{f.relative_to(REPO)}: {m.group(0)}")
        assert not offenders, "發現疑似本機絕對路徑／使用者名稱：\n" + "\n".join(offenders)
