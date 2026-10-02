"""
vOS-VirtualVoice — 容器解包測試 (Phase 1, 任務 1)

這些測試針對「解包」這個 Phase 1 最貴、最容易靜默失敗的環節。
使用 tests/fake_vgmstream.py 當替身，因此**不需要下載 vgmstream-cli**、
也不需要真實遊戲素材，在任何機器上都能跑。

守著的三個真實坑：
  1. `-o` 少了 subsong 萬用字元 → 多 subsong 互相覆蓋，你只拿到最後一條
  2. 不在來源目錄就地解包 → 找不到 .hcakey/.acb 等 companion file
  3. subsong 數量大爆炸 → 一個 .awb 解出數千個檔案，佔滿磁碟
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config as cfgmod  # noqa: E402
import unpack as unpackmod  # noqa: E402
from conftest import ENV_REPO_ROOT, make_fake_exe, run_cli  # noqa: E402
from unpack import (  # noqa: E402
    UnpackOptions,
    build_probe_cmd,
    build_unpack_cmd,
    companion_hint,
    parse_subsong_count,
    unpack_one,
)


# ---------------------------------------------------------------------------
# 替身可執行檔（實作在 conftest.py，所有測試檔共用）
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_exe(tmp_dir: Path) -> Path:
    return make_fake_exe(tmp_dir / "bin")


def make_container(path: Path, subsongs: int) -> Path:
    """
    建立一個假容器檔。副檔名用 .awb（會被 sniff 判為 CriWare 需解包），
    並在檔名嵌入要模擬的 subsong 數量。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"AFS2" + b"\x00" * 64)
    return path


def read_marker(path: Path) -> str:
    """
    回傳檔案的「波形指紋」（給測試判斷 subsong → 檔案的對應）。

    早期版本的替身把 ASCII marker 寫進音訊資料，但那樣產生的不是真實波形，
    就無法測試後續的時長篩選。現在改用波形特徵：不同 subsong 的基頻不同，
    所以指紋不同；同時能確認音訊**不是靜音**。
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    if data.size == 0:
        return "empty"
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    if rms <= 0:
        return "silent"
    # 零交越率可反映基頻高低 → 足以區分不同 subsong
    crossings = int(np.sum(np.diff(np.signbit(data))))
    return f"rms={rms:.4f},zcr={crossings},dur={data.size / sr:.3f}"


# ---------------------------------------------------------------------------
# 1. 純函式：命令列建構與解析
# ---------------------------------------------------------------------------
class TestBuildCommands:
    def test_probe_cmd_uses_metadata_flag(self) -> None:
        cmd = build_probe_cmd("/x/vgmstream-cli", Path("a.awb"))
        assert cmd[1] == "-m", "-m 是『只印資訊不解碼』，探測 subsong 數量必需"
        assert cmd[-1] == "a.awb"

    def test_unpack_cmd_disables_looping(self) -> None:
        """遊戲語音常有 loop 標記；不解 -i 會把 3 秒的語音變成無限長的檔案。"""
        cmd = build_unpack_cmd(
            "/x/vgmstream-cli", Path("a.wem"), Path("out#?s.wav"), all_subsongs=False
        )
        assert "-i" in cmd

    def test_unpack_cmd_includes_subsong_wildcard_contract(self) -> None:
        """
        契約測試：輸出樣式一定要含 `?s`。
        這是本專案踩過的坑 —— 少了它，多 subsong 會全部覆蓋同一個檔案。
        """
        cmd = build_unpack_cmd(
            "/x/vgmstream-cli", Path("a.awb"), Path("bank#?s.wav"), all_subsongs=True
        )
        out = cmd[cmd.index("-o") + 1]
        assert "?s" in out, "輸出樣式必須含 subsong 萬用字元，否則 subsong 會互相覆蓋"
        assert "-S" in cmd and "0" in cmd

    def test_unpack_cmd_single_subsong_omits_dash_S(self) -> None:
        cmd = build_unpack_cmd(
            "/x/vgmstream-cli", Path("a.wem"), Path("a#?s.wav"), all_subsongs=False
        )
        assert "-S" not in cmd, "已知只有 1 個 subsong 時不該展開"

    def test_extra_args_are_passed_through(self) -> None:
        cmd = build_unpack_cmd(
            "/x/vgmstream-cli",
            Path("a.wem"),
            Path("a#?s.wav"),
            all_subsongs=False,
            extra_args=("--loop-count", "1"),
        )
        assert "--loop-count" in cmd and "1" in cmd


class TestParseSubsongCount:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("subsongs: 1250", 1250),
            ("Subsongs: 3", 3),
            ("1250 subsongs found", 1250),
            ("streams: 42", 42),
            ("encoding: Vorbis\nchannels: 2\nsubsongs: 7\n", 7),
        ],
    )
    def test_parses_real_world_shapes(self, text: str, expected: int) -> None:
        assert parse_subsong_count(text) == expected

    def test_unknown_returns_minus_one_not_one(self) -> None:
        """
        關鍵：不確定必須回 -1。
        若回 1，安全機制會誤以為「只有一條語音」而跳過展開，導致只解出第一條。
        """
        assert parse_subsong_count("encoding: Vorbis\nchannels: 2\n") == -1
        assert parse_subsong_count("") == -1


class TestCompanionHint:
    def test_hca_warns_about_key_file(self, tmp_dir: Path) -> None:
        src = tmp_dir / "voice.hca"
        src.write_bytes(b"HCA\x00")
        hint = companion_hint(src, {"voice.hca"})
        assert ".hcakey" in hint

    def test_acb_without_matching_awb_is_flagged(self, tmp_dir: Path) -> None:
        """.acb 只是索引表；沒有配對的 .awb 就什麼都解不出來。"""
        src = tmp_dir / "bgm.acb"
        src.write_bytes(b"@UTF")
        hint = companion_hint(src, {"bgm.acb"})
        assert "bgm.awb" in hint and "找不到" in hint

    def test_acb_with_matching_awb_is_available(self, tmp_dir: Path) -> None:
        src = tmp_dir / "bgm.acb"
        src.write_bytes(b"@UTF")
        hint = companion_hint(src, {"bgm.acb", "bgm.awb"})
        assert "找不到" not in hint

    def test_plain_wav_has_no_hint(self, tmp_dir: Path) -> None:
        src = tmp_dir / "a.wav"
        src.write_bytes(b"RIFF")
        assert companion_hint(src, {"a.wav"}) == ""


# ---------------------------------------------------------------------------
# 2. 使用替身的實際解包
# ---------------------------------------------------------------------------
class TestUnpackWithFakeTool:
    def test_multi_subsong_produces_separate_files(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        核心回歸測試：12 個 subsong 必須產生 12 個獨立檔案，內容各自對應。
        這正是「`-o` 少了 ?s」會壞掉的地方。
        """
        src = make_container(tmp_dir / "00_raw" / "bank_fake_subsongs=12_fake_dur=2.awb", 12)
        dest = tmp_dir / "01_unpacked" / "bank"

        res = unpack_one(src, dest, str(fake_exe))

        assert res.ok, res.error
        assert len(res.out_paths) == 12, f"應產生 12 個檔案，實際 {len(res.out_paths)}"
        assert res.subsongs == 12
        fingerprints = {read_marker(p) for p in res.out_paths}
        assert len(fingerprints) == 12, "每個檔案內容必須不同（否則就是互相覆蓋了）"
        assert not any(f == "silent" for f in fingerprints), "解包產物不該是靜音"

    def test_single_subsong_produces_one_file(self, tmp_dir: Path, fake_exe: Path) -> None:
        src = make_container(tmp_dir / "00_raw" / "voice_01_fake_dur=3.wem", 1)
        res = unpack_one(src, tmp_dir / "out", str(fake_exe))
        assert res.ok, res.error
        assert len(res.out_paths) == 1
        assert read_marker(res.out_paths[0]) != "silent"

    def test_single_subsong_output_name_has_no_wildcard(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        回歸測試（跨平台地雷）：單一 subsong 的輸出檔名**不可含 `?s`**。

        `?s` 只在「展開多個 subsong」時才有機會被 vgmstream 替換掉；
        若容器只有 1 個 subsong 而我們仍傳 `#?s.wav`，那個 `?` 會原封不動
        留在檔名裡 —— 而 `?` 在 Windows 是非法檔名字元，導致
        `OSError: [Errno 22] Invalid argument`，**所有單 subsong 容器全部解包失敗**。

        在 Linux/macOS 上這個 bug 不會出現（`?` 是合法檔名字元），
        所以它是一個只咬 Windows 使用者的隱形殺手。
        """
        src = make_container(tmp_dir / "00_raw" / "voice.wem", 1)
        res = unpack_one(src, tmp_dir / "out", str(fake_exe))
        assert res.ok, res.error
        assert len(res.out_paths) == 1
        name = res.out_paths[0].name
        assert "?" not in name, f"檔名不該含 '?'：{name}"
        assert "#" not in name, f"單一 subsong 不需要編號：{name}"
        assert name == "voice.wav"

    def test_multi_subsong_output_names_carry_index(self, tmp_dir: Path, fake_exe: Path) -> None:
        """反過來：多 subsong 時檔名**必須**帶編號，否則就是互相覆蓋。"""
        src = make_container(tmp_dir / "00_raw" / "bank_fake_subsongs=3_fake_dur=2.awb", 3)
        res = unpack_one(src, tmp_dir / "out", str(fake_exe))
        assert res.ok, res.error
        names = sorted(p.name for p in res.out_paths)
        assert names == ["bank_fake_subsongs=3_fake_dur=2#1.wav",
                         "bank_fake_subsongs=3_fake_dur=2#2.wav",
                         "bank_fake_subsongs=3_fake_dur=2#3.wav"], names
        assert "?" not in "".join(names)

    def test_unpacks_in_source_directory_then_moves(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        就地解包契約：vgmstream 必須在來源檔所在目錄執行，
        否則找不到 .hcakey/.acb/.adxkey 等 companion file。

        驗證方式：解包後來源目錄必須乾淨（暫存子資料夾已刪除），
        且產物出現在目的目錄。
        """
        raw = tmp_dir / "00_raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / ".awbkey").write_text("dummy", encoding="utf-8")  # 模擬 companion file
        src = make_container(raw / "bank_fake_subsongs=3.awb", 3)
        dest = tmp_dir / "01_unpacked"

        res = unpack_one(src, dest, str(fake_exe))
        assert res.ok, res.error
        assert len(res.out_paths) == 3
        assert all(p.parent == dest for p in res.out_paths), "產物必須落在目的資料夾"
        assert all(p.is_file() and p.stat().st_size > 44 for p in res.out_paths)

        leftovers = [p.name for p in raw.iterdir() if p.name.startswith(".vos_unpack_")]
        assert not leftovers, f"暫存資料夾未清乾淨：{leftovers}"
        assert (raw / ".awbkey").is_file(), "companion file 不可被動到"
        assert src.is_file(), "來源容器不可被搬走或刪除"

    def test_subsong_limit_skips_instead_of_exploding(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        安全上限：內含 5000 個 subsong 的容器預設必須『跳過』，
        而不是老實解出 5000 個檔案塞爆磁碟。
        """
        src = make_container(tmp_dir / "00_raw" / "huge_fake_subsongs=5000.awb", 5000)
        dest = tmp_dir / "01_unpacked"
        opts = UnpackOptions(max_subsongs=100)

        res = unpack_one(src, dest, str(fake_exe), opts)

        assert res.skipped, "超過上限必須標記為 skipped"
        assert res.out_paths == []
        assert "5000" in res.error and "100" in res.error
        assert not dest.exists() or not list(dest.glob("*.wav"))

    def test_limit_can_be_raised(self, tmp_dir: Path, fake_exe: Path) -> None:
        src = make_container(tmp_dir / "00_raw" / "big_fake_subsongs=40.awb", 40)
        opts = UnpackOptions(max_subsongs=100)
        res = unpack_one(src, tmp_dir / "out", str(fake_exe), opts)
        assert res.ok and len(res.out_paths) == 40

    def test_missing_tool_reports_clearly(self, tmp_dir: Path) -> None:
        src = make_container(tmp_dir / "00_raw" / "a.wem", 1)
        res = unpack_one(src, tmp_dir / "out", str(tmp_dir / "nope" / "vgmstream-cli"))
        assert not res.ok
        assert "找不到" in res.error

    def test_missing_source_reports_clearly(self, tmp_dir: Path, fake_exe: Path) -> None:
        res = unpack_one(tmp_dir / "ghost.awb", tmp_dir / "out", str(fake_exe))
        assert not res.ok
        assert "不存在" in res.error

    def test_no_output_reports_vgmstream_message(self, tmp_dir: Path, fake_exe: Path) -> None:
        """vgmstream 沒產出時，錯誤訊息要帶上它的輸出，不能只說『失敗』。"""
        bad = tmp_dir / "00_raw" / "broken.awb"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"")  # 空檔 → 真實 vgmstream 無法解析

        res = unpack_one(bad, tmp_dir / "out", str(fake_exe))

        assert not res.ok
        assert "未產生任何輸出" in res.error
        assert "failed to open" in res.error, f"應帶上工具本身的輸出：{res.error}"
        assert not (tmp_dir / "out").exists() or not list((tmp_dir / "out").glob("*.wav"))

    def test_probe_failure_does_not_block_unpack(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        探測失敗時不可直接放棄：就算 `-m` 解析不出 subsong 數，
        只要解包能產出檔案就該成功（subsongs 由實際產出數回填）。
        """
        src = make_container(tmp_dir / "00_raw" / "plain_name.awb", 3)
        opts = UnpackOptions(probe_subsongs=True)
        res = unpack_one(src, tmp_dir / "out", str(fake_exe), opts)
        assert res.ok, res.error
        assert len(res.out_paths) == 1, "檔名無 fake_subsongs → 替身只解 1 條"
        assert res.subsongs == 1

    def test_rerun_skips_already_unpacked_container(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        冪等性：已解包過的容器重跑時必須**略過**，不能一路產生
        `#1__1.wav`、`#1__2.wav`。

        為什麼這件事重要：`all` 會連續跑 unpack + convert，你一天可能跑好幾次；
        沒有冪等性的話，幾千條語音重跑三次就爆量，而且 convert 會把重複檔案
        當成新素材再處理一遍。
        """
        src = make_container(tmp_dir / "00_raw" / "bank_fake_subsongs=2_fake_dur=2.awb", 2)
        dest = tmp_dir / "01_unpacked"

        first = unpack_one(src, dest, str(fake_exe))
        assert first.ok and not first.skipped
        before = {p.name: read_marker(p) for p in first.out_paths}

        second = unpack_one(src, dest, str(fake_exe))
        assert second.skipped, "重跑應標記為『已解包過』"
        assert not second.ok, "略過時不該回報為新產出"
        assert "已解包" in second.error
        assert len(list(dest.glob("*.wav"))) == 2, "不該產生重複檔案"

        # 原有產物必須原封不動
        for name, fingerprint in before.items():
            assert (dest / name).is_file(), f"舊產物不見了：{name}"
            assert read_marker(dest / name) == fingerprint, f"舊產物被動到了：{name}"

    def test_force_reextracts_without_replacing(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        force=True 時強制重解，但仍**不覆蓋**既有檔案 —— 用 `__N` 序號並存。
        原始素材與既有產物都是資產，寧可多佔硬碟也不要有任何一邊消失。
        """
        src = make_container(tmp_dir / "00_raw" / "bank_fake_subsongs=2_fake_dur=2.awb", 2)
        dest = tmp_dir / "01_unpacked"

        first = unpack_one(src, dest, str(fake_exe))
        assert first.ok
        before = {p.name: read_marker(p) for p in first.out_paths}

        second = unpack_one(src, dest, str(fake_exe), force=True)
        assert second.ok and not second.skipped
        assert len(second.out_paths) == 2
        assert len(list(dest.glob("*.wav"))) == 4, "force 重解應產生新檔而非覆蓋"
        assert all("__" in p.name for p in second.out_paths), [p.name for p in second.out_paths]
        assert all(read_marker(p) != "silent" for p in second.out_paths)
        for name, fingerprint in before.items():
            assert read_marker(dest / name) == fingerprint

    def test_no_probe_mode_still_unpacks(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        --no-probe 時完全沒有 subsong 資訊，此時必須走「展開」路徑
        （保守假設可能是多 subsong），不可退化成只解第一條。
        若最後只產出 1 個檔案，檔名要還原成不含 `#1` 的乾淨名稱。
        """
        src = make_container(tmp_dir / "00_raw" / "bank_fake_subsongs=5_fake_dur=2.awb", 5)
        opts = UnpackOptions(probe_subsongs=False)
        res = unpack_one(src, tmp_dir / "out", str(fake_exe), opts)
        assert res.ok
        assert len(res.out_paths) == 5, "未知 subsong 數時必須當成多 subsong 展開"
        assert all("#" in p.name for p in res.out_paths)

    def test_single_file_from_expansion_gets_clean_name(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        走展開路徑但只產出 1 個檔案時，`#1` 必須被拿掉。
        否則幾千個單句語音檔會全部變成 `voice#1.wav` 這種醜名字，
        而且重跑時的 idempotency 判斷也會跟著亂掉。
        """
        src = make_container(tmp_dir / "00_raw" / "solo.wem", 1)  # 替身 -m 只印 1 條
        opts = UnpackOptions(probe_subsongs=False)  # 強制走展開路徑
        res = unpack_one(src, tmp_dir / "out", str(fake_exe), opts)
        assert res.ok, res.error
        assert len(res.out_paths) == 1
        assert res.out_paths[0].name == "solo.wav", res.out_paths[0].name


# ---------------------------------------------------------------------------
# 3. 端到端：透過 CLI 跑 unpack
# ---------------------------------------------------------------------------
def _scaffold(root: Path, *, unpack_patch: dict | None = None) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)
    data = json.loads((REPO / "configs" / "pipeline.default.json").read_text(encoding="utf-8"))
    if unpack_patch:
        data["unpack"].update(unpack_patch)
    (root / "configs" / "pipeline.local.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def _run_cli(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return run_cli(args, root)


class TestUnpackViaCli:
    def test_cli_unpack_creates_products_and_reports(self, tmp_dir: Path, fake_exe: Path) -> None:
        root = _scaffold(tmp_dir, unpack_patch={"vgmstream_bin": str(fake_exe)})
        make_container(root / "data" / "00_raw" / "hero_a" / "bank_fake_subsongs=4.awb", 4)

        proc = _run_cli(["unpack", "-v"], root)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert "容器解包" in proc.stdout
        assert "可用產物共 4 個" in proc.stdout

        # 產物位置：01_unpacked/<來源相對結構>/<容器檔名>/（一個容器一個子資料夾）
        out = root / "data" / "01_unpacked" / "hero_a" / "bank_fake_subsongs=4"
        assert out.is_dir(), f"產物目錄不存在：{out}"
        wavs = sorted(out.glob("*.wav"))
        assert len(wavs) == 4, [p.name for p in wavs]
        # subsong 萬用字元有生效 → 檔名帶 #1..#4，而非單一檔案被覆蓋
        assert {p.name for p in wavs} == {
            "bank_fake_subsongs=4#1.wav",
            "bank_fake_subsongs=4#2.wav",
            "bank_fake_subsongs=4#3.wav",
            "bank_fake_subsongs=4#4.wav",
        }

    def test_cli_unpack_skips_oversized_container(self, tmp_dir: Path, fake_exe: Path) -> None:
        root = _scaffold(
            tmp_dir, unpack_patch={"vgmstream_bin": str(fake_exe), "max_subsongs": 10}
        )
        make_container(root / "data" / "00_raw" / "huge_fake_subsongs=500.awb", 500)

        proc = _run_cli(["unpack"], root)
        assert "跳過" in proc.stdout or "skipped" in proc.stdout
        assert "500" in proc.stdout
        assert not list((root / "data" / "01_unpacked").rglob("*.wav"))

    def test_cli_max_subsongs_flag_overrides_config(self, tmp_dir: Path, fake_exe: Path) -> None:
        root = _scaffold(
            tmp_dir, unpack_patch={"vgmstream_bin": str(fake_exe), "max_subsongs": 10}
        )
        make_container(root / "data" / "00_raw" / "bank_fake_subsongs=50.awb", 50)

        proc = _run_cli(["unpack", "--max-subsongs", "100"], root)
        assert proc.returncode == 0, proc.stdout
        assert "可用產物共 50 個" in proc.stdout

    def test_cli_unpack_then_convert_full_chain(self, tmp_dir: Path, fake_exe: Path) -> None:
        """
        真正的整合測試：容器 → unpack → convert。
        替身產生的 WAV 是 0.05 秒，會被時長規則淘汰 —— 這正好驗證
        「解包產物有被 convert 看到」且淘汰理由正確。
        """
        root = _scaffold(tmp_dir, unpack_patch={"vgmstream_bin": str(fake_exe)})
        make_container(root / "data" / "00_raw" / "bank_fake_subsongs=3.awb", 3)

        proc = _run_cli(["unpack"], root)
        assert proc.returncode == 0, proc.stdout

        proc = _run_cli(["convert", "-v"], root)
        assert proc.returncode == 0, proc.stdout
        assert "過短" in proc.stdout, f"解包產物應被 convert 處理：{proc.stdout}"

        with (root / "data" / "_reports" / "manifest_phase1.csv").open(
            encoding="utf-8-sig", newline=""
        ) as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "manifest 應該有解包產物的紀錄"
        assert all(r["status"] == "rejected" for r in rows)

    def test_cli_unpack_dry_run_writes_nothing(self, tmp_dir: Path, fake_exe: Path) -> None:
        root = _scaffold(tmp_dir, unpack_patch={"vgmstream_bin": str(fake_exe)})
        make_container(root / "data" / "00_raw" / "bank_fake_subsongs=3.awb", 3)
        proc = _run_cli(["unpack", "--dry-run"], root)
        assert proc.returncode == 0, proc.stdout
        assert not list((root / "data" / "01_unpacked").rglob("*.wav"))
