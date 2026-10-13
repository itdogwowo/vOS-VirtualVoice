"""
vOS-VirtualVoice - 工具鏈診斷測試 (Phase 1)

外部工具的安裝失敗方式極具誤導性，這組測試守著三個「會被誤判成別的問題」
的症狀：

  1. vgmstream 找到 exe 但少了 DLL -> Wwise 的 Vorbis 語音解不出來，
     錯誤訊息卻只說「不支援此格式」，讓人以為素材壞了。
  2. 設定檔指到一個路徑，但 doctor 說找不到 -> 使用者開始懷疑設定沒生效。
  3. exe 存在但跑不起來（架構不符／缺 VC runtime／被防毒隔離）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import config as cfgmod  # noqa: E402
import diag  # noqa: E402
from conftest import make_fake_exe, run_cli  # noqa: E402
from diag import (  # noqa: E402
    CRITICAL_DLLS,
    WINDOWS_VGMSTREAM_DLLS,
    check_dlls,
    diagnose_ffmpeg,
    diagnose_vgmstream,
    locate_tool,
)


def _scaffold(root: Path, *, vgmstream_bin: str | None = None) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)
    (root / "tools" / "bin").mkdir(parents=True, exist_ok=True)
    data = json.loads((REPO / "configs" / "pipeline.default.json").read_text(encoding="utf-8"))
    data["grouping"] = {"by": "none"}
    if vgmstream_bin is not None:
        data["unpack"]["vgmstream_bin"] = vgmstream_bin
    (root / "configs" / "pipeline.local.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


class TestLocateTool:
    def test_reports_not_found_when_absent(self, tmp_dir: Path, monkeypatch) -> None:
        root = _scaffold(tmp_dir / "repo")
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(root))
        monkeypatch.delenv("VOS_VGMSTREAM_BIN", raising=False)
        monkeypatch.setenv("PATH", str(root / "empty"))
        cfg = cfgmod.load_config(root / "configs" / "pipeline.local.json")
        path, source = locate_tool("vgmstream-cli", cfg)
        assert path is None
        assert "未找到" in source

    def test_finds_tool_declared_in_config_and_says_so(self, tmp_dir: Path, monkeypatch) -> None:
        """
        設定檔指定路徑時必須找得到**並說出來源**。
        以前 doctor 直接用 find_external_bin(name) 不帶 hint，設定檔寫了也沒用。
        """
        root = _scaffold(tmp_dir / "repo")
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(root))
        monkeypatch.delenv("VOS_VGMSTREAM_BIN", raising=False)
        fake = make_fake_exe(root / "vendor" / "vgmstream")
        cfg_path = root / "configs" / "pipeline.local.json"
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        data["unpack"]["vgmstream_bin"] = str(fake)
        cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        cfg = cfgmod.load_config(cfg_path)

        path, source = locate_tool("vgmstream-cli", cfg)
        assert path is not None and Path(path) == fake
        assert "設定檔" in source, source

    def test_finds_tool_in_repo_tools_bin(self, tmp_dir: Path, monkeypatch) -> None:
        root = _scaffold(tmp_dir / "repo")
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(root))
        monkeypatch.delenv("VOS_VGMSTREAM_BIN", raising=False)
        make_fake_exe(root / "tools" / "bin")
        cfg = cfgmod.load_config(root / "configs" / "pipeline.local.json")
        path, source = locate_tool("vgmstream-cli", cfg)
        assert path is not None, "tools/bin 是本專案建議的放置位置，必須找得到"
        assert "tools/bin" in source

    def test_env_override_wins_and_is_reported(self, tmp_dir: Path, monkeypatch) -> None:
        """
        環境變數優先：這是 CLI 把設定檔路徑傳給底層模組的機制。
        若它不生效，unpack 找得到、convert 卻會說「未找到」。
        """
        root = _scaffold(tmp_dir / "repo")
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(root))
        fake = make_fake_exe(root / "envbin")
        monkeypatch.setenv("VOS_VGMSTREAM_BIN", str(fake))
        cfg = cfgmod.load_config(root / "configs" / "pipeline.local.json")
        path, source = locate_tool("vgmstream-cli", cfg)
        assert path is not None and Path(path) == fake
        assert "VOS_VGMSTREAM_BIN" in source


class TestDllCheck:
    def test_all_dlls_present_reports_nothing(self, tmp_dir: Path) -> None:
        exe = tmp_dir / "vgmstream-cli.exe"
        exe.write_bytes(b"MZ")
        for dll in WINDOWS_VGMSTREAM_DLLS:
            (tmp_dir / dll).write_bytes(b"x")
        assert check_dlls(str(exe)) == []

    def test_missing_dlls_are_listed(self, tmp_dir: Path) -> None:
        exe = tmp_dir / "vgmstream-cli.exe"
        exe.write_bytes(b"MZ")
        (tmp_dir / "libvorbis.dll").write_bytes(b"x")
        missing = check_dlls(str(exe))
        assert "libvorbis.dll" not in missing
        assert "libmpg123-0.dll" in missing
        assert len(missing) == len(WINDOWS_VGMSTREAM_DLLS) - 1

    def test_case_insensitive_matching(self, tmp_dir: Path) -> None:
        """Windows 檔名不分大小寫，檢查也要一致，否則會誤報缺檔。"""
        exe = tmp_dir / "vgmstream-cli.exe"
        exe.write_bytes(b"MZ")
        for dll in WINDOWS_VGMSTREAM_DLLS:
            (tmp_dir / dll.upper()).write_bytes(b"x")
        assert check_dlls(str(exe)) == []

    def test_non_windows_returns_empty(self, tmp_dir: Path, monkeypatch) -> None:
        """非 Windows 沒有「DLL 放隔壁」這種規則，不該亂報缺檔。"""
        exe = tmp_dir / "vgmstream-cli"
        exe.write_bytes(b"x")
        monkeypatch.setattr(diag.os, "name", "posix")
        assert check_dlls(str(exe)) == []

    def test_missing_exe_returns_empty(self, tmp_dir: Path) -> None:
        assert check_dlls(str(tmp_dir / "nope.exe")) == []
        assert check_dlls(None) == []

    def test_critical_dlls_map_to_symptoms(self) -> None:
        """關鍵 DLL 的說明必須點出**症狀**，否則使用者看到「缺 DLL」只會茫然。"""
        assert "libvorbis.dll" in CRITICAL_DLLS
        assert "Vorbis" in CRITICAL_DLLS["libvorbis.dll"]


class TestDiagnose:
    def test_missing_vgmstream_gives_actionable_note(self, tmp_dir: Path, monkeypatch) -> None:
        root = _scaffold(tmp_dir / "repo")
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(root))
        monkeypatch.delenv("VOS_VGMSTREAM_BIN", raising=False)
        monkeypatch.setenv("PATH", str(root / "empty"))
        cfg = cfgmod.load_config(root / "configs" / "pipeline.local.json")
        status = diagnose_vgmstream(cfg, probe=False)
        assert not status.available and not status.healthy
        assert status.notes, "沒找到工具時必須說明後果"
        assert any(".wem" in n or "容器" in n for n in status.notes)

    def test_real_fake_tool_runs(self, tmp_dir: Path, monkeypatch) -> None:
        root = _scaffold(tmp_dir / "repo")
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(root))
        fake = make_fake_exe(root / "bin")
        monkeypatch.setenv("VOS_VGMSTREAM_BIN", str(fake))
        cfg = cfgmod.load_config(root / "configs" / "pipeline.local.json")
        status = diagnose_vgmstream(cfg, probe=True)
        assert status.available and status.runs, status.notes

    @pytest.mark.skipif(os.name != "nt", reason="DLL 檢查只在 Windows 有意義")
    def test_missing_dlls_make_status_unhealthy_with_symptom(
        self, tmp_dir: Path, monkeypatch
    ) -> None:
        """
        核心情境：執行檔在、跑得起來，但**少了 DLL**。

        這是最危險的狀態 —— vgmstream 會啟動、會處理部分格式，卻在 Wwise 的
        Vorbis 語音上失敗，而錯誤訊息只說「不支援此格式」。
        所以 healthy 必須是 False，且說明要點出症狀。
        """
        root = _scaffold(tmp_dir / "repo")
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(root))
        fake = make_fake_exe(root / "bin")  # 刻意不放任何 DLL
        monkeypatch.setenv("VOS_VGMSTREAM_BIN", str(fake))
        cfg = cfgmod.load_config(root / "configs" / "pipeline.local.json")
        status = diagnose_vgmstream(cfg, probe=True)
        assert status.available and status.runs, "執行檔本身是好的"
        assert not status.healthy, "缺 DLL 不可算健康"
        assert "libvorbis.dll" in status.missing_dlls
        assert any("Vorbis" in n for n in status.notes), status.notes

    def test_ffmpeg_missing_note_mentions_formats(self, tmp_dir: Path, monkeypatch) -> None:
        root = _scaffold(tmp_dir / "repo")
        monkeypatch.setenv(cfgmod.ENV_REPO_ROOT, str(root))
        monkeypatch.delenv("VOS_FFMPEG_BIN", raising=False)
        monkeypatch.setenv("PATH", str(root / "empty"))
        cfg = cfgmod.load_config(root / "configs" / "pipeline.local.json")
        status = diagnose_ffmpeg(cfg, probe=False)
        assert not status.available
        assert any("m4a" in n or "aac" in n for n in status.notes), status.notes


class TestDoctorIntegration:
    def test_doctor_shows_tool_source(self, tmp_dir: Path) -> None:
        """
        doctor 必須顯示工具是從哪裡找到的 —— 否則使用者明明設好了設定檔
        卻看到紅字，只會開始懷疑設定沒生效。
        """
        fake = make_fake_exe(tmp_dir / "bin")
        root = _scaffold(tmp_dir / "repo", vgmstream_bin=str(fake))
        proc = run_cli(["doctor"], root)
        assert proc.returncode == 0, proc.stdout
        assert "vgmstream-cli" in proc.stdout
        assert "設定檔" in proc.stdout or "VOS_VGMSTREAM_BIN" in proc.stdout, proc.stdout