"""
vOS-VirtualVoice - 磁碟空間預檢測試 (Phase 1)

守著一個「不是失敗、而是部分成功」的危險情況：

解包會把壓縮音訊變成 PCM，資料量放大 8~18 倍。磁碟寫爆的後果不是乾淨的
失敗，而是**容器只解出一部分 subsong 卻沒有明顯錯誤** —— Phase 2 會把截斷的
素材當成正常資料餵進模型。所以必須在開始之前就算給使用者看。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import space  # noqa: E402
from space import (  # noqa: E402
    BYTES_PER_SECOND_24BIT_48K_MONO,
    MIN_FREE_BYTES,
    SAFETY_FACTOR,
    estimate_for_seconds,
    estimate_from_source_bytes,
    free_bytes,
    human_bytes,
)


class TestConstants:
    def test_bytes_per_second_matches_24bit_48k_mono(self) -> None:
        """
        24-bit 48000 Hz 單聲道 = 48000 樣本/秒 x 3 bytes = 144000 bytes/s。
        這個數字是整個預估的基礎，寫錯會讓所有估算失真。
        """
        assert BYTES_PER_SECOND_24BIT_48K_MONO == 144_000

    def test_expansion_range_is_physically_sensible(self) -> None:
        """8~18 倍是從「144000 B/s 除以 64~128 kbps」推出來的。"""
        assert 1.0 < space.UNPACK_EXPANSION_LOW < space.UNPACK_EXPANSION_HIGH < 100.0

    def test_safety_factor_above_one(self) -> None:
        assert SAFETY_FACTOR >= 1.2, "安全邊際太小就失去意義"
        assert MIN_FREE_BYTES >= 100 * 1024 * 1024, "至少留 100 MiB 緩衝"


class TestFreeBytes:
    def test_returns_positive_for_existing_dir(self, tmp_dir: Path) -> None:
        assert free_bytes(tmp_dir) > 0

    def test_walks_up_to_existing_ancestor(self, tmp_dir: Path) -> None:
        """
        使用者常常在「還沒建立的輸出目錄」上問空間。
        那必須能回答，而不是直接失敗 —— 否則預檢會在第一次執行時就壞掉。
        """
        missing = tmp_dir / "not" / "yet" / "created"
        assert not missing.exists()
        assert free_bytes(missing) > 0

    def test_never_raises(self, tmp_dir: Path) -> None:
        assert free_bytes(Path("")) >= 0


class TestEstimate:
    def test_compressed_source_uses_expansion_range(self, tmp_dir: Path) -> None:
        src = 100 * 1024 * 1024
        est = estimate_from_source_bytes(src, tmp_dir)
        assert est.low_bytes == int(src * space.UNPACK_EXPANSION_LOW)
        assert est.high_bytes == int(src * space.UNPACK_EXPANSION_HIGH)
        assert est.high_bytes > est.low_bytes, "壓縮率未知，必須給區間"

    def test_pcm_source_does_not_get_expansion_multiplier(self, tmp_dir: Path) -> None:
        """
        來源已經是未壓縮音訊時，不該再乘 8~18 倍 ——
        那會讓預估變成天文數字，使用者會以為工具壞了。
        """
        src = 100 * 1024 * 1024
        est = estimate_from_source_bytes(src, tmp_dir, already_pcm=True)
        assert est.high_bytes < src * 1.5, f"PCM 來源不該被放大：{est.high_bytes}"
        assert est.low_bytes < est.high_bytes

    def test_needed_includes_safety_and_buffer(self, tmp_dir: Path) -> None:
        est = estimate_from_source_bytes(10 * 1024 * 1024, tmp_dir)
        assert est.needed_bytes == int(est.high_bytes * SAFETY_FACTOR) + MIN_FREE_BYTES

    def test_sufficient_when_plenty_of_space(self, tmp_dir: Path) -> None:
        est = estimate_from_source_bytes(1024, tmp_dir)
        assert est.sufficient
        assert est.shortfall_bytes == 0

    def test_insufficient_when_source_is_absurdly_large(self, tmp_dir: Path) -> None:
        """
        來源 500 TB -> 一定不夠（沒有任何測試機器有這種空間）。
        這驗證 insufficient 這條路徑真的會被觸發，而不是永遠回 True。
        """
        est = estimate_from_source_bytes(500 * 1024**4, tmp_dir)
        assert not est.sufficient
        assert est.shortfall_bytes > 0

    def test_estimate_for_seconds_is_exact(self, tmp_dir: Path) -> None:
        """已知時長時可以精確預估：Phase 1 產物大小非常可預測。"""
        est = estimate_for_seconds(600.0, tmp_dir)  # 10 分鐘
        expected = int(600 * BYTES_PER_SECOND_24BIT_48K_MONO)
        assert est.low_bytes == est.high_bytes == expected
        assert 80 * 1024 * 1024 < expected < 85 * 1024 * 1024  # 約 82.4 MiB


class TestRender:
    def test_sufficient_message_is_reassuring(self, tmp_dir: Path) -> None:
        text = estimate_from_source_bytes(1024, tmp_dir).render()
        assert "空間充足" in text
        assert "預估產物" in text
        assert "建議保留" in text

    def test_insufficient_message_shows_shortfall(self, tmp_dir: Path) -> None:
        text = estimate_from_source_bytes(500 * 1024**4, tmp_dir).render()
        assert "空間不足" in text
        assert "還缺" in text

    def test_render_mentions_safety_factor(self, tmp_dir: Path) -> None:
        """要把安全邊際講出來，使用者才知道為什麼數字比預期大。"""
        text = estimate_from_source_bytes(1024, tmp_dir).render()
        assert "安全邊際" in text

    def test_human_bytes_units(self) -> None:
        assert human_bytes(0) == "0.0 B"
        assert human_bytes(2048).endswith("KiB")
        assert human_bytes(5 * 1024**2).endswith("MiB")
        assert human_bytes(3 * 1024**3).endswith("GiB")
        assert human_bytes(2 * 1024**4).endswith("TiB")


class TestCliPreflight:
    def test_unpack_reports_space_estimate(self, tmp_dir: Path) -> None:
        """實際執行時一定要印出空間預估 —— 使用者才能事前判斷。"""
        import json

        from conftest import make_fake_exe, run_cli

        root = tmp_dir / "repo"
        (root / "configs").mkdir(parents=True)
        for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
            (root / "data" / sub).mkdir(parents=True, exist_ok=True)
        fake = make_fake_exe(tmp_dir / "bin")
        data = json.loads((REPO / "configs" / "pipeline.default.json").read_text(encoding="utf-8"))
        data["grouping"] = {"by": "none"}
        data["unpack"]["vgmstream_bin"] = str(fake)
        (root / "configs" / "pipeline.local.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "data" / "00_raw" / "v_fake_subsongs=2_fake_dur=3.awb").write_bytes(
            b"AFS2" + b"\x00" * 64
        )

        proc = run_cli(["unpack"], root)
        assert proc.returncode == 0, proc.stdout
        assert "預估產物" in proc.stdout, proc.stdout
        assert "空間充足" in proc.stdout, proc.stdout

    def test_dry_run_skips_space_check(self, tmp_dir: Path) -> None:
        """
        dry-run 不寫檔，所以不該被空間預檢擋下 ——
        否則使用者連「看看會發生什麼」都做不到。
        """
        import json

        from conftest import make_fake_exe, run_cli

        root = tmp_dir / "repo"
        (root / "configs").mkdir(parents=True)
        for sub in ("00_raw", "01_unpacked", "02_wav24_48k", "99_rejected", "_reports"):
            (root / "data" / sub).mkdir(parents=True, exist_ok=True)
        fake = make_fake_exe(tmp_dir / "bin")
        data = json.loads((REPO / "configs" / "pipeline.default.json").read_text(encoding="utf-8"))
        data["grouping"] = {"by": "none"}
        data["unpack"]["vgmstream_bin"] = str(fake)
        (root / "configs" / "pipeline.local.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "data" / "00_raw" / "v_fake_subsongs=2_fake_dur=3.awb").write_bytes(
            b"AFS2" + b"\x00" * 64
        )

        proc = run_cli(["unpack", "--dry-run"], root)
        assert proc.returncode == 0, proc.stdout
        assert "預估產物" not in proc.stdout