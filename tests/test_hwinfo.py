"""
vOS-VirtualVoice - Phase 3 環境偵測與參數擬合測試

守著兩個容易搞錯、但代價很高的判斷：

  1. **「沒有 GPU」與「裝到 CPU-only 的 PyTorch」必須分開。**
     兩者症狀完全一樣（cuda.is_available() == False），但解法完全不同。
     誤判會讓人白花錢買卡或租雲端。

  2. **等效 batch size 必須守恆。**
     batch_size × grad_accum 才是影響梯度品質的量。VRAM 不夠時該縮小
     batch_size、提高 grad_accum，而不是降低等效 batch size ——
     否則換一張卡就會改變學習動態，訓練結果無法互相比較。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import hwinfo  # noqa: E402
import trainfit  # noqa: E402
from hwinfo import (  # noqa: E402
    CPU_ONLY_BUILD_MARKER,
    DeviceInfo,
    EnvReport,
    _guess_vram_from_name,
    detect_torch_environment,
    render_env_report,
)
from trainfit import (  # noqa: E402
    TARGET_EFFECTIVE_BATCH,
    VRAM_PROFILES,
    fit_training_params,
    render_params,
)


def _report(kind: str, vram: float = 0.0, cpu_count: int = 16, cpu_only: bool = False) -> EnvReport:
    rep = EnvReport(cpu_count=cpu_count, torch_version="2.13.0")
    if kind == "cuda":
        rep.devices.append(DeviceInfo("cuda", "Test GPU", total_vram_gb=vram, free_vram_gb=vram))
    elif kind == "mps":
        rep.devices.append(DeviceInfo("mps", "Apple M2"))
    rep.torch_is_cpu_only_build = cpu_only
    rep.devices.append(DeviceInfo("cpu", "Test CPU"))
    return rep


class TestEnvironmentDetection:
    def test_detects_something_on_this_machine(self) -> None:
        """任何機器上都至少要有 CPU 這個後備裝置。"""
        rep = detect_torch_environment()
        assert rep.devices, "至少要偵測到 CPU"
        assert any(d.kind == "cpu" for d in rep.devices)
        assert rep.cpu_count > 0

    def test_reports_torch_version_when_installed(self) -> None:
        rep = detect_torch_environment()
        assert rep.torch_version, "本專案環境一定有 torch（列在 requirements）"

    def test_primary_prefers_gpu_over_cpu(self) -> None:
        rep = _report("cuda", vram=24)
        assert rep.primary.kind == "cuda"
        assert rep.gpu_available

    def test_gpu_available_false_for_cpu_only(self) -> None:
        rep = _report("cpu")
        assert not rep.gpu_available
        assert rep.primary.kind == "cpu"

    def test_cpu_only_build_is_flagged_distinctly(self) -> None:
        """
        核心區分：CPU-only build 必須產生**不同的**警告訊息。

        這是本模組存在的理由 —— 同一個 False，兩種完全不同的處置。
        """
        rep = detect_torch_environment()
        if rep.torch_is_cpu_only_build:
            assert any("CPU-only" in w for w in rep.warnings), rep.warnings
            assert any("永遠" in w or "用不到" in w for w in rep.warnings), (
                "必須強調這個 build 永遠看不到 GPU，否則使用者會以為是硬體問題"
            )

    def test_marker_constant_matches_real_version_strings(self) -> None:
        """`+cpu` 是 PyTorch 標記 CPU-only build 的方式，不可寫錯。"""
        assert CPU_ONLY_BUILD_MARKER == "+cpu"
        assert CPU_ONLY_BUILD_MARKER in "2.13.0+cpu"

    def test_detection_never_raises(self) -> None:
        """探測失敗不該讓工具崩潰 —— 它只是報告。"""
        rep = detect_torch_environment()
        assert isinstance(rep.warnings, list)


class TestVramGuess:
    @pytest.mark.parametrize(
        "name,expected_min",
        [
            ("NVIDIA GeForce RTX 4090", 24),
            ("NVIDIA GeForce RTX 3060", 12),
            ("NVIDIA GeForce RTX 4060 Ti", 16),
            ("Tesla T4", 16),
        ],
    )
    def test_known_cards(self, name: str, expected_min: float) -> None:
        assert _guess_vram_from_name(name) >= expected_min

    def test_unknown_card_returns_zero(self) -> None:
        assert _guess_vram_from_name("Some Unknown Accelerator") == 0.0

    def test_case_insensitive(self) -> None:
        assert _guess_vram_from_name("nvidia geforce rtx 4090") == 24.0


class TestParameterFitting:
    def test_effective_batch_never_falls_below_target(self) -> None:
        """
        最重要的一條：**所有** VRAM 等級的等效 batch 都不可低於目標值。

        batch_size × grad_accum 才是影響梯度品質的量；換一張卡不該讓
        梯度品質下降，否則訓練結果無法互相比較。

        為什麼是「不低於」而不是「等於」：grad_accum 必須向上取整。
        例如 16 GB profile 的 batch 是 6，round(8/6) = 1 會讓等效 batch
        只剩 6（低於目標）；ceil(8/6) = 2 得到等效 12（略高）。
        **略高只是慢一點，低於則是品質妥協** —— 這個不對稱是刻意的。
        """
        for vram in (24, 16, 12, 10, 8, 6, 4, 3):
            params = fit_training_params(_report("cuda", vram=float(vram)))
            assert params.effective_batch >= TARGET_EFFECTIVE_BATCH, (
                f"{vram} GB 的等效 batch 只有 {params.effective_batch}，"
                f"不可低於目標 {TARGET_EFFECTIVE_BATCH}"
            )

    def test_16gb_profile_uses_ceil_not_round(self) -> None:
        """
        回歸測試：16 GB profile 曾因用 round 而得到等效 batch 6。

        batch 6、目標 8：round(1.33)=1（錯，等效 6）、ceil(1.33)=2（對，等效 12）。
        這個案例特別值得固定下來，因為它是唯一會踩到 round 的等級。
        """
        params = fit_training_params(_report("cuda", vram=16.0))
        assert params.batch_size == 6
        assert params.grad_accum == 2, f"應為 2（ceil），得到 {params.grad_accum}"
        assert params.effective_batch == 12

    def test_smaller_vram_gets_smaller_batch_larger_accum(self) -> None:
        """VRAM 越小，batch 越小、accum 越大 —— 這是唯一正確的取捨方向。"""
        big = fit_training_params(_report("cuda", vram=24))
        small = fit_training_params(_report("cuda", vram=8))
        assert small.batch_size < big.batch_size, "小 VRAM 的 batch 應該更小"
        assert small.grad_accum >= big.grad_accum, "小 VRAM 的累積步數應該不減"

    def test_low_vram_enables_gradient_checkpointing(self) -> None:
        """記憶體吃緊時必須啟用 gradient checkpointing。"""
        small = fit_training_params(_report("cuda", vram=6))
        assert small.grad_checkpointing, "6 GB 應該啟用 gradient checkpointing"
        big = fit_training_params(_report("cuda", vram=24))
        assert not big.grad_checkpointing, "24 GB 不需要犧牲速度"

    def test_big_vram_uses_bf16_small_uses_fp16(self) -> None:
        """
        bf16 是數值穩定性取捨（範圍同 fp32，不易 NaN），需要 Ampere 以後；
        舊硬體用 fp16。不是純效能選擇。
        """
        assert fit_training_params(_report("cuda", vram=24)).precision == "bf16"
        assert fit_training_params(_report("cuda", vram=6)).precision == "fp16"

    def test_cpu_uses_fp32_and_zero_workers(self) -> None:
        """CPU 訓練：fp16 反而更慢（缺原生支援），且多 worker 只會搶核心。"""
        params = fit_training_params(_report("cpu", cpu_count=20))
        assert params.device == "cpu"
        assert params.precision == "fp32"
        assert params.num_workers == 0

    def test_cpu_params_warn_about_infeasibility(self) -> None:
        """純 CPU 必須明確說出「不實用」，而不是默默給一組參數。"""
        params = fit_training_params(_report("cpu"))
        joined = " ".join(params.notes)
        assert "CPU" in joined
        assert "數小時" in joined or "不建議" in joined or "雲端" in joined

    def test_cpu_only_build_warning_surfaces_in_params(self) -> None:
        """
        有 GPU 但裝到 CPU-only build 時，參數建議必須警告 ——
        否則使用者會以為「反正就是慢」，而不知道問題其實可修。
        """
        rep = _report("cuda", vram=24, cpu_only=True)
        params = fit_training_params(rep)
        joined = " ".join(params.notes)
        assert "CPU-only" in joined or "用不到" in joined, params.notes

    def test_workers_never_starve_the_main_process(self) -> None:
        """worker 數不可吃滿所有核心，否則主行程會被餓死。"""
        for cpu_count in (1, 2, 4, 8, 16, 64):
            params = fit_training_params(_report("cuda", vram=24, cpu_count=cpu_count))
            assert params.num_workers >= 0
            if cpu_count > 2:
                assert params.num_workers <= cpu_count - 1, (
                    f"{cpu_count} 核卻配了 {params.num_workers} 個 worker"
                )

    def test_mps_uses_unified_memory_assumption(self) -> None:
        params = fit_training_params(_report("mps"))
        assert params.device == "mps"
        assert params.vram_gb > 0, "MPS 應以保留給模型的統一記憶體估算"
        assert any("統一記憶體" in n for n in params.notes)

    def test_unknown_vram_falls_back_to_name_guess(self) -> None:
        """PyTorch 回報不到 VRAM 時，用型號名稱反推而不是直接放棄。"""
        rep = EnvReport(cpu_count=8)
        # total_vram_gb = 0 但名稱認得
        rep.devices.append(DeviceInfo("cuda", "NVIDIA GeForce RTX 4090", total_vram_gb=0.0))
        rep.devices.append(DeviceInfo("cpu", "CPU"))
        params = fit_training_params(rep)
        assert params.vram_gb >= 20, f"應從型號推測出 24 GB，得到 {params.vram_gb}"
        assert any("型號" in n for n in params.notes)


class TestRendering:
    def test_env_report_mentions_every_device(self) -> None:
        text = render_env_report(_report("cuda", vram=12))
        assert "Test GPU" in text
        assert "Test CPU" in text
        assert "12.0 GB" in text

    def test_env_report_gives_verdict(self) -> None:
        assert "可用 GPU 訓練" in render_env_report(_report("cuda", vram=12))
        assert "只能用 CPU 訓練" in render_env_report(_report("cpu"))

    def test_cpu_verdict_recommends_alternatives(self) -> None:
        """CPU 結論必須給出路，不能只說「很慢」。"""
        text = render_env_report(_report("cpu"))
        assert "Colab" in text or "雲端" in text, text

    def test_params_render_includes_copy_paste_snippet(self) -> None:
        """參數表要能直接複製貼上，否則使用者還得手動轉抄。"""
        text = render_params(fit_training_params(_report("cuda", vram=12)))
        assert "batch_size" in text
        assert "gradient_accumulation_steps" in text
        assert "{" in text and "}" in text, "應包含可貼上的 JSON 片段"

    def test_params_render_shows_effective_batch(self) -> None:
        text = render_params(fit_training_params(_report("cuda", vram=8)))
        assert "等效 batch" in text


class TestProfilesTable:
    def test_profiles_are_ordered_descending(self) -> None:
        """比對邏輯由高到低，順序錯會選到錯的 profile。"""
        thresholds = [p["min_vram_gb"] for p in VRAM_PROFILES]
        assert thresholds == sorted(thresholds, reverse=True), thresholds

    def test_profiles_cover_zero(self) -> None:
        """必須有 min_vram_gb = 0 的後備，否則小 VRAM 會找不到 profile。"""
        assert min(p["min_vram_gb"] for p in VRAM_PROFILES) == 0

    def test_every_profile_is_internally_consistent(self) -> None:
        for p in VRAM_PROFILES:
            assert p["batch_size"] >= 1
            assert p["precision"] in ("fp16", "bf16", "fp32")
            assert p["label"]
            assert p["note"]