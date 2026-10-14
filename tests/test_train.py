"""
vOS-VirtualVoice - Phase 3 訓練啟動器測試

守著三個設計主張：

  1. **precheck 是強制擋門，不是建議。** 訓練失敗的代價是數小時，
     擋在門口的代價是 10 秒。這條界線必須自動化，不能靠人記得。
  2. **參數自動寫進設定檔。** 人抄數字一定會出錯，而且出錯時不會報錯 ——
     訓練只是跑得比較差，你無從察覺。
  3. **第一次使用時必然缺件。** 沒有 GPU、沒有 GPT-SoVITS、沒有資料集 ——
     啟動器在那種狀態下必須給出精確的下一步，而不是一句「失敗」。
     一個只在萬事俱備時能運作的啟動器沒有價值。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from hwinfo import DeviceInfo, EnvReport  # noqa: E402
from precheck import CheckResult, PrecheckReport, run_precheck  # noqa: E402
from train import (  # noqa: E402
    GPT_SOVITS_MARKERS,
    build_training_config,
    check_training_deps,
    locate_gpt_sovits,
    plan_training,
    render_plan,
    write_config,
)
from trainfit import fit_training_params  # noqa: E402


def _env(has_gpu: bool = True) -> EnvReport:
    rep = EnvReport(cpu_count=16, torch_version="2.13.0")
    if has_gpu:
        rep.devices.append(DeviceInfo("cuda", "Test GPU", total_vram_gb=12, free_vram_gb=12))
    rep.devices.append(DeviceInfo("cpu", "Test CPU"))
    return rep


def _ok_precheck() -> PrecheckReport:
    rep = PrecheckReport()
    rep.add("PyTorch 可用", "pass", "ok")
    return rep


def _failing_precheck() -> PrecheckReport:
    rep = PrecheckReport()
    rep.add("GPU 可用性", "fail", "沒有可用的 GPU", "見 PHASE3 文件")
    return rep


def _fake_gpt_sovits(root: Path) -> Path:
    """建立一個看起來像 GPT-SoVITS 的目錄。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "s2_train.py").write_text("# stub", encoding="utf-8")
    (root / "GPT_SoVITS").mkdir(exist_ok=True)
    return root


class TestLocateGptSovits:
    def test_returns_none_when_absent(self, tmp_dir: Path, monkeypatch) -> None:
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir))
        monkeypatch.delenv("VOS_GPT_SOVITS_DIR", raising=False)
        path, source = locate_gpt_sovits(None)
        assert path is None
        assert "未找到" in source

    def test_finds_dir_with_marker_files(self, tmp_dir: Path, monkeypatch) -> None:
        root = _fake_gpt_sovits(tmp_dir / "repo" / "GPT-SoVITS")
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        monkeypatch.delenv("VOS_GPT_SOVITS_DIR", raising=False)
        path, source = locate_gpt_sovits(None)
        assert path == root
        assert "GPT-SoVITS/" in source

    def test_empty_dir_is_not_accepted(self, tmp_dir: Path, monkeypatch) -> None:
        """
        只有空目錄不算找到 —— 否則啟動器會以為萬事俱備，
        然後在真正啟動時才發現沒有訓練入口檔。
        """
        (tmp_dir / "repo" / "GPT-SoVITS").mkdir(parents=True)
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        monkeypatch.delenv("VOS_GPT_SOVITS_DIR", raising=False)
        path, _ = locate_gpt_sovits(None)
        assert path is None, "空目錄不該被當成 GPT-SoVITS"

    def test_env_var_takes_priority(self, tmp_dir: Path, monkeypatch) -> None:
        custom = _fake_gpt_sovits(tmp_dir / "elsewhere")
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        monkeypatch.setenv("VOS_GPT_SOVITS_DIR", str(custom))
        path, source = locate_gpt_sovits(None)
        assert path == custom
        assert "VOS_GPT_SOVITS_DIR" in source

    def test_markers_include_training_entrypoints(self) -> None:
        """s1_train.py / s2_train.py 是啟動器實際要呼叫的檔案。"""
        assert "s1_train.py" in GPT_SOVITS_MARKERS or "s2_train.py" in GPT_SOVITS_MARKERS


class TestDependencyCheck:
    def test_reports_missing_as_pairs(self) -> None:
        """缺件必須帶 pip 名稱，否則使用者不知道要裝什麼。"""
        missing = check_training_deps()
        for module, package in missing:
            assert module
            assert package, f"{module} 缺少對應的 pip 套件名"

    def test_known_heavy_deps_are_checked(self) -> None:
        missing_names = {m for m, _ in check_training_deps()}
        # 這幾個是本機確實沒有的（實測），必須在檢查範圍內
        assert "pytorch_lightning" in missing_names or "librosa" in missing_names, (
            "至少應偵測到本機缺少的訓練依賴"
        )


class TestTrainingPlan:
    def test_failing_precheck_blocks(self, tmp_dir: Path) -> None:
        """
        核心主張：**precheck 有 fail 就必須阻擋。**
        這是整個啟動器存在的理由。
        """
        env = _env(has_gpu=False)
        params = fit_training_params(env)
        plan = plan_training(env, params, _failing_precheck())
        assert not plan.ready
        assert any("環境" in b and "GPU" in b for b in plan.blockers)

    def test_precheck_hint_is_carried_into_blocker(self) -> None:
        """阻擋訊息要帶上原本的建議，否則使用者只知道壞了、不知怎麼修。"""
        env = _env(has_gpu=False)
        plan = plan_training(env, fit_training_params(env), _failing_precheck())
        joined = " ".join(plan.blockers)
        assert "PHASE3" in joined or "文件" in joined

    def test_missing_gpt_sovits_blocker_gives_clone_command(self, tmp_dir: Path, monkeypatch) -> None:
        """
        缺 GPT-SoVITS 時必須給出**可直接執行**的指令，
        而不是「請先安裝 GPT-SoVITS」這種廢話。
        """
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        monkeypatch.delenv("VOS_GPT_SOVITS_DIR", raising=False)
        env = _env()
        plan = plan_training(env, fit_training_params(env), _ok_precheck())
        joined = " ".join(plan.blockers)
        assert "git clone" in joined, joined
        assert "github.com" in joined

    def test_missing_dataset_blocker_points_to_convert(self, tmp_dir: Path, monkeypatch) -> None:
        """缺資料集時要指向產出資料集的那個指令。"""
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        env = _env()
        plan = plan_training(
            env,
            fit_training_params(env),
            _ok_precheck(),
            data_dir=tmp_dir / "no-such-dir",
        )
        joined = " ".join(plan.blockers)
        assert "convert" in joined or "Phase" in joined, joined

    def test_empty_dataset_is_flagged(self, tmp_dir: Path, monkeypatch) -> None:
        """目錄存在但沒有 wav 也算缺件 —— 空目錄不能當作準備好了。"""
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        empty = tmp_dir / "dataset"
        empty.mkdir(parents=True)
        env = _env()
        plan = plan_training(
            env, fit_training_params(env), _ok_precheck(), data_dir=empty
        )
        assert any("沒有 .wav" in b for b in plan.blockers), plan.blockers

    def test_dataset_with_wavs_is_reported_ready(self, tmp_dir: Path, monkeypatch) -> None:
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        ds = tmp_dir / "dataset"
        ds.mkdir(parents=True)
        (ds / "a.wav").write_bytes(b"RIFF")
        env = _env()
        plan = plan_training(env, fit_training_params(env), _ok_precheck(), data_dir=ds)
        assert any("1 個 wav" in n for n in plan.notes), plan.notes

    def test_gpt_sovits_found_is_noted(self, tmp_dir: Path, monkeypatch) -> None:
        root = _fake_gpt_sovits(tmp_dir / "repo" / "GPT-SoVITS")
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        monkeypatch.delenv("VOS_GPT_SOVITS_DIR", raising=False)
        env = _env()
        plan = plan_training(env, fit_training_params(env), _ok_precheck())
        assert plan.gpt_sovits_dir == root
        assert any("GPT-SoVITS" in n for n in plan.notes)

    def test_never_raises_with_no_arguments(self) -> None:
        """最原始的呼叫（什麼都沒給）也必須能運作。"""
        env = _env(has_gpu=False)
        plan = plan_training(env, fit_training_params(env), _failing_precheck())
        assert isinstance(plan.blockers, list)


class TestTrainingConfig:
    def test_config_uses_fitted_params_not_defaults(self) -> None:
        """
        核心主張：設定檔裡的參數必須來自 `trainfit` 的擬合，
        而不是任何寫死的預設值 —— 否則在小 VRAM 上就是 OOM。
        """
        env = _env()
        params = fit_training_params(env)
        config = build_training_config(env, params, data_dir=Path("/d"), output_dir=Path("/o"))
        assert config["training"]["batch_size"] == params.batch_size
        assert config["training"]["gradient_accumulation_steps"] == params.grad_accum
        assert config["training"]["precision"] == params.precision

    def test_config_declares_it_is_generated(self) -> None:
        """必須標明是自動產生的，否則使用者會手動編輯然後被覆蓋。"""
        env = _env()
        config = build_training_config(
            env, fit_training_params(env), data_dir=Path("/d"), output_dir=Path("/o")
        )
        assert "_generated_by" in config
        assert "請勿手動編輯" in config["_note"] or "覆蓋" in config["_note"]

    def test_config_is_json_serializable(self) -> None:
        env = _env()
        config = build_training_config(
            env, fit_training_params(env), data_dir=Path("/d"), output_dir=Path("/o")
        )
        assert json.loads(json.dumps(config, ensure_ascii=False))

    def test_write_config_reports_failure_without_raising(self, tmp_dir: Path) -> None:
        """
        寫檔失敗要回報而不是拋例外 —— 而且必須變成阻擋條件，
        否則啟動器會以為設定檔寫好了。
        """
        env = _env()
        params = fit_training_params(env)
        plan = plan_training(env, params, _ok_precheck())
        # 用一個不可能寫入的路徑（檔案當目錄用）
        blocker_file = tmp_dir / "not-a-dir"
        blocker_file.write_text("x", encoding="utf-8")
        ok = write_config(plan, {"a": 1}, blocker_file / "config.json")
        assert not ok
        assert not plan.config_written
        assert any("設定檔" in b for b in plan.blockers)


class TestRendering:
    def test_shows_fitted_parameters(self) -> None:
        env = _env()
        params = fit_training_params(env)
        text = render_plan(plan_training(env, params, _ok_precheck()), params)
        assert str(params.batch_size) in text
        assert "等效" in text

    def test_lists_every_blocker(self) -> None:
        env = _env(has_gpu=False)
        params = fit_training_params(env)
        plan = plan_training(env, params, _failing_precheck())
        text = render_plan(plan, params)
        assert str(len(plan.blockers)) in text
        for blocker in plan.blockers:
            assert blocker[:20] in text

    def test_ready_message_is_clear(self, tmp_dir: Path, monkeypatch) -> None:
        """
        「就緒」訊息必須明確 —— 但本機必然缺訓練依賴（pytorch_lightning /
        librosa / ffmpeg），所以要把依賴檢查隔離掉才測得到那條路徑。

        這也說明一件事：**依賴檢查本身必須有效**，否則這條測試會意外通過，
        而使用者會在啟動訓練時才撞到 ImportError。
        """
        import train as train_mod

        monkeypatch.setattr(train_mod, "check_training_deps", lambda: [])
        monkeypatch.setenv("VOS_REPO_ROOT", str(tmp_dir / "repo"))
        # 也要提供 GPT-SoVITS，否則會卡在「找不到 GPT-SoVITS」——
        # 本機確實沒有，所以必須在沙箱內造一個假的
        monkeypatch.setenv(
            "VOS_GPT_SOVITS_DIR", str(_fake_gpt_sovits(tmp_dir / "fake-gsv"))
        )

        env = _env(has_gpu=True)
        params = fit_training_params(env)
        plan = plan_training(env, params, _ok_precheck())
        assert plan.ready, f"隔離依賴並提供 GPT-SoVITS 後應為就緒，仍有：{plan.blockers}"

        text = render_plan(plan, params)
        assert "可以啟動訓練" in text

    def test_blocked_message_explains_rationale(self) -> None:
        """阻擋訊息要說明「為什麼在這裡擋」，否則只是又多一個紅字。"""
        env = _env(has_gpu=False)
        params = fit_training_params(env)
        text = render_plan(plan_training(env, params, _failing_precheck()), params)
        assert "數小時" in text or "10 秒" in text


class TestEndToEnd:
    def test_full_chain_on_this_machine_produces_plan(self) -> None:
        """
        在本機實跑整條鏈路：env -> trainfit -> precheck -> plan。
        這台機器必然缺件（無 GPU、無 GPT-SoVITS、無資料集），
        所以重點是「能給出完整的阻擋清單」，而不是「通過」。
        """
        from hwinfo import detect_torch_environment

        env = detect_torch_environment()
        params = fit_training_params(env)
        pre = run_precheck(env, params, checkpoint_dir=REPO)
        plan = plan_training(env, params, pre)
        text = render_plan(plan, params)

        assert "訓練參數" in text
        assert plan.blockers, "本機必然有阻擋條件（無 GPU）"
        assert any("GPU" in b for b in plan.blockers)
