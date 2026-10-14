"""
vOS-VirtualVoice - Phase 3：訓練啟動器

為什麼需要這個模組
------------------
`env` 算出參數、`precheck` 找出問題 —— 但那些結果若只印在畫面上，
就會變成「你必須記得先跑，然後手動把數字抄進設定檔」。

這個模組把三件事接起來：

    precheck（自動擋門）-> 擬合參數寫成設定檔 -> 啟動訓練

三個刻意的設計：

1. **precheck 是強制擋門，不是建議。**
   有 fail 就拒絕啟動。訓練失敗的代價是數小時，擋在門口的代價是 10 秒。

2. **參數自動寫進設定檔，不靠人手抄。**
   人抄數字一定會出錯，而且出錯時不會報錯 —— 訓練只是跑得比較差。

3. **缺件時給出精確的下一步，不是一句「失敗」。**
   GPT-SoVITS 沒裝、資料集不存在、標註檔不存在 —— 每一種都告訴你
   該跑哪個指令或去哪個文件。

本模組的無 GPU／無 GPT-SoVITS 路徑都會正常回報，因為那正是**第一次使用時
必然的狀態** —— 一個只能在萬事俱備時運作的啟動器沒有價值。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from hwinfo import EnvReport
from precheck import PrecheckReport
from trainfit import TrainingParams

#: GPT-SoVITS 的識別檔（相對於 repo 根目錄）。用它判斷「這裡是不是 GPT-SoVITS」。
GPT_SOVITS_MARKERS: tuple[str, ...] = (
    "s2_train.py",
    "s1_train.py",
    "GPT_SoVITS",
    "webui.py",
    "api.py",
)

#: GPT-SoVITS 自己的依賴中，本專案 requirements 沒有的部分。
#: 用來在缺件時給出精確的 pip 指令，而不是讓使用者看 ImportError 猜。
GPT_SOVITS_EXTRA_DEPS: tuple[tuple[str, str], ...] = (
    ("pytorch_lightning", "pytorch-lightning"),
    ("librosa", "librosa"),
    ("transformers", "transformers"),
    ("ffmpeg", "ffmpeg-python"),
)

#: 訓練所需的相對路徑（相對於 GPT-SoVITS 根目錄）。
#: 這些是 GPT-SoVITS 官方流程的產物位置。
REQUIRED_TRAINING_INPUTS: tuple[tuple[str, str], ...] = (
    ("logs", "訓練輸出目錄（GPT-SoVITS 會寫入這裡）"),
)


@dataclass
class TrainPlan:
    """一次訓練啟動的完整計畫。"""

    gpt_sovits_dir: Path | None = None
    missing_deps: list[tuple[str, str]] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    config_path: Path | None = None
    config_written: bool = False
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers


def locate_gpt_sovits(cfg=None) -> tuple[Path | None, str]:
    """
    找出 GPT-SoVITS 的安裝位置，回傳 (路徑, 來源說明)。

    搜尋順序：
      1. 設定檔的 training.gpt_sovits_dir
      2. 環境變數 VOS_GPT_SOVITS_DIR
      3. repo 內的 GPT-SoVITS/ 或 gpt-sovits/
      4. 家目錄下的 GPT-SoVITS/
    """
    from config import repo_root

    candidates: list[tuple[Path, str]] = []

    hint = None
    if cfg is not None:
        hint = getattr(cfg, "training_gpt_sovits_dir", None)
    if hint and str(hint).lower() != "auto":
        p = Path(hint)
        if not p.is_absolute():
            p = repo_root() / p
        candidates.append((p, f"設定檔 training.gpt_sovits_dir（{hint}）"))

    env_val = os.environ.get("VOS_GPT_SOVITS_DIR")
    if env_val:
        candidates.append((Path(env_val), "環境變數 VOS_GPT_SOVITS_DIR"))

    root = repo_root()
    candidates.append((root / "GPT-SoVITS", "repo 內 GPT-SoVITS/"))
    candidates.append((root / "gpt-sovits", "repo 內 gpt-sovits/"))
    candidates.append((Path.home() / "GPT-SoVITS", "家目錄 GPT-SoVITS/"))

    for path, source in candidates:
        if _looks_like_gpt_sovits(path):
            return path, source
    return None, "未找到"


def _looks_like_gpt_sovits(path: Path) -> bool:
    """路徑存在而且看起來像 GPT-SoVITS（有任一識別檔）。"""
    try:
        if not path.is_dir():
            return False
        return any((path / marker).exists() for marker in GPT_SOVITS_MARKERS)
    except OSError:
        return False


def check_training_deps() -> list[tuple[str, str]]:
    """回傳缺少的訓練依賴 [(import 名稱, pip 名稱)]。"""
    missing: list[tuple[str, str]] = []
    for module, package in GPT_SOVITS_EXTRA_DEPS:
        if module == "ffmpeg":
            if shutil.which("ffmpeg") is None:
                missing.append((module, package))
            continue
        try:
            __import__(module)
        except Exception:
            missing.append((module, package))
    return missing


def build_training_config(
    env: EnvReport,
    params: TrainingParams,
    *,
    data_dir: Path,
    output_dir: Path,
) -> dict:
    """
    產生 GPT-SoVITS 訓練設定（本專案的參數 + 路徑）。

    刻意不用 GPT-SoVITS 自己的預設檔 —— 那份是為 24 GB 級 GPU 寫的。
    這裡的每個數字都來自 `trainfit` 對本機 VRAM 的擬合。
    """
    return {
        "_generated_by": "vOS-VirtualVoice Phase 3 train launcher",
        "_note": (
            "本檔由 train 子命令自動產生，參數依本機 VRAM 擬合。"
            "請勿手動編輯 —— 重跑 train 會覆蓋它。"
        ),
        "device": params.device,
        "device_name": params.device_name,
        "vram_gb": params.vram_gb,
        "training": {
            "batch_size": params.batch_size,
            "gradient_accumulation_steps": params.grad_accum,
            "effective_batch_size": params.effective_batch,
            "precision": params.precision,
            "gradient_checkpointing": params.grad_checkpointing,
            "cache_dataset": params.cache_dataset,
            "num_workers": params.num_workers,
        },
        "paths": {
            "dataset_dir": str(data_dir),
            "output_dir": str(output_dir),
        },
        "environment": {
            "torch": env.torch_version,
            "torch_cuda": env.torch_cuda_build,
            "cpu_count": env.cpu_count,
            "os": env.os_name,
        },
    }


def plan_training(
    env: EnvReport,
    params: TrainingParams,
    precheck_report: PrecheckReport,
    *,
    cfg=None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> TrainPlan:
    """
    規劃一次訓練啟動，收集所有阻擋條件。

    刻意**不**在這裡啟動訓練 —— 規劃與執行分開，才能先讓使用者看到
    完整狀況（缺什麼、參數是什麼），而不是啟動後才失敗。
    """
    plan = TrainPlan()

    # ---- 1. precheck 是強制擋門 ----
    # 訓練失敗的代價是數小時；擋在門口的代價是 10 秒。
    if not precheck_report.passed:
        for check in precheck_report.failures:
            plan.blockers.append(
                f"[環境] {check.name}：{check.detail}"
                + (f" → {check.hint}" if check.hint else "")
            )

    # ---- 2. GPT-SoVITS 本體 ----
    gpt_dir, source = locate_gpt_sovits(cfg)
    plan.gpt_sovits_dir = gpt_dir
    if gpt_dir is None:
        plan.blockers.append(
            "[缺件] 找不到 GPT-SoVITS → "
            "git clone https://github.com/RVC-Boss/GPT-SoVITS 到專案根目錄，"
            "或在 configs/pipeline.local.json 設定 training.gpt_sovits_dir"
        )
    else:
        plan.notes.append(f"GPT-SoVITS：{gpt_dir}（來自 {source}）")

    # ---- 3. 訓練依賴 ----
    plan.missing_deps = check_training_deps()
    if plan.missing_deps:
        pkgs = " ".join(pkg for _mod, pkg in plan.missing_deps)
        plan.blockers.append(
            f"[缺件] 缺少訓練依賴：{', '.join(mod for mod, _ in plan.missing_deps)}"
            f" → pip install {pkgs}"
        )

    # ---- 4. 資料集與標註 ----
    if data_dir is not None:
        if not data_dir.is_dir():
            plan.blockers.append(
                f"[缺件] 資料集目錄不存在：{data_dir} → "
                f"先跑完 Phase 1 / Phase 2 產出乾淨素材"
            )
        else:
            wavs = list(data_dir.rglob("*.wav"))
            if not wavs:
                plan.blockers.append(
                    f"[缺件] {data_dir} 內沒有 .wav → 先跑 convert"
                )
            else:
                plan.notes.append(f"資料集：{len(wavs)} 個 wav")

    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            plan.blockers.append(f"[環境] 無法建立輸出目錄 {output_dir}：{exc}")

    return plan


def render_plan(plan: TrainPlan, params: TrainingParams) -> str:
    """人類可讀的訓練計畫。"""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" Phase 3 訓練啟動計畫")
    lines.append("=" * 72)

    lines.append(" 訓練參數（依本機 VRAM 擬合）：")
    lines.append(
        f"   batch {params.batch_size} × accum {params.grad_accum}"
        f" = 等效 {params.effective_batch}　|　精度 {params.precision}"
    )
    lines.append(
        f"   gradient checkpointing {params.grad_checkpointing}"
        f"　|　資料集快取 {params.cache_dataset}"
        f"　|　workers {params.num_workers}"
    )
    lines.append(f"   裝置 {params.device.upper()}　{params.device_name}")

    if plan.notes:
        lines.append("\n 已就緒：")
        for note in plan.notes:
            lines.append(f"   [OK] {note}")

    if plan.blockers:
        lines.append(f"\n 阻擋條件（{len(plan.blockers)} 項）：")
        for blocker in plan.blockers:
            lines.append(f"   [X] {blocker}")
        lines.append(
            "\n 為什麼要在啟動前擋下來：訓練失敗的代價是數小時，"
            "而擋在門口的代價是 10 秒。"
        )
    else:
        lines.append("\n 全部就緒 —— 可以啟動訓練。")

    if plan.config_path:
        state = "已寫出" if plan.config_written else "尚未寫出"
        lines.append(f"\n 訓練設定檔：{plan.config_path}（{state}）")
    return "\n".join(lines)


def write_config(plan: TrainPlan, config: dict, path: Path) -> bool:
    """寫出訓練設定檔。回傳是否成功。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        plan.config_path = path
        plan.config_written = True
        return True
    except OSError as exc:
        plan.blockers.append(f"[環境] 無法寫出設定檔 {path}：{exc}")
        return False


def launch_training(gpt_dir: Path, config_path: Path, *, timeout: int = 86400):
    """
    啟動 GPT-SoVITS 訓練。

    刻意用子程序而不是 import 它的模組 —— 訓練是長時間、獨立的工作，
    不該與本工具的行程綁在一起（本工具結束不該殺掉訓練）。
    """
    entry = None
    for candidate in ("s2_train.py", "s1_train.py"):
        if (gpt_dir / candidate).is_file():
            entry = gpt_dir / candidate
            break
    if entry is None:
        raise RuntimeError(
            f"在 {gpt_dir} 找不到 s1_train.py 或 s2_train.py —— "
            f"這可能不是 GPT-SoVITS 的根目錄"
        )

    return subprocess.run(
        [sys.executable, str(entry), "--config", str(config_path)],
        cwd=str(gpt_dir),
        check=False,
    )


__all__ = [
    "GPT_SOVITS_EXTRA_DEPS",
    "GPT_SOVITS_MARKERS",
    "TrainPlan",
    "build_training_config",
    "check_training_deps",
    "launch_training",
    "locate_gpt_sovits",
    "plan_training",
    "render_plan",
    "write_config",
]