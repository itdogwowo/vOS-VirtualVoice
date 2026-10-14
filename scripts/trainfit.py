"""
vOS-VirtualVoice - Phase 3：VRAM 自適應訓練參數矩陣

為什麼需要這個模組
------------------
GPT-SoVITS 的預設訓練參數是為「24 GB 級 GPU」寫的。直接拿去 8 GB 的卡上跑
就是 OOM；反過來在 24 GB 上保守設定則是浪費（訓練時間多好幾倍）。

這個模組把「可用 VRAM」對應到一組**實際能跑完**的參數：

    batch_size            每次前向/反向的樣本數
    grad_accum            梯度累積步數（用時間換等效 batch size）
    precision             fp16 / bf16 / fp32
    grad_checkpointing    用計算時間換記憶體
    num_workers           dataloader 平行度（受 CPU 核心數限制）
    cache 策略            是否把整個資料集預先載入 VRAM

關鍵洞察：**等效 batch size 比實際 batch size 重要**。
batch_size × grad_accum 才是影響梯度品質的量。所以 VRAM 不夠時的
正確做法是「縮小 batch_size、提高 grad_accum」，而不是降低等效 batch size。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hwinfo import EnvReport

#: 等效 batch size 的目標值。
#:
#: 這是「batch_size × grad_accum」應該達到的量。所有 VRAM 等級都用同一個
#: 目標，這樣不同硬體訓練出來的模型品質才具可比性 —— 換一張卡不該改變
#: 學習動態。8 是 GPT-SoVITS 社群常用的值。
TARGET_EFFECTIVE_BATCH = 8

#: 各 VRAM 等級的參數。鍵是「至少需要幾 GB」。
#: 由高到低比對，第一個符合的就是建議值。
VRAM_PROFILES: list[dict] = [
    {
        "min_vram_gb": 22,
        "label": "24 GB 級（RTX 3090 / 4090 / A5000）",
        "batch_size": 8,
        "precision": "bf16",
        "grad_checkpointing": False,
        "cache_dataset": True,
        "note": "可以全速訓練，不需要任何記憶體取捨",
    },
    {
        "min_vram_gb": 14,
        "label": "16 GB 級（RTX 4060 Ti 16G / 4080 / A4000）",
        "batch_size": 6,
        "precision": "bf16",
        "grad_checkpointing": False,
        "cache_dataset": True,
        "note": "舒適區間；若同時跑其他程式可降到 batch 4",
    },
    {
        "min_vram_gb": 10,
        "label": "12 GB 級（RTX 3060 12G / 4070 / 3080 Ti）",
        "batch_size": 4,
        "precision": "bf16",
        "grad_checkpointing": False,
        "cache_dataset": True,
        "note": "最常見的消費級配置，訓練時間與品質平衡良好",
    },
    {
        "min_vram_gb": 7,
        "label": "8 GB 級（RTX 3060 Ti / 3070 / 4060）",
        "batch_size": 3,
        "precision": "fp16",
        "grad_checkpointing": True,
        "cache_dataset": False,
        "note": "需要 gradient checkpointing；關閉資料集快取以省 VRAM",
    },
    {
        "min_vram_gb": 5,
        "label": "6 GB 級（RTX 2060 / 1060）",
        "batch_size": 2,
        "precision": "fp16",
        "grad_checkpointing": True,
        "cache_dataset": False,
        "note": "上限；遇到 OOM 就把 batch_size 降到 1 並提高 grad_accum",
    },
    {
        "min_vram_gb": 3,
        "label": "4 GB 級（GTX 1650 等）",
        "batch_size": 1,
        "precision": "fp16",
        "grad_checkpointing": True,
        "cache_dataset": False,
        "note": "勉強可跑，但訓練時間會非常長；建議改用雲端 GPU",
    },
    {
        "min_vram_gb": 0,
        "label": "CPU（無 GPU）",
        "batch_size": 1,
        "precision": "fp32",
        "grad_checkpointing": True,
        "cache_dataset": False,
        "note": "不建議 —— 見下方說明",
    },
]


@dataclass
class TrainingParams:
    """一組建議的訓練參數。"""

    device: str                 # "cuda" | "mps" | "cpu"
    device_name: str
    vram_gb: float
    label: str
    batch_size: int
    grad_accum: int
    precision: str
    grad_checkpointing: bool
    cache_dataset: bool
    num_workers: int
    notes: list[str] = field(default_factory=list)

    @property
    def effective_batch(self) -> int:
        """等效 batch size = batch_size × grad_accum。"""
        return self.batch_size * self.grad_accum


def _pick_profile(vram_gb: float) -> dict:
    for profile in VRAM_PROFILES:
        if vram_gb >= profile["min_vram_gb"]:
            return profile
    return VRAM_PROFILES[-1]


def _workers_for(cpu_count: int, device: str) -> int:
    """
    dataloader 的 worker 數。

    原則：留至少一個核心給主行程；CPU 訓練時 worker 反而會搶資源，
    所以固定給小值。
    """
    if device == "cpu":
        return 0  # CPU 訓練時多 worker 只會互相搶核心
    if cpu_count <= 2:
        return 0
    return max(1, min(8, cpu_count - 2))


def fit_training_params(report: EnvReport) -> TrainingParams:
    """
    依環境報告產生一組實際能跑完的訓練參數。

    核心邏輯：**先決定等效 batch size，再依 VRAM 拆成 batch_size × grad_accum。**
    這樣不同硬體上的學習動態才一致。
    """
    device = report.primary
    notes: list[str] = []

    if device.kind == "cuda":
        vram = device.total_vram_gb or device.free_vram_gb or 0.0
        if vram <= 0:
            # 極少數情況 PyTorch 回報不到總量，用型號反推
            from hwinfo import _guess_vram_from_name

            vram = _guess_vram_from_name(device.name) or 8.0
            notes.append(f"PyTorch 未回報 VRAM 總量，依型號推測為 {vram:.0f} GB")
        profile = _pick_profile(vram)
        kind = "cuda"
    elif device.kind == "mps":
        # Apple Silicon 的統一記憶體：保守取 8 GB 可用給模型
        vram = 8.0
        profile = _pick_profile(vram)
        kind = "mps"
        notes.append(
            "Apple Silicon 使用統一記憶體；上表以保留 8 GB 給模型估算。"
            "MPS 對某些運算的支援仍不完整，遇到不支援的 op 會回退 CPU 而變慢。"
        )
    else:
        vram = 0.0
        profile = VRAM_PROFILES[-1]
        kind = "cpu"

    batch_size = int(profile["batch_size"])
    precision = str(profile["precision"])
    if kind == "cpu":
        precision = "fp32"  # CPU 上 fp16 通常更慢（缺少原生支援）

    # 依 VRAM 反推 grad_accum。
    #
    # ⚠ 必須用 ceil 而不是 round：
    #   batch 6 時 round(8/6) = round(1.33) = 1 → 等效 batch 只有 6 < 目標 8。
    #   梯度品質會低於其他 VRAM 等級，等於「換一張 16 GB 的卡訓練結果就不同」。
    #   向上取整（ceil(8/6) = 2 → 等效 12）寧可略高於目標，也不要低於 ——
    #   略高只是慢一點，低於則是品質妥協。
    import math

    grad_accum = max(1, math.ceil(TARGET_EFFECTIVE_BATCH / max(1, batch_size)))

    if kind == "cpu":
        notes.append(
            "純 CPU 訓練：GPT-SoVITS 微調需要數小時到數天，且不支援混合精度。"
        )
        notes.append(
            "建議改用雲端 GPU（Colab / RunPod / Vast.ai），或先解決 GPU 可用性問題。"
        )
    elif device.kind == "cuda" and report.torch_is_cpu_only_build:
        notes.append(
            "警告：目前是 CPU-only 的 PyTorch build，即使有 GPU 也用不到。"
            "必須先改裝對應 CUDA 版本的 PyTorch。"
        )

    if grad_accum > 1:
        notes.append(
            f"VRAM 不足以直接跑 batch {TARGET_EFFECTIVE_BATCH}，"
            f"改用 batch {batch_size} × 累積 {grad_accum} 步 —— "
            f"等效 batch 仍為 {batch_size * grad_accum}，學習動態與大 VRAM 一致，"
            f"代價是每步稍慢。"
        )

    return TrainingParams(
        device=kind,
        device_name=device.name,
        vram_gb=round(vram, 2),
        label=str(profile["label"]),
        batch_size=batch_size,
        grad_accum=grad_accum,
        precision=precision,
        grad_checkpointing=bool(profile["grad_checkpointing"]),
        cache_dataset=bool(profile["cache_dataset"]),
        num_workers=_workers_for(report.cpu_count, kind),
        notes=notes + ([str(profile["note"])] if profile.get("note") else []),
    )


def render_params(p: TrainingParams) -> str:
    """人類可讀的參數表（含可直接複製的設定片段）。"""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" Phase 3 建議訓練參數（依 VRAM 自動擬合）")
    lines.append("=" * 72)
    lines.append(f" 裝置       {p.device.upper()}  {p.device_name}")
    lines.append(f" VRAM       {p.vram_gb:.1f} GB（{p.label}）")
    lines.append("")
    lines.append(f" batch_size           {p.batch_size}")
    lines.append(f" gradient_accum       {p.grad_accum}")
    lines.append(f" 等效 batch size      {p.effective_batch}（目標 {TARGET_EFFECTIVE_BATCH}）")
    lines.append(f" 精度                 {p.precision}")
    lines.append(f" gradient_checkpoint  {p.grad_checkpointing}")
    lines.append(f" 資料集快取           {p.cache_dataset}")
    lines.append(f" dataloader workers   {p.num_workers}")

    if p.notes:
        lines.append("\n 說明：")
        for note in p.notes:
            lines.append(f"   - {note}")

    lines.append("\n 可直接貼進訓練設定的片段：")
    lines.append(
        f'   {{ "batch_size": {p.batch_size}, "gradient_accumulation_steps": {p.grad_accum},'
    )
    lines.append(
        f'     "precision": "{p.precision}",'
        f' "gradient_checkpointing": {str(p.grad_checkpointing).lower()},'
    )
    lines.append(
        f'     "cache_dataset": {str(p.cache_dataset).lower()},'
        f' "num_workers": {p.num_workers} }}'
    )
    return "\n".join(lines)


__all__ = [
    "TARGET_EFFECTIVE_BATCH",
    "VRAM_PROFILES",
    "TrainingParams",
    "fit_training_params",
    "render_params",
]