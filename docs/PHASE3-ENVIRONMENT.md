# Phase 3 — 跨平台環境配置與參數最佳化

> **一句話總結**：先弄清楚這台機器能不能訓練，再依 VRAM 決定參數 ——
> 兩件事都不需要素材，所以**可以在整理資料之前就先做完**。

---

## 0. 先跑這兩行

```bash
# 我有什麼裝置？該用什麼參數？
python scripts/phase1_prep.py env

# 這個環境真的能跑完一次訓練嗎？（實際配置記憶體、跑一次 backward）
python scripts/phase1_prep.py precheck
```

`env` 回答「我有什麼裝置」，`precheck` 回答「**這個環境真的能跑完一次訓練嗎**」。
後者用合成張量把問題提前引爆 —— 那些問題通常要等訓練進入第一個 backward
pass 才發生，而那時你已經等了很久。

```
  [OK] PyTorch 可用  -- 2.13.0+cpu
  [X] GPU 可用性  -- 沒有可用的 GPU
      -> 純 CPU 訓練在實務上不可行；見 docs/PHASE3-ENVIRONMENT.md §4
  [OK] 反向傳播  -- 前向 + 反向 + 梯度檢查正常
  [OK] optimizer  -- AdamW 可建立並執行 step
  [OK] 精度 fp32  -- CPU 訓練固定用 fp32
  [X] 系統記憶體  -- 可用僅 3.8 GB（總計 31.4 GB）
      -> 有大量記憶體被其他程式佔用。CPU 訓練吃系統 RAM，
         請先關閉瀏覽器／其他服務並確認可用量 ≥ 16 GB
  [OK] checkpoint 空間  -- 506.8 GB 可用
```

`precheck` 有 fail 時回傳 exit 4，方便腳本化擋在訓練之前。

### 為什麼「可用記憶體」比「總記憶體」重要

上例的機器有 **31.4 GB 總記憶體**，看起來很充裕 —— 但**可用只剩 3.8 GB**。
只看總量會誤判為沒問題，實際上訓練幾乎一定會失敗。`precheck` 兩者都看。

---

## 1. 最重要的一件事：區分「沒有 GPU」與「裝錯 PyTorch」

這兩種情況的症狀一模一樣（`torch.cuda.is_available()` 回 `False`），但解法完全不同：

| 情況 | 判斷方式 | 解法 |
|------|----------|------|
| **真的沒有 NVIDIA GPU** | 系統資訊顯示只有內顯 | 換機器、用雲端 GPU，或接受 CPU 訓練 |
| **裝到 CPU-only 的 PyTorch** | 版本字串帶 `+cpu`（例如 `2.13.0+cpu`） | 改裝對應 CUDA 版本的 PyTorch |

**為什麼這個區分這麼重要**：CPU-only 的 build **永遠**看不到 GPU，就算你插了一張 4090 也一樣。
如果誤判成「我沒有 GPU」而去買卡或租雲端，就白花錢了。

`env` 會直接告訴你是哪一種：

```
 PyTorch    2.13.0+cpu
 CUDA 編譯   （無 —— 這是 CPU-only 建置）

 注意：
   * PyTorch 是 CPU-only 建置（2.13.0+cpu）—— 即使這台機器有 NVIDIA GPU，
     這個 build 也**永遠**看不到它。
```

### CPU-only build 要怎麼修

先確認機器上真的有 NVIDIA 卡（`nvidia-smi` 有輸出），然後**移除並重裝** PyTorch：

```bash
# 1) 先看驅動支援到哪個 CUDA 版本
nvidia-smi        # 右上角的 "CUDA Version: 12.x" 就是上限

# 2) 移除 CPU-only build
pip uninstall -y torch torchvision torchaudio

# 3) 裝對應版本的 CUDA build（以 CUDA 12.1 為例）
#    Windows / Linux
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#    macOS（Apple Silicon 用 MPS，不需要 CUDA）
pip install torch torchvision torchaudio
```

> ⚠️ **一定要加 `--index-url`**。不加的話 pip 會從 PyPI 裝到預設版本，
> 而那通常又是 CPU-only —— 你會以為自己裝了 GPU 版，其實沒有。
> 裝完務必再跑一次 `env` 確認 `cuda.is_available()` 變成 `True`。

---

## 2. 三種加速路徑

| 平台 | 後端 | 啟用條件 | 備註 |
|------|------|----------|------|
| Windows / Linux + NVIDIA | **CUDA** | 驅動 + CUDA build 的 PyTorch | 首選，速度最快 |
| Apple Silicon (M1~M4) | **MPS** | macOS 12.3+ | 統一記憶體是優勢也是限制；部分 op 不支援會回退 CPU |
| 任何平台 | **CPU** | 永遠可用 | 可行但**極慢**，見下方 |

`env` 會自動偵測並選出最適合的裝置，你不需要手動指定。

---

## 3. VRAM 自適應參數矩陣

核心原則：**等效 batch size 比實際 batch size 重要。**

```
等效 batch size = batch_size × gradient_accumulation_steps
```

所有 VRAM 等級都朝同一個目標（8）擬合，這樣**換一張卡不會改變學習動態**。
VRAM 不夠時的正確做法是「縮小 `batch_size`、提高 `grad_accum`」，
而不是降低等效 batch size —— 代價是每步稍慢，但訓練品質一致。

| VRAM | 典型顯卡 | batch | accum | 精度 | ckpt | 快取 |
|------|----------|------:|------:|------|:----:|:----:|
| ≥ 22 GB | RTX 3090 / 4090 / A5000 | 8 | 1 | bf16 | ✗ | ✓ |
| ≥ 14 GB | RTX 4060 Ti 16G / 4080 | 6 | 2 | bf16 | ✗ | ✓ |
| ≥ 10 GB | RTX 3060 12G / 4070 | 4 | 2 | bf16 | ✗ | ✓ |
| ≥ 7 GB | RTX 3060 Ti / 3070 / 4060 | 3 | 3 | fp16 | ✓ | ✗ |
| ≥ 5 GB | RTX 2060 / 1060 | 2 | 4 | fp16 | ✓ | ✗ |
| ≥ 3 GB | GTX 1650 | 1 | 8 | fp16 | ✓ | ✗ |
| 無 GPU | CPU | 1 | 8 | fp32 | ✓ | ✗ |

（`ckpt` = gradient checkpointing，用計算時間換記憶體；`快取` = 資料集預先載入 VRAM）

### 為什麼大 VRAM 用 bf16 而小 VRAM 用 fp16

`bf16` 的數值範圍與 fp32 相同，訓練穩定、不容易出現 NaN，但需要較新的硬體
（Ampere 以後）。`fp16` 範圍較窄，需要 loss scaling，在小顯卡上偶爾不穩，
但對舊硬體支援較好。這不是效能取捨，是**數值穩定性**取捨。

### 遇到 OOM 的處理順序

1. 把 `batch_size` 減半、`grad_accum` 加倍（維持等效 batch）
2. 關閉資料集快取
3. 啟用 gradient checkpointing
4. 降低取樣率或縮短訓練片段長度
5. 換更小的模型（SoVITS / GPT 分開訓練，不要同時載入）

> ⚠️ **不要**把等效 batch size 直接砍半來解決 OOM —— 那會改變學習動態，
> 讓你的訓練結果無法與其他設定比較。

---

## 4. 純 CPU 訓練的現實

**如果你的 `env` 結論是「只能用 CPU」，請先讀完這一節再決定要不要投入時間。**

GPT-SoVITS 的微調在純 CPU 上：

- **時間**：數小時到數天（依資料量與步數），而且沒有混合精度加速
- **可行性**：技術上可跑完，但除錯週期會被拉長到不實用 ——
  每次調整參數要等好幾小時才知道結果
- **記憶體**：CPU 訓練吃系統 RAM，16 GB 以下很容易被迫縮小模型或資料量

### 三個實際的替代方案

| 方案 | 成本 | 適合情況 |
|------|------|----------|
| **Google Colab**（免費 T4 16 GB） | 免費 | 先驗證流程、跑小規模微調 |
| **Colab Pro / RunPod / Vast.ai** | 每小時數十元 | 正式訓練，可挑 24 GB 卡 |
| **本機補一張 GPU** | 一次性 | 長期反覆實驗（推薦 RTX 3060 12G 起） |

> **建議順序**：先用 Colab 免費額度把「Phase 1 → Phase 4 → Phase 5」整條鏈路
> 跑通一次，確認素材與流程都沒問題，**再**決定要不要投資硬體。
> 反過來做（先買卡）風險很高 —— 你可能會發現素材本身需要重挖。

---

## 5. 環境配置檢查清單

在進入 Phase 4（ASR 打標）之前確認：

- [ ] `python scripts/phase1_prep.py env` 的結論符合你的預期
- [ ] 若顯示 `+cpu` 而機器有 NVIDIA 卡 → 依 §1 重裝 PyTorch
- [ ] 確認 `batch_size × grad_accum` 等於 8（或你刻意選的其他值）
- [ ] 若 VRAM < 8 GB：先跑一次小規模訓練確認不會 OOM，再全量訓練
- [ ] 已讀 §4 並接受訓練時間，或已選定雲端方案

---

---

## 7. 訓練啟動器（`train`）

`env` 算參數、`precheck` 找問題 —— 但那些結果若只印在畫面上，就變成
「你必須記得先跑，然後手動把數字抄進設定檔」。`train` 把三件事接起來：

```
precheck（自動擋門）→ 擬合參數寫成設定檔 → 啟動訓練
```

```bash
# 規劃模式：檢查環境、寫出設定檔，但不啟動（預設）
python scripts/phase1_prep.py train

# 真的啟動訓練
python scripts/phase1_prep.py train --yes
```

### 三個刻意的設計

| 設計 | 為什麼 |
|------|--------|
| **precheck 是強制擋門，不是建議** | 訓練失敗的代價是數小時，擋在門口的代價是 10 秒。有 `fail` 就拒絕啟動。 |
| **參數自動寫進設定檔** | 人抄數字一定會出錯，而且出錯時**不會報錯** —— 訓練只是跑得比較差，你無從察覺。 |
| **缺件時給出精確的下一步** | 不是「失敗」，而是「`git clone …`」或「`pip install …`」。 |

### 輸出範例（典型的第一次使用狀態）

```
 訓練參數（依本機 VRAM 擬合）：
   batch 1 × accum 8 = 等效 8　|　精度 fp32
   gradient checkpointing True　|　資料集快取 False　|　workers 0

 阻擋條件（5 項）：
   [X] [環境] GPU 可用性：沒有可用的 GPU → 純 CPU 訓練在實務上不可行
   [X] [環境] 系統記憶體：可用僅 5.1 GB（總計 31.4 GB） → 請關閉其他程式
   [X] [缺件] 找不到 GPT-SoVITS → git clone https://github.com/RVC-Boss/GPT-SoVITS ...
   [X] [缺件] 缺少訓練依賴：pytorch_lightning, librosa, ffmpeg
              → pip install pytorch-lightning librosa ffmpeg-python
   [X] [缺件] 資料集目錄不存在 → 先跑完 Phase 1 / Phase 2 產出乾淨素材

 為什麼要在啟動前擋下來：訓練失敗的代價是數小時，而擋在門口的代價是 10 秒。
```

`train` 有阻擋條件時回傳 exit 4；參數設定檔仍會寫出（在
`data/_reports/train/vos_training_config.json`），因為就算還不能訓練，
你也需要看到擬合結果。

> **注意**：訓練會執行數小時，所以 `train` **刻意不預設啟動** ——
> 要加 `--yes` 才會真的開始。訓練以獨立行程執行，本工具結束不會中斷它。

---

## 8. 相關文件

- [Phase 1 操作手冊](PHASE1-INGEST.md) — 素材處理（**不依賴 GPU**，可先做完）
- [格式決策手冊](FORMAT-PLAYBOOK.md) — 音訊格式排查
- [PyTorch 官方安裝指引](https://pytorch.org/get-started/locally/)
- [GPT-SoVITS 官方倉庫](https://github.com/RVC-Boss/GPT-SoVITS)