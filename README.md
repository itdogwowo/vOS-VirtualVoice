# vOS-VirtualVoice

跨平台（Windows / macOS / Linux）遊戲語音 → **GPT-SoVITS** 長篇有聲書生產線。

核心路線：

> 遊戲日語配音素材（純淨、情感豐富）→ 微調 GPT-SoVITS → **日語音色 × 中文情感輸出** → 長篇有聲書

---

## 五階段架構

| Phase | 名稱 | 目標 | 狀態 |
|:-----:|------|------|:----:|
| **1** | 數據解包與格式標準化 | `.wem/.acb/.fsb/.ogg` → `24bit/48kHz` WAV，時長篩選 | ✅ 工具完成，待你的素材 |
| **2** | 數據清洗與音訊增強 | UVR5 人聲淨化、`-23 LUFS` 響度標準化 | ⏳ |
| **3** | 跨平台環境配置與參數最佳化 | CUDA / MPS / CPU 自動適配、VRAM 參數矩陣 | ⏳ |
| **4** | ASR 自動打標籤與人工校對 | SenseVoice / Faster-Whisper 日語打標 + Labeling WebUI | ⏳ |
| **5** | 跨語言有聲書推理與情感調教 | 日語參考音訊 → 中文文本，長文本切句與情感控制 | ⏳ |

**一次只做一個階段。** 目前焦點：**Phase 1**。

Phase 1 的六個子命令（`scan` / `unpack` / `convert` / `audition` / `tune` / `report`）
都已完成並有 149 個測試守著，包含**全鏈路整合測試**（容器 → 解包 → 標準化 →
篩選 → 試聽 → 報告）。剩下的是你的真實素材 —— 先跑 `scan` 確認格式。

---

## 快速開始（Phase 1）

```bash
# 1) 安裝依賴
pip install -r requirements.txt

# 2) 先確認工具鏈在這台機器上能不能動（不需要任何素材，幾秒鐘）
#    用合成音訊跑一次真正的流程：標準化 → 篩選 → 試聽 → 報告
python scripts/phase1_prep.py selftest

# 3) 檢查環境（Python 依賴 / ffmpeg / vgmstream / 資料夾）
python scripts/phase1_prep.py doctor

# 3b) 偵測運算環境並依 VRAM 擬合訓練參數（Phase 3，不需要素材）
#     會區分「沒有 GPU」與「裝到 CPU-only 的 PyTorch」—— 兩者解法完全不同
python scripts/phase1_prep.py env

# 4) 把遊戲語音放進 data/00_raw/（可含子資料夾），然後嗅探真實格式
python scripts/phase1_prep.py scan

# 5) 解包遊戲容器（若 scan 顯示「需解包」才需要）
python scripts/phase1_prep.py unpack

# 6) 標準化：24bit/48kHz 單聲道 WAV + 1.5~15 秒時長篩選 + 品質關卡
python scripts/phase1_prep.py convert

# 7) 人工試聽驗收：抽樣 → 產生「試聽帶」與對照清單（不可省略的一步）
python scripts/phase1_prep.py audition --per-group 5

# 8) 不確定門檻該不該調？讓數據說話
python scripts/phase1_prep.py tune

# 9) 產生綜合報告（驗收清單 + 下一步指令，Phase 1 → Phase 2 的交接文件）
python scripts/phase1_prep.py report

# 或一條龍（scan → unpack → convert）
python scripts/phase1_prep.py all
```

### 多角色專案（建議這樣做）

如果素材來自多個角色，請放成 `data/00_raw/<角色名>/`，並在
`configs/pipeline.local.json` 開啟分組：

```json
{ "grouping": { "by": "folder" } }
```

產物就會各自進到 `data/02_wav24_48k/<角色名>/`，manifest 也會帶 `group` 欄位。
`convert` 結束時會多印一張逐角色素材量表：

```
 逐角色素材量（Phase 4 會依此分開打標籤、Phase 5 分開選參考音訊）：
   群組                       合格     淘汰     錯誤       分鐘
   凛                        12      1      0      0.8
   桜                         3      1      0      0.1
   ナレーション                    1      1      0      0.1

   ⚠ 以下群組素材 < 3 分鐘，建議補齊或考慮合併角色：ナレーション、凛、桜
```

**為什麼多角色非開不可**：不同角色有大量同名檔案（`凛/voice_001.wav` 與
`桜/voice_001.wav`）。不分組時合格區與淘汰區會互相覆蓋，且 Phase 4 無從得知
哪句話屬於哪個角色 —— Phase 5 選參考音訊時混到別人的音色，會直接毀掉音色一致性。

詳細說明：[docs/PHASE1-INGEST.md](docs/PHASE1-INGEST.md)

---

## 目錄結構

```
vOS-VirtualVoice/
├── configs/
│   ├── pipeline.default.json      # 主設定（門檻值都在這裡，程式碼不寫死）
│   └── pipeline.local.json        # 本機覆寫（已 gitignore，優先被讀取）
├── data/                          # 音訊資料（不進版控）
│   ├── 00_raw/                    # ← 把遊戲語音放這裡（多角色用 00_raw/<角色名>/）
│   ├── 01_unpacked/               # 容器解包產物
│   ├── 02_wav24_48k/              # ✅ Phase 1 合格產物（Phase 2 的輸入）
│   ├── 99_rejected/               # ✂ 被淘汰的檔案（複製，便於人工複核）
│   └── _reports/
│       ├── manifest_phase1.csv    # 逐檔明細（含 group、來源→產物、時長、RMS/Peak…）
│       ├── phase1_summary.json    # 機器可讀摘要（含逐角色統計，Phase 2 直接讀）
│       ├── phase1_tune.json       # 門檻分析結果
│       ├── audition_*.wav / *.md  # 人工試聽帶與對照清單
│       └── PHASE1_REPORT.md/.json # 綜合報告（驗收清單 + 下一步）
├── docs/
│   ├── PHASE1-INGEST.md           # Phase 1 完整操作手冊
│   ├── FORMAT-PLAYBOOK.md         # 遊戲音訊格式決策手冊（Wwise/CriWare/FMOD/Unity/Unreal）
│   └── PHASE3-ENVIRONMENT.md      # Phase 3 環境配置與 VRAM 參數矩陣
│   ├── hwinfo.py                  # CUDA/MPS/CPU 偵測（不依賴 WMI）
│   ├── trainfit.py                # VRAM -> 訓練參數擬合
├── scripts/
│   ├── phase1_prep.py             # 主程式（doctor/scan/unpack/convert/audition/tune/report/all）
│   ├── run_phase1.ps1 / .sh       # 一鍵入口（Windows / macOS+Linux）
│   ├── config.py                  # 設定載入 + 跨平台路徑與外部工具尋找
│   ├── formats.py                 # 內容嗅探：看 magic bytes，不看副檔名
│   ├── unpack.py                  # vgmstream 容器解包（就地解包、subsong 上限、冪等）
│   ├── audition.py                # 人工試聽抽樣：分層抽樣 + 響度對齊試聽帶
│   ├── tune.py                    # 門檻調校分析：分辨「門檻太嚴」與「素材就是這樣」
│   ├── report.py                  # 綜合報告：收斂六個來源成一份交接文件
│   └── _audioio.py                # 解碼 / 重採樣 / 單聲道 / 量測 / 寫檔
├── tests/
│   ├── conftest.py                # 自管暫存目錄 + 共用合成素材 + CLI 執行器
│   ├── fake_vgmstream.py          # vgmstream-cli 測試替身（不需下載二進位檔）
│   ├── test_phase1.py             # 嗅探 / 設定 / 音訊 I/O / 端到端
│   ├── test_unpack.py             # 解包：subsong 展開、就地解包、上限、冪等
│   ├── test_audition.py           # 試聽抽樣、響度對齊、可重現性
│   ├── test_tune.py               # 邊界群聚判斷、建議正確性
│   └── test_pipeline.py           # 全鏈路整合：容器 → unpack → convert → tune → audition → report
└── tools/bin/                     # 放 vgmstream-cli / ffmpeg（已 gitignore）
```

---

## 設計原則

1. **看內容不看副檔名** — 大量遊戲 `.wem` 其實是改名後的 Ogg Vorbis，反之亦然。
   `scripts/formats.py` 一律用 magic bytes 判斷，並在 `scan` 時提示副檔名不符的檔案。
2. **門檻值集中管理** — 時長、位元深度、LUFS 等一律放 `configs/pipeline.*.json`，
   調參不改程式碼；並用 `manifest` 留下每次跑批的可追溯紀錄。
3. **不靜默失敗** — 每個檔案的結果（ok / rejected / skipped / error）與原因都寫進 manifest。
   批次處理最貴的成本是「以為成功了」。
4. **原始素材唯讀** — 淘汰的檔案是「複製」到 `99_rejected/`，不搬移、不刪除。
5. **品質優先於數量** — `convert` 結束會直接告訴你合格素材的總分鐘數，
   並在 < 5 分鐘或 > 60 分鐘時給出對應建議。
6. **相對路徑 + 不寫死使用者名稱** — 路徑一律相對於 repo 根目錄，跨機器、跨平台可攜。
   需要把產物寫到別處時，用環境變數 `VOS_REPO_ROOT` 覆寫根目錄即可。
7. **測試不得污染真實資料** — 端到端測試用 `VOS_REPO_ROOT` 把 CLI 關進沙箱資料夾，
   測試素材永遠不會寫進你真正的 `data/`。
8. **解包不靠「希望參數是對的」** — vgmstream 的呼叫方式用 `tests/fake_vgmstream.py`
   完整驗證（多 subsong 展開、就地解包、安全上限），不需要下載二進位檔就能測。
9. **人工驗收要可行，否則不會發生** — 數千個檔案不可能手動聽完，
   所以 `audition` 把抽樣、響度對齊、對照清單一次做完。
10. **門檻調校靠數據，不靠感覺** — `tune` 分辨「0.5 秒的喘氣被淘汰」（門檻正確）
    與「1.3 秒的短台詞被淘汰」（門檻太嚴）；後者才給建議，且只建議不改檔。

---

## 硬體需求（各階段不同）

| 階段 | 主要負載 | 說明 |
|:----:|----------|------|
| 1 | CPU + 磁碟 I/O | 幾乎無 GPU 需求，可邊聽音樂邊跑 |
| 2 | CPU（UVR5 MDX-Net 可選 GPU） | 雜訊多的素材會明顯變慢 |
| 3–5 | GPU（建議 ≥ 8 GB VRAM） | Apple Silicon 走 MPS，純 CPU 可跑但很慢 |

Phase 3 會提供 CUDA / MPS / CPU 自動偵測與 VRAM 對應的 batch size 矩陣。
