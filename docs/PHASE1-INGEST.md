# Phase 1 — 數據解包與格式標準化

> **一句話總結**：把遊戲裡挖出來的一堆 `.wem/.acb/.fsb/.ogg`（還常常副檔名騙人），
> 變成 `24bit / 48kHz / 單聲道 PCM WAV`，同時把「不可能拿來訓練的檔案」提前剔除。
>
> **為什麼這階段最值得花時間**：Phase 4 的標註與 Phase 5 的推理品質，
> 上限完全由這裡決定。後面再怎麼調參，都救不回一個底噪很重或長度只有 0.4 秒的素材。

---

## 0. 目錄與資料流

### 單一角色（`grouping.by = "none"`，預設）

```
data/00_raw/          ← 你把遊戲語音丟這裡（可含任意層子資料夾）
      │
      │ ① scan    嗅探真實格式（不寫檔，先看清楚敵情）
      │ ② unpack  vgmstream-cli 解開容器
      ▼
data/01_unpacked/     ← 容器解出的獨立音訊檔
      │
      │ ③ convert 解碼 → 單聲道 → 48kHz → 24bit WAV
      │           ＋ 時長篩選 ＋ 品質關卡
      ├──────────────────────────────▶ data/02_wav24_48k/   ✅ 合格（Phase 2 的輸入）
      └──────────────────────────────▶ data/99_rejected/    ✂ 淘汰（原檔複製，供人工複核）

data/_reports/manifest_phase1.csv   逐檔明細：來源→產物、時長、Peak、RMS、淘汰原因
data/_reports/phase1_summary.json   機器可讀摘要（Phase 2 直接讀，含逐角色統計）
```

### 多角色（`grouping.by = "folder"`，強烈建議）

多角色的素材請放成 `data/00_raw/<角色名>/`：

```
data/00_raw/
├── 凛/                    ├── 桜/                    └── ナレーション/
│   ├── voice_001.wav      │   ├── voice_001.wav      └── narr_01.wav
│   └── voice_002.wav      │   └── voice_002.wav
      │                            │                         │
      ▼                            ▼                         ▼
data/02_wav24_48k/凛/      data/02_wav24_48k/桜/      data/02_wav24_48k/ナレーション/
```

**為什麼多角色非用不可**：不同角色有大量**同名檔案**（`凛/voice_001.wav` 與
`桜/voice_001.wav`）。不分組的話：

1. 合格區與淘汰區都會互相覆蓋，你看到的檔案數是假的
2. Phase 4 打標籤時無從得知哪句話屬於哪個角色
3. Phase 5 選參考音訊時會混到別人的音色 —— 這是**最嚴重**的，會直接毀掉音色一致性

開啟方式（`configs/pipeline.local.json`）：

```json
{ "grouping": { "by": "folder" } }
```

`convert` 結束時會多印一張**逐角色素材量表**，並標出素材不足 3 分鐘的角色：

```
 逐角色素材量（Phase 4 會依此分開打標籤、Phase 5 分開選參考音訊）：
   群組                       合格     淘汰     錯誤       分鐘
   凛                        12      1      0      0.8
   桜                         3      1      0      0.1
   ナレーション                    1      1      0      0.1

   ⚠ 以下群組素材 < 3 分鐘，建議補齊或考慮合併角色：ナレーション、凛、桜
```

---

## 1. 環境準備（三平台）

### 1.1 Python 依賴

```bash
# 所有平台一致
pip install -r requirements.txt

# 建議用虛擬環境（可選）
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
```

安裝後先自我檢查：

```bash
python scripts/phase1_prep.py doctor
```

`doctor` 會告訴你：Python 版本、soundfile / soxr / scipy 是否就緒、
ffmpeg 與 vgmstream-cli 找不找得到、五個資料夾是否建立、目標格式與門檻值。

### 1.2 vgmstream-cli（**遊戲容器解包的關鍵工具**）

只有當你的素材是 `.wem / .awb / .acb / .fsb / .cpk / .pck` 這類**容器**時才需要。
判斷方式很簡單：先跑 `scan`，它會直接告訴你有幾個檔案「需先解包」。

| 平台 | 安裝方式 |
|------|----------|
| **Windows** | 到 [vgmstream 官網](https://vgmstream.org/) 下載 `vgmstream-win64.zip`，把 `vgmstream-cli.exe` **與所有 `.dll`** 一起放進本專案的 `tools/bin/`（**最省事，且不污染系統 PATH**） |
| **macOS** | `brew install vgmstream` |
| **Linux** | `sudo apt install vgmstream`（Debian/Ubuntu）／`yay -S vgmstream`（Arch）／或從 Releases 下載靜態檔放 `tools/bin/` |
| **通用** | 直接下載對應平台 binary 丟進 `tools/bin/`，本工具會自動找到 |

> ⚠️ **Windows 一定要連 DLL 一起放**。少了 `libvorbis.dll` / `libmpg123-0.dll` /
> `avcodec-vgmstream-*.dll`，Wwise 的 Vorbis 語音會解不出來 —— 而錯誤訊息通常
> 只是「不支援此格式」，非常容易誤判成「這些檔案壞了」。
>
> **為什麼用 vgmstream 而不是自己寫 parser**：Wwise 的 codec 有 Vorbis / Opus /
> ADPCM / PCM / PTADPCM 多種，CriWare 的 HCA 有加密金鑰問題，FMOD 的 FSB 各版本
> 格式互不相容。vgmstream 是社群維護十年以上的事實標準，重寫不划算。
>
> 遇到沒見過的格式、或 `unpack` 失敗時，查
> **[格式決策手冊](FORMAT-PLAYBOOK.md)**（涵蓋 Wwise / CriWare / FMOD / Unity /
> Unreal / KiriKiri，含 companion file 對照表與未知格式的四步排查）。

### 1.3 ffmpeg（強烈建議）

處理 `.m4a / .aac / .wma` 或 libsndfile 吃不下來的冷門編碼時會用到。

| 平台 | 安裝方式 |
|------|----------|
| Windows | `winget install Gyan.FFmpeg` 或 `choco install ffmpeg`，也可放 `tools/bin/` |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` |

---

## 1.4 步驟 ⓿：`selftest` — 先確認工具能動（**不需要素材**）

在你花時間整理遊戲檔案之前，先花幾秒鐘確認這套工具在你的機器上真的跑得起來：

```bash
python scripts/phase1_prep.py selftest

# 出錯時保留臨時工作區，方便進去查看現場
python scripts/phase1_prep.py selftest --keep-workspace
```

它用**合成音訊**跑一次真正的流程（標準化 → 篩選 → 試聽 → 報告），並驗證：

```
  ✔ 音訊 I/O 依賴（soundfile + numpy）  -- libsndfile 1.2.2
  ✔ 重採樣後端（soxr 或 scipy）  -- soxr（VHQ）
  ✔ 產生合成素材  -- 5 個檔案（涵蓋合格／過短／過長／過弱）
  ✔ convert 執行成功  -- exit 0
  ✔ 篩選規則正確（該過的過、該擋的擋）  -- 5 筆紀錄、0 筆不符
  ✔ 合格區檔案數正確  -- 2 個（預期 2）
  ✔ 產物規格為 48 kHz / 24-bit / 單聲道  -- 2 個檔案
  ✔ 淘汰檔複製到 99_rejected 且原檔保留  -- 淘汰區 3 個、原檔完整=True
  ✔ audition 產生試聽帶  -- exit 0、1 份試聽帶
  ✔ tune 產生分析結果  -- exit 0
  ✔ report 產生綜合報告  -- exit 0

 通過 11/11 項

 ✔ 工具鏈在你的機器上可以正常運作。
    下一步：把遊戲語音放進 data/00_raw/，然後跑 scan
```

三個刻意的設計：

| 設計 | 為什麼 |
|------|--------|
| **完全隔離** | 在暫存區建立臨時 repo，用 `VOS_REPO_ROOT` 把 CLI 關進去，**絕不碰你真正的 `data/`**。跑完就刪。 |
| **涵蓋每一條篩選規則** | 合成素材刻意包含合格、過短、過長、訊號過弱四種，所以它同時驗證「該過的過、**該擋的擋**」—— 不是只驗證 happy path。 |
| **不依賴外部工具** | vgmstream / ffmpeg 缺席也能跑完（Phase 1 對純 WAV 素材本來就不需要它們）。缺的工具會列成「提醒」而不是「失敗」。 |

> 這一項的價值在於**發現時機**：環境問題若等到下載完素材、整理好資料夾才發現，
> 排查成本會高得多（你會先懷疑素材，而不是環境）。

---

## 2. 步驟 ①：`scan` — 先看清楚素材的真實長相

```bash
python scripts/phase1_prep.py scan
```

**這一步不寫任何檔案**，只做三件事：

1. 統計 `data/00_raw/` 內每個副檔名的數量與總大小
2. 用 **magic bytes**（不是副檔名！）判斷每個檔案的真實格式
3. 提示「副檔名與實際格式不符」的檔案

實際輸出長這樣：

```
素材掃描：9 個檔案，3.9 MiB
格式                                             數量     需解包
RIFF/WAVE (PCM 或 ADPCM)                         7       否
FMOD Sample Bank (.fsb/.bank)                    1       是
Wwise Vorbis (.wem, RIFF+vorb)                   1       是
  可直接解碼：7 個   需先解包：2 個
```

### 為什麼一定要「看內容不看副檔名」

真實世界最常見的四個坑：

| 症狀 | 真相 | 本工具的處理 |
|------|------|--------------|
| 一堆 `.wem` 解不開 | 它其實是**改名後的 Ogg Vorbis**，根本不用解包 | 嗅探為 `ogg_vorbis`，直接進 `convert` |
| `.acb` 解出來什麼都沒有 | `.acb` 只是 **cue sheet 索引表**，音訊本體在 `.awb` | 標記為 `cri_acb`，提示需要配對的 `.awb` |
| `.ogg` 用一般播放器打不開 | 它其實是 **Wwise Vorbis**（RIFF 容器內含 `vorb` chunk） | 嗅探為 `wem_vorbis`，走 vgmstream |
| `.wav` 聽起來像雜訊 | 檔頭是 WAV 但 `wFormatTag = 0xFFFF`（Wwise 專有編碼） | 嗅探為 `wem_other`，走 vgmstream |

---

## 3. 步驟 ②：`unpack` — 解開遊戲容器

```bash
python scripts/phase1_prep.py unpack

# 實用選項
python scripts/phase1_prep.py unpack --dry-run              # 只報告不寫檔
python scripts/phase1_prep.py unpack --max-subsongs 10000   # 放寬單容器上限
python scripts/phase1_prep.py unpack --no-probe             # 跳過 subsong 探測（較快）
python scripts/phase1_prep.py unpack -v                     # 逐檔輸出
```

產物放進 `data/01_unpacked/<來源結構>/<容器檔名>/`，**一個容器一個子資料夾**。

### 這個步驟內部做了四件事（每一件都是踩過的坑）

| 步驟 | 為什麼 |
|------|--------|
| ① 先跑 `-m` 探測 subsong 數量 | 一個 `.awb` 可能內含數千條語音。不先問就全解，會直接佔滿磁碟 |
| ② 超過 `max_subsongs` 就**跳過**並提示 | 寧可跳過讓你決定，也不要靜默展開 5000 個檔案 |
| ③ **在來源目錄就地解包**，再搬走產物 | CriWare 的加密 `.hca` 需要同目錄的 `.hcakey`；`.awb` 需要配對的 `.acb`。vgmstream 只在輸入檔所在目錄找這些 companion file |
| ④ 輸出樣式一定含 `?s` 萬用字元 | 官方文件明講：`-s 1 -S 5 -o bgm.wav` 會「5 個 subsong 全部覆蓋同一個檔案 = 錯誤」。少了它你只會拿到最後一條語音 |

另外解包時一律加 `-i`（不要循環）：遊戲語音常有 loop 標記，
不加的話一段 3 秒的台詞可能被解成無限長的檔案。

### companion file 對照表

這些檔案必須跟主要檔案放在**同一個資料夾**，否則解包會失敗或得到無聲結果：

| 主檔 | 需要的同伴 | 少了會怎樣 |
|------|-----------|-----------|
| `.acb` | 配對的 `.awb` | `.acb` 只是索引表，**什麼都解不出來** |
| `.awb` | 配對的 `.acb` | 音訊可解，但失去 cue 名稱（無法得知哪句話是什麼） |
| `.hca` | `.hcakey` | 加密檔無法解碼 |
| `.adx` | `.adxkey` | 同上 |
| `.fsb` | `.fsbkey` | 同上 |
| `.xwb` | `.xsb` | 失去 cue 名稱 |

`unpack` 失敗時會自動檢查同目錄並把缺哪個檔案直接講出來，不用自己猜。

### 解包產物命名

```
data/00_raw/hero_a/bank.awb
        ↓ unpack
data/01_unpacked/hero_a/bank/bank#1.wav
data/01_unpacked/hero_a/bank/bank#2.wav
data/01_unpacked/hero_a/bank/bank#3.wav
        ...
```

`#N` 就是 subsong 編號。重跑解包時不會覆蓋既有產物，會加上 `__1` 序號。

---

## 4. 步驟 ③：`convert` — 標準化 + 篩選（本階段核心）

```bash
python scripts/phase1_prep.py convert

# 實用選項
python scripts/phase1_prep.py convert --limit 50 -v     # 先試 50 個，逐檔輸出
python scripts/phase1_prep.py convert --dry-run         # 只報告不寫檔（先看淘汰率）
python scripts/phase1_prep.py convert --workers 8       # 平行解碼（預設自動，最多 8）
python scripts/phase1_prep.py convert --overwrite       # 連已存在的輸出一起重做
python scripts/phase1_prep.py convert --no-quality-gate # 只做時長篩選
```

### 4.1 處理管線（每個檔案的完整旅程）

```
嗅探格式 ─▶ 解碼 (soundfile → ffmpeg → vgmstream 依序嘗試)
        ─▶ 轉單聲道（等功率平均，不是只取左聲道）
        ─▶ 量測時長
        ─▶ 【規則 1】時長篩選
        ─▶ 量測 Peak / RMS
        ─▶ 【規則 2】品質關卡
        ─▶ 重採樣到 48 kHz（soxr VHQ，缺則 scipy polyphase）
        ─▶ 寫出 24-bit PCM WAV
        ─▶ 回讀檔頭驗證規格（防止靜默降級）
```

### 4.2 為什麼是 `24bit / 48kHz / 單聲道`

| 參數 | 選擇 | 理由 |
|------|------|------|
| **48 kHz** | SoVITS 系列原生訓練取樣率 | 統一格式後，訓練端不必再對每個檔做重採樣；來源若 > 48k 降採樣順便抗鋸齒 |
| **24 bit** | 保留動態範圍 | 16-bit 在低電平細節（氣音、尾韻、呼吸）的量化損失，對情感模型是實質傷害 |
| **單聲道** | 訓練只需單聲道 | 立體聲讓資料量翻倍卻無增益；雙聲道轉單聲道用等功率平均，比只取左聲道乾淨 |
| **PCM WAV** | Phase 2 的 UVR5 與 loudnorm 對整數 PCM 較友善 | 不用 32-bit float 中繼檔 |

### 4.3 篩選規則（門檻都寫在 `configs/pipeline.default.json`）

| 規則 | 預設值 | 為什麼 |
|------|--------|--------|
| **過短** | `< 1.5 秒` 淘汰 | 純喘氣、單字哼聲、「はい」「えっ」這類反應音。它們會讓模型學到破碎的韻律 |
| **過長** | `> 15 秒` 淘汰 | 長句在 Phase 4 的 ASR 對齊容易出錯，且有聲書推理時 GPT-SoVITS 的參考音訊建議 3~10 秒 |
| **全靜音** | Peak = -inf | 錄音事故或解碼失敗留下的空檔 |
| **削波** | `Peak > -0.5 dBFS` | 已經爆掉的波形，UVR5 也救不回來 |
| **訊號過弱** | `RMS < -50 dBFS` | 幾乎聽不到的遠場錄音，放大只是把底噪一起放大 |

> ⚠️ **這些門檻是「起點」不是「聖旨」。** 跑完 `convert` 看到淘汰率高得離譜時，
> 正確的做法是**先看 `99_rejected/` 裡的檔案**，再決定是素材真的差，還是門檻太嚴。
> 調整方式見 §6。

### 4.4 淘汰檔為什麼是「複製」而不是「搬移」

`99_rejected/` 裡的檔案是 `copy2` 複製過來的，**原檔永遠留在 `00_raw/`**。

理由：調參是反覆的過程。今天你覺得 1.2 秒太短，明天可能想把門檻放寬到 1.0 秒；
如果當初是搬移，素材就已經被你自己搞丟了。多佔一點硬碟，換取反悔的權利。

---

## 5. 讀懂報表

### 5.1 `data/_reports/manifest_phase1.csv`

每個檔案一列，共 17 欄。最重要的幾欄：

| 欄位 | 意義 | 用途 |
|------|------|------|
| `status` | `ok` / `rejected` / `skipped` / `error` | 篩選結果 |
| `reason` | 中文原因 | 人工複核時直接看這欄 |
| `group` | 群組（多角色 = 角色名；單一角色 = `default`） | **Phase 4 分開打標籤的依據** |
| `src_path` → `out_path` | 來源與產物的對應 | **Phase 4 打標籤時的回查依據** |
| `src_sr` / `out_sr` | 來源與輸出取樣率 | 確認重採樣真的發生 |
| `src_channels` | 原始聲道數 | 確認雙聲道已正確合併 |
| `duration_sec` | 時長 | 排版參考音訊、估算訓練步數 |
| `peak_dbfs` / `rms_dbfs` | 電平 | 判斷素材是否過輕或爆掉 |
| `decoder` | 實際使用的解碼器 | 排查特定格式問題 |

檔案以 `utf-8-sig` 編碼寫出，Windows Excel 直接雙擊開啟不會亂碼。

`phase1_summary.json` 除了總計，還帶一份逐角色的 `groups` 統計，Phase 2 可以直接讀：

```json
{
  "group_by": "folder",
  "counts": { "ok": 16, "rejected": 3, "skipped": 0, "error": 0 },
  "total_ok_seconds": 66.0,
  "groups": {
    "凛":          { "ok": 12, "rejected": 1, "seconds": 48.0 },
    "桜":          { "ok": 3,  "rejected": 1, "seconds": 9.0 }
  }
}
```

### 5.2 畫面上的「合格素材總長」

`convert` 結束時會直接告訴你合格素材的總分鐘數，並依你的 **Quality Over Quantity** 原則給建議：

| 總長 | 提示 |
|------|------|
| `< 5 分鐘` | ⚠ 太少。建議 ≥ 10 分鐘，最佳 15~60 分鐘 |
| `5~60 分鐘` | ✔ 落在微調最佳區段 |
| `> 60 分鐘` | ℹ 建議挑 15~60 分鐘「最乾淨、情感最豐富」的子集 |

**為什麼 15 分鐘可能勝過 2 小時**：GPT-SoVITS 的微調會**等比例學習資料裡的所有特性**，
包含底噪、混響、壓縮失真。1000 條乾淨語音教出來的是「乾淨的發聲方式」，
1000 條含雜音的語音教出來的是「在雜音環境下的發聲方式」——後者會讓推理輸出永遠帶一層霧感。

### 5.3 步驟 ④：`audition` — 人工試聽（**不可省略**）

自動化能抓時長、電平、削波，但「這句話有沒有感情」「有沒有被截頭截尾」
「這個檔案裡是不是兩個人在講話」只有耳朵知道 —— 而這正是整個專案的核心價值。

問題是合格素材動輒數千個檔案，你不可能一個一個開。這個指令把它變成可行：

```bash
python scripts/phase1_prep.py audition --per-group 5

# 實用選項
python scripts/phase1_prep.py audition --dry-run              # 只列出抽中哪些，不產檔
python scripts/phase1_prep.py audition --groups 凛,桜         # 只抽查特定角色
python scripts/phase1_prep.py audition --per-group 10         # 每個角色抽 10 個
python scripts/phase1_prep.py audition --seed 12345           # 換一組樣本
python scripts/phase1_prep.py audition --no-align             # 不做響度對齊
```

它會產生兩個檔案在 `data/_reports/`：

1. **`audition_Nsamples_seedX.wav`** — 把所有樣本接成**單一試聽帶**
   （每段之間 0.9 秒靜音），用任何播放器順聽即可，不需要播放清單
2. **`audition_Nsamples_seedX.md`** — 對照清單，記下「第幾段」就能回查檔名

```markdown
| 段 | 群組 | 起始(秒) | 長度(秒) | Peak | RMS | 檔案 |
|---:|------|--------:|--------:|-----:|----:|------|
| 1 | ささやき | 0.0 | 2.80 | -48.0 | -53.3 | `ささやき_001.wav` |
| 3 | ナレーション | 7.1 | 3.10 | -8.0 | -13.3 | `ナレーション_002.wav` |
| 5 | 凛 | 15.4 | 8.50 | -8.0 | -13.3 | `凛_020.wav` |
```

#### 三個關鍵設計

| 設計 | 為什麼 |
|------|--------|
| **分層抽樣**（每個群組各抽 N 個） | 全域隨機在素材量懸殊時會幾乎只抽到大群組（凛 3000 條、旁白 40 條），小群組的問題永遠聽不出來 —— 而那通常正是最需要檢查的地方 |
| **固定亂數種子**（預設 `20240601`） | 同一批素材永遠抽出同一組樣本。這樣你「修完再聽一次」才有比較的意義。每組各自的種子，補新角色不會打亂既有抽樣 |
| **響度對齊**（試聽帶專用，+30 dB 硬上限） | 原始素材電平差異可能達 30 dB。不對齊的話，聽小聲檔要轉大音量、下一段大聲的會爆耳朵，你就會放棄試聽。遊戲刻意錄小聲的耳語／內心戲台詞會被提升到聽得見，但不會無上限放大成噪音 |

試聽帶輸出為 16-bit（用途是「聽」，不是訓練素材），**合格區完全不會被修改**。

#### 試聽時逐項確認

- [ ] 有無底噪／電流聲／空間混響
- [ ] 有無被截頭截尾（開頭或結尾被切掉）
- [ ] 是否同一個人（沒有「一句話裡突然換人」）
- [ ] 情感種類是否多樣（平靜、悲傷、憤怒、驚訝、耳語各有一些）

若某類問題**集中在特定群組** → 該角色的素材需要重挑或重新解包，
而不是調高門檻把問題掩蓋掉。

---

## 6. 調參與常見狀況

### 6.1 調整門檻

複製設定檔再改（**不要直接改 default**，那份要進版控）：

```bash
cp configs/pipeline.default.json configs/pipeline.local.json
# 之後所有指令都會自動優先用 pipeline.local.json
```

`pipeline.local.json` 已被 `.gitignore` 排除，你的實驗不會污染 repo。

### 6.2 狀況排除表

| 症狀 | 原因 | 對策 |
|------|------|------|
| `scan` 說「找不到任何目標副檔名」 | 檔案沒放進 `data/00_raw/`，或副檔名不在 `scan.extensions` | 確認位置；若格式特殊，把副檔名加進設定檔的 `scan.extensions` |
| `scan` 顯示「未知格式」 | 需要先確認遊戲引擎 | 見 [格式決策手冊](FORMAT-PLAYBOOK.md) §7 的四步排查 |
| `convert` 全數 `skipped`，理由「容器檔」 | 還沒跑 `unpack` | 先 `unpack`；若 `unpack` 說找不到 vgmstream-cli，見 §1.2 |
| `unpack` 說「找不到配對的 .acb/.awb」 | CriWare 的 `.acb` 只是索引表，音訊在 `.awb` | 把配對的兩個檔案放在**同一層**資料夾（見 [手冊 §2](FORMAT-PLAYBOOK.md#2-criware-引擎--acb--awb--hca--adx)） |
| `unpack` 解出無聲檔案 | 加密格式缺金鑰 | 需要 `.hcakey` / `.adxkey` / `.fsbkey`（見 [手冊 §8](FORMAT-PLAYBOOK.md#8-vgmstream-cli-安裝三平台)） |
| 大量 `error: 解碼失敗` | 缺 ffmpeg，或該格式太冷門 | 裝 ffmpeg 後重跑；仍失敗就看手冊的排查流程 |
| 大量「削波風險」淘汰 | 素材本身音量過大（常見於遊戲 BGM 混音後的語音） | 先目視 `99_rejected/` 確認；若確實只是太大聲，把 `peak_dbfs_max` 放寬到 `0.0`，把音量處理留給 Phase 2 |
| 「訊號過弱」誤殺 | 遊戲刻意做小聲的耳語/內心戲台詞 | 把 `min_rms_dbfs` 放寬到 `-60`；這類素材對情感模型其實很有價值 |
| 不同角色的同名檔案在 `99_rejected/` 只剩一個 | 群組沒開，或多角色素材沒按角色分資料夾 | 設 `grouping.by = "folder"` 並把素材整理成 `00_raw/<角色名>/`（見 §0） |
| Phase 4 不知道哪句話是哪個角色 | manifest 沒有 `group` 資訊 | 同上，開啟 `grouping.by = "folder"` 後重跑 `convert` |
| 不確定門檻該不該調 | — | 跑 `python scripts/phase1_prep.py tune`（見 §6.3） |
| 想重跑但不想重做已完成的部分 | — | 直接重跑即可。`convert` 具冪等性：已存在且未加 `--overwrite` 時會標記「已存在（未覆寫）」而略過 |

### 6.3 不確定門檻該不該調？用 `tune` 讓數據說話

**淘汰率本身沒有意義，淘汰的內容才有意義。** 40% 的淘汰率可能是門檻正確運作，
也可能是你正在丟掉最珍貴的短台詞與耳語。這個指令會分辨兩者：

```bash
python scripts/phase1_prep.py tune

# 實用選項
python scripts/phase1_prep.py tune --dry-run    # 只看報告，不寫 JSON
```

輸出範例：

```
 manifest：24 個檔案  合格 20（83.3%） 淘汰 4 略過 0 錯誤 0
 合格素材總長：2.0 分鐘

 現行門檻
   時長      1.5s ~ 15.0s
   Peak      ≤ -0.5 dBFS
   RMS       ≥ -50.0 dBFS

 淘汰原因分布
   過短              4 個（其中 4 個緊貼門檻）

 建議
 ⚠ [1] 考慮把最短時長從 1.5s 降到約 1.2s
    有 4 個檔案落在 1.30~1.45s 之間被淘汰（門檻是 1.5s）。
    緊貼門檻的失敗者通常不是垃圾，而是差一點的短台詞 ——
    它們對情感模型比標準長度的旁白更有價值。
    設定片段 → configs/pipeline.local.json
      { "duration_filter": { "min_seconds": 1.2 } }
```

**它的判斷邏輯是「邊界群聚」**：

| 情況 | `tune` 的反應 |
|------|--------------|
| 淘汰的是 0.3~0.8 秒的喘氣聲 | 「門檻看起來是對的，維持現值」 |
| 淘汰的是 1.3~1.45 秒的短台詞 | 「考慮降到 1.2s」+ 可貼上的設定片段 |
| 淘汰的是 RMS -52~-56 dBFS 的耳語 | 「考慮放寬 RMS 下限」+ 提醒 Phase 2 會處理音量 |
| 淘汰的是削波檔案 | **不給設定片段** —— 削波是物理損壞，沒有門檻能修好它 |
| 只有 2 筆邊界樣本 | 不建議（樣本太少 = 噪音） |

它也一併體檢素材總量與逐角色分布：

```
 逐角色合格時長
   凛                      12.5 分鐘
   桜                       0.8 分鐘  ← 不足 3 分鐘
```

> `tune` **只建議、不改檔**。門檻是藝術判斷 —— 工具負責給數據，你負責決定。
> 動手改之前，先用 `audition` 聽聽 `99_rejected/` 裡的邊界檔案。

### 6.3 中斷了怎麼辦

任何時候 `Ctrl+C` 都安全。已寫出的檔案會保留，重跑時自動略過。
（淘汰檔的複製也是同樣邏輯。）

---

## 7. 完成 Phase 1 的驗收標準

### 7.1 一鍵檢查：`report`

不用自己一項一項核對 —— 這個指令會把所有產物收斂成一份報告，逐項打勾：

```bash
python scripts/phase1_prep.py report
```

```
 驗收條件：3/5 項達標
   ✔ 已產生 manifest（convert 跑過） — 14 筆紀錄
   · 合格素材 ≥ 10 分鐘（建議 15~60） — 目前 0.6 分鐘
   · 每個角色 ≥ 3 分鐘 — 不足：桜(0.1分)、ナレーション(0.2分)
   · 人工試聽已執行（audition） — 尚未執行 audition —— 這是唯一無法自動化的驗收步驟
   · 門檻已檢視（tune） — 2 則建議待評估
```

報告寫在 `data/_reports/PHASE1_REPORT.md`（人讀）與 `PHASE1_REPORT.json`（機器讀），
內含八個章節：總覽、原始素材格式、逐角色素材量、篩選分析、**驗收清單**、
環境狀態、產物位置、**下一步指令**。

> 這份報告同時是 Phase 1 → Phase 2 的交接文件。進入 Phase 2 前請確認 §5 全部達標。

### 7.2 逐項確認（報告會自動檢查前五項）

- [ ] 已產生 manifest（`convert` 跑過）
- [ ] `data/02_wav24_48k/` 內所有檔案都是 **48 kHz / 24-bit / 單聲道 PCM WAV**
      （抽查：`python -c "import soundfile as sf; print(sf.info('你的檔案.wav'))"`）
- [ ] 合格素材總長 **≥ 10 分鐘**（建議 15~60 分鐘，以乾淨度為優先）
- [ ] 多角色專案：每個角色 **≥ 3 分鐘**（`convert` 與 `report` 都會標出不足的角色）
- [ ] 已跑過 `audition` 並**從頭聽完試聽帶** —— 這項工具無法代替你：
      - [ ] 沒有明顯底噪 / 電流聲 / 空間混響
      - [ ] 沒有被截頭截尾（開頭或結尾被切掉）
      - [ ] 沒有「一句話裡突然換人」的檔案
      - [ ] **情感種類有覆蓋**：平靜旁白、悲傷、憤怒、驚訝、耳語各有一些

> 最後一項是人工不可省略的步驟。自動化能處理長度與電平，
> 但「這句台詞有沒有感情」只有耳朵知道——而這正是整個專案的核心價值。
> `audition` 就是為了讓這一步在數千個檔案面前仍然可行而存在的。
> `report` 只能記錄你跑過工具，不能代替你的耳朵。

---

## 7.3 順帶一提：Phase 3 的環境指令

以下兩個指令**不需要素材**，也不屬於 Phase 1，但你可以隨時先跑，提早知道硬體狀況：

```bash
python scripts/phase1_prep.py env        # 偵測 CUDA/MPS/CPU 並擬合訓練參數
python scripts/phase1_prep.py precheck   # 訓練前環境自檢（合成張量）
```

詳見 [Phase 3 環境配置](PHASE3-ENVIRONMENT.md)。

---

## 8. 跨平台注意事項

本階段所有腳本都已處理以下差異，你不需要手動調整：

| 項目 | 處理方式 |
|------|----------|
| 路徑分隔符 | 一律用 `pathlib.Path`，Windows `\` 與 Unix `/` 自動處理 |
| 外部工具尋找 | `shutil.which` 邏輯 + Windows `.exe/.cmd/.bat` 副檔名 + `tools/bin/` 優先 |
| 主控台編碼 | Windows cp950/cp936/cp932 無法輸出 `✔`，會自動降級為 `[OK]`（並強制 stdout 走 UTF-8） |
| 執行子程序 | 不使用 `shell=True`，避開 `cmd.exe` 與 `sh` 的引號差異；Windows 下不彈黑色視窗 |
| 檔名大小寫 | Windows 去重時用 lower-case key，避免 `A.wav` 與 `a.wav` 互撞 |
| CSV 編碼 | `utf-8-sig`，Windows Excel 與 macOS Numbers 都正常 |
| 一鍵入口 | `scripts/run_phase1.ps1`（Windows）／`scripts/run_phase1.sh`（macOS/Linux） |

---

## 9. 指令速查

```bash
# 環境檢查
python scripts/phase1_prep.py doctor

# 嗅探格式（不寫檔，永遠從這裡開始）
python scripts/phase1_prep.py scan

# 解包容器（多 subsong 自動展開，有安全上限）
python scripts/phase1_prep.py unpack

# 標準化 + 篩選
python scripts/phase1_prep.py convert

# 人工試聽（產生試聽帶 + 對照清單）
python scripts/phase1_prep.py audition --per-group 5

# 門檻調校分析（不確定該不該改門檻時）
python scripts/phase1_prep.py tune

# 綜合報告（驗收清單 + 下一步指令，Phase 1 → Phase 2 的交接文件）
python scripts/phase1_prep.py report

# 一條龍（scan → unpack → convert）
python scripts/phase1_prep.py all

# 小量試跑（強烈建議第一次這樣做）
python scripts/phase1_prep.py convert --limit 50 -v

# 只看結果不寫檔
python scripts/phase1_prep.py convert --dry-run

# 跑測試（合成素材 + 假 vgmstream，不需要真實音檔或外部工具）
python -m pytest tests/ -q

# 一鍵包裝
.\scripts\run_phase1.ps1 convert --limit 50 -v     # Windows
bash scripts/run_phase1.sh convert --limit 50 -v   # macOS / Linux
```

---

## 10. 下一步（Phase 2 預告）

Phase 1 完成後，`data/02_wav24_48k/` 就是 Phase 2 的輸入。Phase 2 會處理：

1. **UVR5 人聲淨化** — 用 MDX-Net / VR Arch 模型濾除殘留的環境噪音、混響、變音
2. **響度標準化** — 全部統一至 `-23 LUFS`，確保有聲書長文本聆聽時音量均衡

在進入 Phase 2 之前，請先完成 §7 的三項驗收。
