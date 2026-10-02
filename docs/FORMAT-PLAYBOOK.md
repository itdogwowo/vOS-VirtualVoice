# 遊戲音訊格式決策手冊（Phase 1 任務 1 附錄）

> 用途：當 `scan` 顯示的格式你沒見過、或 `unpack` 失敗時，查這份手冊。
> 每個段落都給「怎麼判斷 → 該用什麼工具 → 具體指令 → 失敗時看哪裡」。

**先看這張表再往下讀。** 90% 的情況會落在前三列：

| `scan` 顯示 | 意義 | 你要做的事 |
|---|---|---|
| `Ogg Vorbis` / `RIFF/WAVE` | 已經是獨立音訊檔 | 什麼都不用裝，直接 `convert` |
| `Wwise Vorbis` / `Wwise 其他編碼` | 需要 vgmstream | 裝 [vgmstream](#1-wwise-引擎--wem) 後 `unpack` |
| `FMOD Sample Bank` / `CriWare AWB` | 需要 vgmstream + 可能需要金鑰 | 讀對應段落，**companion file 是關鍵** |
| `未知格式` | 需要先找出引擎 | 讀 [§7 未知格式的處理流程](#7-未知格式的處理流程) |

---

## 0. 判斷你的遊戲用哪個引擎

不知道引擎時，用**檔案指紋**反推最快：

| 你看到的檔案 | 幾乎可以確定是 |
|---|---|
| `.wem`、`.bnk`、`.pck`、`SoundbanksInfo.xml`、`GeneratedSoundBanks/` | **Wwise**（Audiokinetic） |
| `.acb`、`.awb`、`.cpk`、`.usm`、`.adx`、`.hca`、`CriAtom` | **CRIWARE**（常見於日系遊戲 ← 你的專案大概率在這） |
| `.fsb`、`.bank`、`.guids.txt`、`fmod.dll`、`Master.strings.bank` | **FMOD** |
| `.assets`、`.resource`、`.bundle`、`*_Data/`、`globalgamemanagers` | **Unity**（音訊可能裝在 AssetBundle 裡） |
| `.pak`、`.utoc`/`.ucas`、`.uasset`、`Engine/`、`Content/` | **Unreal Engine** |
| `.ogg` 大量散落 + `.sli`、`KiriKiri`、`.xp3` | **KiriKiri** 或自製引擎 |

> **日系遊戲的經驗法則**：`acb`/`awb`/`hca`/`adx` 是 CRIWARE 的組合拳，
> 從《Fate》《原神》《崩壞》到多數 Galgame 都用它。
> 遇到這組就重點看 [§2](#2-criware-引擎--acb--awb--hca--adx)。

---

## 1. Wwise 引擎 — `.wem`

### 最常見的誤解：`.wem` 不一定要解包

Wwise 的 `.wem` 有四種長相，**只有兩種真的需要 vgmstream**：

| 實際內容 | `scan` 顯示 | 需要解包？ |
|---|---|---|
| 改了副檔名的 Ogg Vorbis | `Ogg Vorbis` | ❌ 直接 `convert` |
| RIFF 容器 + `vorb` chunk | `Wwise Vorbis` | ✅ 要 |
| RIFF 容器 + Wwise 專有 codec | `Wwise 其他編碼` | ✅ 要 |
| 原始 PCM/ADPCM | `RIFF/WAVE` | ❌ 直接 `convert` |

**判斷方法**：跑 `scan`。它會直接告訴你「需先解包：N 個」。

> 為什麼會這樣：Wwise 為了讓音訊能直接餵給某些平台的原生解碼器，
> 會刻意輸出「標準容器 + 自訂 codec」的混合體。這也解釋了為什麼
> 「同一個遊戲的 `.wem`，有些能直接播、有些不能」。

### 指令

```bash
# 1) 先看敵情
python scripts/phase1_prep.py scan

# 2) 只有「需先解包」的才會被處理
python scripts/phase1_prep.py unpack

# 3) 標準化
python scripts/phase1_prep.py convert
```

### Wwise 的坑

| 症狀 | 原因 | 對策 |
|---|---|---|
| `unpack` 說找不到 vgmstream-cli | 沒裝 | 見下方「安裝」 |
| 解出的檔案有嚴重的長尾／回音 | `.wem` 帶 loop 標記 | `unpack` 已預設加 `-i`（不要循環） |
| 只有一部分 `.wem` 解出來 | 混了兩種格式 | 這正常 —— `scan` 的分布表會顯示 |
| 想找 BGM 卻挖到一堆音效 | `SoundbanksInfo.xml` 裡有分類 | 用 `SoundbanksInfo.xml` 過濾出 Voice 類別後再處理 |

### 進階：用 `SoundbanksInfo.xml` 精準挑素材

Wwise 專案通常會附一份 `SoundbanksInfo.xml`（或 `SoundbanksInfo.json`），
裡面記錄每個 `.wem` 的**事件名稱與分類**。這是把「語音」跟「音效／BGM」分開的關鍵：

```bash
# 找出所有含 Voice / Dialog 關鍵字的項目
# macOS / Linux
grep -iE 'voice|dialog|talk|serif' SoundbanksInfo.xml | head -50

# Windows PowerShell
Select-String -Path SoundbanksInfo.xml -Pattern 'voice|dialog|talk' | Select-Object -First 50
```

有了名稱對照，你甚至可以把檔名還原成人類可讀的名字（對 Phase 4 標註很有幫助）。

---

## 2. CRIWARE 引擎 — `.acb` / `.awb` / `.hca` / `.adx`

**這是日系遊戲最常見的組合，也是本專案最可能遇到的情況，請仔細讀。**

### 關鍵觀念：`.acb` 不是音訊檔

| 檔案 | 角色 | 類比 |
|---|---|---|
| `.acb` | **索引表**（cue sheet）：哪些 cue 對應哪個 wave、叫什麼名字 | 書的目錄 |
| `.awb` | **音訊本體**：一堆 `.hca`/`.adx` 打包在一起 | 書的內文 |
| `.hca` / `.adx` | 個別音訊（可能是加密的） | 單一頁 |

**所以：只有 `.acb` 是解不出任何東西的。** 你必須 `.acb` + `.awb` 配對。

> 好消息：`unpack` 失敗時會自動檢查同目錄，直接告訴你「找不到配對的 `xxx.awb`」。

### 正確的處理流程

```
data/00_raw/<角色名>/
├── bgm_voice.acb      ← 索引表
├── bgm_voice.awb      ← 音訊本體（必須跟 .acb 放一起！）
└── bgm_voice.hcakey   ← 若有加密才需要
        ↓ unpack
data/01_unpacked/<角色名>/bgm_voice/bgm_voice#1.wav ...
```

**順序很重要**：`.acb` 與 `.awb` 必須在同一層。若你把 `.awb` 放到別的子資料夾，
vgmstream 找不到它，你會得到「什麼都沒解出來」而且錯誤訊息很含糊。

### 加密金鑰的三種情況

| 情況 | 判斷 | 對策 |
|---|---|---|
| 未加密 | 直接解成功 | 沒事 |
| 金鑰內附在 `.awb` 裡（新版 CRIWARE） | 直接解成功 | 沒事 |
| 需要外部金鑰檔 | 解出無聲／失敗 | 需要 `.hcakey` 或 `.adxkey` |

金鑰檔的製作方式（vgmstream 官方格式）：

```bash
# .hcakey：一行純文字，內容是 keystring 或 64-bit 數字
# 檔名為 .hcakey（整目錄生效）或 <檔名>.hcakey（單檔生效）
echo -n "1234567890123456" > .hcakey
```

> ⚠️ **金鑰通常來自遊戲執行檔**，需要逆向或社群整理。
> 若你的遊戲是熱門作品，先搜尋 `<遊戲名> hcakey` —— 社群通常已經有整理。
> 本專案不提供繞過 DRM 的方法；請自行確認你的用途符合當地法規與遊戲授權條款。

### `.adx` / `.hca` 單獨出現時

若遊戲直接把音訊存成 `.adx` 或 `.hca`（沒有包在 `.awb` 裡），
`scan` 會顯示 `CRI ADX` / `CRI HCA`，且 `needs_unpack = no`
—— 它們是獨立音訊，會直接進 `convert`。

但**加密的 `.hca` 仍需要 `.hcakey`**，而且注意：`convert` 用的是
soundfile/ffmpeg，它們不認得 `.hca`。這種情況要強制走 vgmstream：

```json
// configs/pipeline.local.json
{
  "decode": { "backend_priority": ["vgmstream", "ffmpeg", "soundfile"] }
}
```

---

## 3. FMOD 引擎 — `.fsb` / `.bank`

### 判斷

- 檔名像 `Master.bank`、`Master.strings.bank` → FMOD Studio 的 bank
- `Music.bank`、`SFX.bank` → 同上
- `.fsb` → 較舊的 FMOD Sample Bank

### 處理

```bash
python scripts/phase1_prep.py scan     # 會顯示 FMOD Sample Bank
python scripts/phase1_prep.py unpack   # FSB 內含多個 subsong，會全部展開
```

### FMOD 的坑

| 症狀 | 原因 | 對策 |
|---|---|---|
| 解不出來 | `.fsb` 加密 | 需要 `.fsbkey`（hex 字串） |
| 檔名無意義（數字流水號） | 名稱存在 `.bank` 的 strings 表 | `.strings.bank` 是用來對照名稱的，可用 `fmod_bank_tools` 之類工具還原 |
| 音訊是短的（1 秒內） | 遊戲用「prefetch」機制 | 這是正常的，完整版在別處；`unpack` 的 1.5 秒門檻會自動剔除 |

> **prefetch 是什麼**：遊戲為了掩蓋載入時間，會把一小段音訊放在記憶體裡先播，
> 同時背景載入完整檔案。這些 prefetch 檔不是 bug，但對訓練無用 ——
> 而它們通常短於 1.5 秒，所以 Phase 1 的時長篩選會自動清掉。

---

## 4. Unity 引擎 — AssetBundle / `.assets`

### 判斷

看到 `*_Data/`、`globalgamemanagers`、`.assets`、`.resource`、`.bundle` → Unity。

### 三種情況

| 情況 | 特徵 | 工具 |
|---|---|---|
| 音訊是散落的 `.wav`/`.ogg` | 直接可見 | 不用處理，`convert` 即可 |
| 音訊在 `.assets` / `.resource` 內 | 需要解包 | [AssetRipper](https://github.com/AssetRipper/AssetRipper) 或 [UABEA](https://github.com/nesrak1/UABEA) |
| 音訊在 AssetBundle（`.bundle`，可能加密） | 需要解包 + 可能需金鑰 | AssetRipper 或 [AssetStudio](https://github.com/Perfare/AssetStudio)（已停止維護但常仍可用） |

### 處理流程

1. 用 AssetRipper / UABEA 開啟遊戲的 `*_Data` 資料夾
2. 篩選類型 `AudioClip`，匯出成 WAV
3. 匯出結果放進 `data/00_raw/<角色名>/`，然後 `convert`

> Unity 的 `AudioClip` 匯出後通常是標準 WAV，所以後續流程完全一樣。
> 但注意 Unity 常見 22050 Hz 或 44100 Hz 的來源 —— `convert` 會統一升到 48 kHz。

---

## 5. Unreal Engine — `.pak` / `.utoc` / `.ucas`

### 判斷

看到 `Engine/`、`Content/`、`.pak`、`.uasset`、`.utoc`/`.ucas`（UE5）→ Unreal。

### 工具

| 工具 | 用途 |
|---|---|
| [FModel](https://fmodel.app/) | **首選**：GUI，支援 UE4/UE5，可直接匯出音訊 |
| [UnrealPak](https://github.com/allcoolthingsatoneplace/UnrealPakTool) | 官方工具的封裝版 |
| [repak](https://github.com/trumank/repak) | 命令列，跨平台，適合腳本化 |

### 處理流程

1. FModel 指向遊戲的 `Paks` 資料夾
2. 找 `Sound` / `Audio` 目錄下的 `SoundWave` / `MediaSoundWave` 資產
3. 匯出成 WAV → 放進 `data/00_raw/` → `convert`

### Unreal 的坑

- UE 的 `.uasset` 可能存的是**壓縮過的 ADPCM 或 Bink Audio**，匯出工具通常會處理，但偶爾會給出損壞的 WAV
- `SoundCue` 是組合邏輯（隨機選一個 wave、加效果），要匯出底層的 `SoundWave` 才是原始語音

---

## 6. 其他常見封裝

| 封裝 | 遊戲／引擎 | 工具 |
|---|---|---|
| `.xp3` | KiriKiri（大量日系 Galgame） | [GARbro](https://github.com/morkt/GARbro) ← **日系遊戲首選萬用工具** |
| `.cpk` | CRIWARE 封裝 | GARbro 或 `unpack` 直接處理 |
| `.dat` / `.bin` + 索引檔 | 自製引擎 | [QuickBMS](http://aluigi.altervista.org/quickbms.htm) + 對應 script |
| `.arc` / `.pak` / `.resS` | 各類自製 | GARbro 先試，失敗再找 QuickBMS script |
| `.ogg` 大量 + `.sli` | 自製引擎 | `.sli` 是循環點資訊，可忽略（不影響訓練） |

> **GARbro 值得先試**：它支援的日系遊戲封裝格式數量遠超其他工具，
> 而且有 GUI，不需要寫 script。對 Phase 1 來說它能省下大量時間。

---

## 7. 未知格式的處理流程

當 `scan` 顯示「未知格式」時，按這個順序排查：

### 步驟 1：看 hex 檔頭

```bash
# macOS / Linux
xxd -l 64 "你的檔案"

# Windows PowerShell
Format-Hex -Path "你的檔案" -Count 64
```

把前 16 bytes 拿去搜尋。常見的魔數（magic bytes）對照：

| 檔頭（hex / ASCII） | 格式 |
|---|---|
| `52 49 46 46` = `RIFF` | RIFF 家族（WAV/WEM/AVI） |
| `4F 67 67 53` = `OggS` | Ogg |
| `66 4C 61 43` = `fLaC` | FLAC |
| `46 53 42 34/35` = `FSB4`/`FSB5` | FMOD |
| `41 46 53 32` = `AFS2` | CRIWARE AWB |
| `40 55 54 46` = `@UTF` | CRIWARE ACB |
| `41 4B 50 4B` = `AKPK` | Wwise PCK / 新版 bank |
| `48 43 41 00` = `HCA\0` | CRI HCA |
| `43 52 49 44` = `CRID` | CRIWARE USM（影片） |
| `1A 45 DF A3` | Matroska / WebM |
| `FF F1` / `FF FB` | AAC / MP3 |
| `00 00 00 20 66 74 79 70` = `ftyp` | MP4 / M4A |

### 步驟 2：嘗試 vgmstream

vgmstream 支援的格式遠超它的名聲。**不知道是什麼就直接餵給它**：

```bash
# 用 -m 只問資訊，不解碼
vgmstream-cli -m "你的檔案"

# 如果它說得出來，就直接解
vgmstream-cli -i -o out#?s.wav -S 0 "你的檔案"
```

### 步驟 3：檢查是否為容器

若 vgmstream 也說不出來，可能是**自製封裝**。判斷方法：
看檔案裡有沒有一段一段可辨識的壓縮資料（例如搜尋 `OggS` 出現在非開頭的位置）。

```bash
# 找 OggS 在檔案中的位置（結尾以外的位置 = 內含 Ogg 子檔）
python -c "
import sys
data = open(sys.argv[1],'rb').read()
pos, hits = 0, []
while (i := data.find(b'OggS', pos)) != -1 and len(hits) < 10:
    hits.append(i); pos = i + 4
print('OggS 出現在 offset:', hits)
" "你的檔案"
```

有的話就是容器 → 用 GARbro / QuickBMS 拆。

### 步驟 4：最後手段 —— 用 TXTH 自製標頭

若你已知是原始 PCM/ADPCM 資料但沒有標頭（`start_offset`、`sample_rate`
等資訊藏在 `.exe` 裡），vgmstream 支援用 `.txth` 文字檔描述格式：

```
# bgm_01.pcm.txth
codec = PCM16LE
channels = @0x04
sample_rate = 48000
start_offset = 0x10
num_samples = data_size
```

完整說明見 [vgmstream TXTH 文件](https://github.com/vgmstream/vgmstream/blob/master/doc/TXTH.md)。

### 步驟 5：還是不行 → 告訴我

把以下資訊貼給我，我可以幫你改寫 `scripts/formats.py` 的嗅探邏輯：

1. `scan` 的完整輸出
2. 檔案的 hex 前 64 bytes
3. 檔名樣式（有沒有規律？例如 `voice_0001.bin`）
4. 遊戲名稱／引擎（若知道）

---

## 8. vgmstream-cli 安裝（三平台）

`unpack` 唯一的依賴。**Windows 的使用者請注意要一起放 DLL**，
否則會出現「明明解出來但沒聲音」或「不支援 Ogg」的情況。

### Windows（建議：放進 `tools/bin/`）

1. 到 [vgmstream 官網](https://vgmstream.org/) 或
   [GitHub Releases](https://github.com/vgmstream/vgmstream/releases) 下載 `vgmstream-win64.zip`
2. 解壓後，把 **`vgmstream-cli.exe` 與所有 `.dll`** 一起複製到本專案的 `tools/bin/`
3. 驗證：

```powershell
.\tools\bin\vgmstream-cli.exe
# 應印出 Usage 與版本資訊
```

> ⚠️ **一定要連 DLL 一起放**。少了 `libvorbis.dll` / `libmpg123-0.dll` /
> `avcodec-vgmstream-*.dll` 等檔案，Wwise 的 Vorbis 語音會解不出來 ——
> 而且錯誤訊息通常只是「不支援此格式」，非常容易誤判成「這個檔案壞了」。

`tools/bin/` 已在 `.gitignore` 內，不會被 commit。

### macOS

```bash
brew install vgmstream
vgmstream-cli      # 驗證
```

### Linux

```bash
# Debian / Ubuntu
sudo apt install vgmstream

# Arch
yay -S vgmstream

# 或從 Releases 下載靜態檔放進 tools/bin/
```

### ffmpeg（同樣建議安裝）

處理 `.m4a` / `.aac` / `.wma` 等 libsndfile 吃不下來的格式時需要：

| 平台 | 指令 |
|---|---|
| Windows | `winget install Gyan.FFmpeg` 或 `choco install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` |

安裝後跑 `python scripts/phase1_prep.py doctor` 確認全部就緒。

---

## 9. 決策流程圖（懶人版）

```
scan 顯示什麼？
│
├─ Ogg Vorbis / RIFF/WAVE ──────────────► convert        （不用裝任何工具）
│
├─ Wwise Vorbis / Wwise 其他編碼 ───────► 裝 vgmstream → unpack → convert
│
├─ FMOD Sample Bank ────────────────────► 裝 vgmstream → unpack → convert
│                                         （加密則需 .fsbkey）
│
├─ CriWare ACB/AWB ─────────────────────► 確認 .acb 與 .awb 同目錄
│                                         → 裝 vgmstream → unpack → convert
│                                         （加密則需 .hcakey）
│
├─ CRI ADX / CRI HCA ───────────────────► 改 decode.backend_priority 為 vgmstream
│                                         → convert
│
└─ 未知格式 ────────────────────────────► §7 的四步排查
                                          仍無解 → 把 hex 與檔名貼給我
```

---

## 10. 相關文件

- [Phase 1 操作手冊](PHASE1-INGEST.md) — 主要流程與驗收標準
- [vgmstream 官方使用說明](https://github.com/vgmstream/vgmstream/blob/master/doc/USAGE.md)
- [vgmstream 支援格式清單](https://github.com/vgmstream/vgmstream/blob/master/doc/FORMATS.md)
- [GARbro](https://github.com/morkt/GARbro) — 日系遊戲封裝萬用工具
- [FModel](https://fmodel.app/) — Unreal Engine 資產匯出
- [AssetRipper](https://github.com/AssetRipper/AssetRipper) — Unity 資產匯出
