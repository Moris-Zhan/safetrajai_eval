# Proposal

## Why

`CLAUDE.md` 的「設定檔」章節說 `load_track_config()` 順序錯了「只會悄悄用到空的或舊的參數」。讀程式碼後，「空的」不成立：`tracking/config.py` 在 import 時就會載入預設的 `config/bytetrack.yaml`（檔尾 `load_track_config(DEFAULT_TRACK_CONFIG_PATH)`），所以全域參數從來不會是空的。實際發生的是：太早建立的 tracker 或 `Detector` 用到**預設情境（bytetrack）的參數**，或先前載入的另一份設定，而不是使用者指定的情境。

這段文字是給 Claude 與維護者的操作規則，措辭不準確會讓人去找「空參數」的錯誤，卻忽略真正的風險：指定 `botsort_reid_timm.yaml` 等情境時，悄悄用的是 ByteTrack 預設參數。另外，BoTSORT 在建立時就固定類別集合（spec `track-config-loading`「BoTSORT 在建立時固定類別集合」），之後再載入設定不會更新已建立的 tracker，這點目前沒有寫進同一段。

## What Changes

- 更正 `CLAUDE.md`「設定檔」章節中 `class_tracking:` 那一條：
  - 把「悄悄用到空的或舊的參數」改為準確描述：模組 import 時已載入預設的 `config/bytetrack.yaml`，順序錯了只會悄悄用到這份預設參數或先前載入的設定。
  - 補一句：BoTSORT 在建立時固定類別集合，之後載入的設定不會更新已建立的 tracker，要使用新設定必須重新建立。
- 其他句子（必須在 `create_tracker()`／`Detector` 之前呼叫、`--track-config` 不指定時讀 `--config` 本身）維持原意。
- 不改程式碼、不改測試、不改 spec（行為本身已由 `track-config-loading` 規格涵蓋，這裡只更正說明文字）。
- 不處理 `Markdown/USAGE.md:221` 的同樣句子——該檔另開 change 整理。

## Capabilities

### New Capabilities
<!-- 無 -->

### Modified Capabilities
<!-- 無：只更正文件措辭，沒有需求變動；此 change 設定 skip_specs: true -->

## Impact

- 檔案：只有 `CLAUDE.md`（一條說明文字）。
- 行為：無。程式碼、測試、設定檔、spec 都不變。
- 風險：`CLAUDE.md` 是 Claude Code 每次載入的專案指示，措辭須與程式碼一致；apply 時以 `config.py` 檔尾的 import 時載入與 `test_botsort_keeps_the_class_set_it_was_created_with` 為依據核對。
