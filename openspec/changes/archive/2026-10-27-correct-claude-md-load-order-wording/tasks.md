# Tasks

## 1. 更正 CLAUDE.md 措辭

- [x] 1.1 核對事實：確認 `src/safetrajai/tracking/config.py` 檔尾在 import 時呼叫 `load_track_config(DEFAULT_TRACK_CONFIG_PATH)`，且 `tests/test_track_config.py::test_botsort_keeps_the_class_set_it_was_created_with` 通過；驗證：`grep -n "^load_track_config" src/safetrajai/tracking/config.py` 有輸出，且該測試通過
- [x] 1.2 修改 `CLAUDE.md`「設定檔」章節 `class_tracking:` 那一條：把「悄悄用到空的或舊的參數」改為「模組 import 時已載入預設的 `config/bytetrack.yaml`，順序錯了只會悄悄用到這份預設參數或先前載入的設定」，並補上 BoTSORT 建立時固定類別集合、要使用新設定必須重新建立 tracker；其餘句子維持原意；驗證：`grep -n "空的或舊的" CLAUDE.md` 無輸出，`git diff --stat` 只有 `CLAUDE.md` 且只動該條
- [x] 1.3 確認沒有其他地方被連帶影響；驗證：完整 `pytest` 通過（與先前基準 739 passed、11 skipped 相同），`git status --short` 中本 change 相關的只有 `CLAUDE.md` 與 `openspec/changes/`

## Workflow follow-up

- 完成後歸檔（`/opsx:archive`）；本 change 沒有 delta spec，歸檔時無 spec 同步
- 另開 change：整理 `Markdown/USAGE.md`（含第 221 行同樣的舊措辭）
