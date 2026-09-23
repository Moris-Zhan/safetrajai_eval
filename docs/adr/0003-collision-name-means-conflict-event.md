# 程式碼沿用 `collision` 命名，領域術語是「衝突事件」

stage3 用 TTC／PET 找出的是交通衝突事件（接近碰撞的潛在風險），不是實際碰撞。領域術語統一為**衝突事件（Conflict Event）**（見 `CONTEXT.md`），但程式碼的套件名稱（`safetrajai/collision/`）、設定檔（`config/collision_detect.yaml`）與相關 skill 刻意沿用 `collision`：這些名稱已經是既有介面，而且專案重構的範圍只限於結構，改名的價值不足以抵銷連帶的破壞。讀程式碼或輸出時，`collision` 一律理解為衝突事件，事件數不能當作碰撞次數或正式的安全結論。

## Considered Options

- **改名為 `conflict/`**：語意最精確；如果之後有其他原因要動 stage3 的介面，可以重新評估。
