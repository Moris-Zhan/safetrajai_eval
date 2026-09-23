# Tracker 的 ultralytics override 方法凍結不動

`ClassAwareBOTSORT` 與 `ClassAwareBYTETracker` 中 override 自 ultralytics 的方法（`__init__`、`update`、`remove_duplicate_stracks`，以及 BoTSORT 的 `init_track`）由使用者明確指定凍結：方法本體一字不改，不搬移到其他類別，也不抽成共用函式，即使這些方法彼此重複、或含有已知的落差（例如 `track_buffer` 在兩個版本中的單位相差 30 倍）也一樣。這些方法是所有 HOTA／AssA 實驗結果的基準，任何改動都會讓既有實驗無法比較。整個檔案可以搬到新位置，但只能修改檔案層級的 import；smoke 測試會用 AST 比對方法本體，確保這項約束成立。擴充類別感知（class-aware）邏輯時，一律寫在本專案新增的方法（例如 `get_dists_class_aware()`）或 `matching.py`、`config.py` 裡。
