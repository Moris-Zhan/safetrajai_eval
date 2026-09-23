# `third_party/trackeval` 是修改過的 vendored 副本

TrackEval 以 vendored 方式放在 `third_party/trackeval/`，而不是用 pip 安裝上游版本，因為本專案改過它的資料集定義（例如 `datasets/mot_challenge_2d_box.py` 的 6 類別對照與 1-based 類別 ID），讓它能評估本專案固定的 6 個類別。這份副本視為唯讀：不格式化、不做局部修補。要升級到上游版本時，必須先比對並重新套用這些修改，否則評估會靜默地用錯類別。
