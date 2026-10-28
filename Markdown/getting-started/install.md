# 安裝

只能用 conda env `track_eval`，**不要用 uv 或另外建 venv**；安裝時一律加 `--no-deps`，避免 pip 把 CUDA 版 torch 換成 PyPI 版。

```bash
conda activate track_eval
pip install -e . --no-deps
pip install pytest          # dev 依賴
```

- 依賴以 `pyproject.toml` 為準；`requirement.txt` 是舊的 pip 腳本，不是權威來源。
- 還沒安裝就執行 `python track_eval.py` 時，程式會印出上面的安裝指令並以 exit 1 結束。
- 所有指令都**從 repo 根目錄執行**（`models/`、`dataset/`、`config/` 都是相對路徑）。
- 執行時可能會出現 `Error importing BURST due to missing underlying dependency: No module named 'tabulate'`，這是 vendored TrackEval 的選用資料集模組，本專案不會用到，可以忽略。
