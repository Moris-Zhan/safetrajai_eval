## 最可靠修法：避開 Claude 的 OSC 52

在 tmux 裡啟動 Claude 前設定：

```export
export CLAUDE_CODE_DISABLE_MOUSE=1
claude
```

或只對單次執行套用：

```
CLAUDE_CODE_DISABLE_MOUSE=1 claude
```

這樣 Claude 不會接管滑鼠文字選取。你應改成：

```
用外層 terminal 的滑鼠拖曳選取
→ 終端原生 Copy（常見為 Ctrl+Shift+C）
→ 貼上
```

 **不要使用** ：

* Claude TUI 內的滑鼠選取／點擊 copy。
* Claude 的 `/copy`。
* tmux copy-mode 中會觸發 OSC 52 的 copy binding。
