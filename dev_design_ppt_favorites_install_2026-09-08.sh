#!/usr/bin/env bash
set -e

# ============================================================
# 依 Notion「Claude Code Resources」資料庫 Favorite 項目整理
# 需求範圍：ppt簡報製作 / 釐清問題 / 設計方案 / 規劃實作 /
#           程式開發 / 軟體架構設計 / 記憶管理 / 程式碼審查
# ============================================================

# ===== Plugin =====

# planning-with-files - Planning with Files marketplace 與其實際 Claude Code plugin，
# 透過持久化檔案保存規劃狀態，並整合 skills、hooks、slash commands 與 scripts。(規劃實作)
claude plugin marketplace add OthmanAdi/planning-with-files
claude plugin install planning-with-files@planning-with-files -y

# mattpocock-skills - 已列入 Claude Code 官方 marketplace 的工程 skills plugin。(程式開發)
npx skills@latest add mattpocock/skills  --agent claude-code -y

# superpowers - Obra Superpowers 的實際 plugin，聚焦 TDD、debugging、規劃與協作開發。(程式開發)
# 風險備註：先在測試 repository 驗證 commands、agents 與 hooks 行為。
claude plugin marketplace add obra/superpowers-marketplace
claude plugin install superpowers@superpowers-marketplace -y

claude plugin marketplace add affaan-m/everything-claude-code
claude plugin install ecc@ecc -y

# ===== Skill =====

# nature-paper2ppt - 將學術論文轉換為簡報 (ppt簡報製作)
npx skills add https://github.com/yuan1z0825/nature-skills --skill nature-paper2ppt --agent claude-code -y

# ppt-visual - 簡報視覺設計優化 (ppt簡報製作)
npx skills add https://github.com/claude-office-skills/skills --skill ppt-visual --agent claude-code -y

# brainstorming - 腦力激盪流程引導，用於發散思考／方案探索 (釐清問題 / 設計方案)
npx skills add obra/superpowers@brainstorming --agent claude-code -y

# pptx - 建立與編輯 PowerPoint 簡報 (ppt簡報製作)
npx skills add anthropics/skills@pptx --agent claude-code -y

# requesting-code-review - 請求程式碼審查流程 (程式碼審查)
npx skills add obra/superpowers@requesting-code-review --agent claude-code -y

# architecture-patterns - 軟體架構設計模式 (軟體架構設計)
npx skills add wshobson/agents@architecture-patterns --agent claude-code -y

# project-planner - 專案規劃工具 (規劃實作)
npx skills add https://github.com/modimihir07/agentic-os/tree/main/skills/project-planner --agent claude-code -y

# open-code-review - 開源程式碼審查工具 (程式碼審查)
npx skills add https://github.com/alibaba/open-code-review --skill open-code-review --agent claude-code -y

# memory-evolution - 記憶演化與更新機制 (記憶管理)
npx skills add https://github.com/OPPO-Mente-Lab/X-OmniClaw/tree/main/app/src/main/assets/skills/memory-evolution --agent claude-code -y

# writing-plans - 撰寫實作計劃 (規劃實作)
npx skills add obra/superpowers@writing-plans --agent claude-code -y

# ===== 其他（Framework/Workflow）=====

# OpenSpec - Spec-Driven Development Framework，透過 /opsx:propose、/opsx:apply
# 在寫程式前先對齊需求規劃 (規劃實作)
npm install -g @fission-ai/openspec@latest
openspec init
# 更新用：npm install -g @fission-ai/openspec@latest && openspec update

# open-slide - Agent 原生簡報製作框架，以 /create-slide、/slide-authoring、/apply-comments
# 驅動「產出→點選留言→套用修改」的簡報製作循環工作流程 (ppt簡報製作)
npx @open-slide/cli init my-slide
