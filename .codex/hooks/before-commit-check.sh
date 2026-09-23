#!/usr/bin/env bash
#
# Run this script before creating a commit. It formats active staged Python
# files in-place, then lints them with Ruff. Review and re-stage any files
# changed by the formatter before committing.
#
# Usage:
#   bash .claude/hooks/before-commit-check.sh

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Error: this script must run inside a Git repository." >&2
    exit 1
}
cd "$repo_root"

if command -v ruff >/dev/null 2>&1; then
    ruff_cmd=(ruff)
elif python -m ruff --version >/dev/null 2>&1; then
    ruff_cmd=(python -m ruff)
else
    cat >&2 <<'EOF'
Error: Ruff is required for the before-commit check.

Install it in the active development environment, for example:
  python -m pip install ruff
EOF
    exit 1
fi

mapfile -t staged_python_files < <(
    git diff --cached --name-only --diff-filter=ACMR -- '*.py'
)

target_files=()
for file in "${staged_python_files[@]}"; do
    case "$file" in
        archive/*|trackeval/*|markdown/reference_code/*|markdown/reference_results/*)
            # Historical, vendored, and reference code are intentionally not reformatted.
            ;;
        *)
            target_files+=("$file")
            ;;
    esac
done

if ((${#target_files[@]} == 0)); then
    echo "before-commit: no active staged Python files to format or lint."
    exit 0
fi

echo "before-commit: formatting ${#target_files[@]} staged Python file(s)..."
"${ruff_cmd[@]}" format "${target_files[@]}"

echo "before-commit: linting staged Python file(s)..."
"${ruff_cmd[@]}" check "${target_files[@]}"

if ! git diff --quiet -- "${target_files[@]}"; then
    cat <<'EOF'

Formatting changed one or more files. Review the changes, then stage them again:
  git add <changed files>
EOF
fi

echo "before-commit: format and lint checks passed."
