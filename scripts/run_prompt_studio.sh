#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "未找到 Python，请先安装 Python 3.10+"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[Prompt Studio] 创建虚拟环境..."
  "$PYTHON_CMD" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[Prompt Studio] 安装/检查依赖..."
python -m pip install -r requirements.txt --disable-pip-version-check >/dev/null

if [ ! -f "configs/model_config.yaml" ] && [ -f "configs/model_config.template.yaml" ]; then
  cp "configs/model_config.template.yaml" "configs/model_config.yaml"
  echo "[Prompt Studio] 已从模板创建 configs/model_config.yaml"
fi

echo "[Prompt Studio] 启动中: http://127.0.0.1:8610/studio"
python -m uvicorn prompt_studio_app:app --host 0.0.0.0 --port 8610 --reload
