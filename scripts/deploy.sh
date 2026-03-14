#!/usr/bin/env bash
# 前后端分离部署脚本
# 用法:
#   ./scripts/deploy.sh backend     # 仅启动后端 API (默认 8000)
#   ./scripts/deploy.sh frontend    # 仅启动前端开发服务器 (默认 3000)
#   ./scripts/deploy.sh dev          # 同时启动后端+前端(开发模式，后端后台运行)
#   ./scripts/deploy.sh build        # 仅构建前端静态资源(可设 VITE_API_BASE)
#   ./scripts/deploy.sh prod         # 构建前端 + 启动后端 + 用静态服务提供前端(同机部署)

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MODEL_PATH=${MODEL_PATH:-$ROOT/EAW-Yolo11/weights/best.pt}

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
# 生产环境前端请求的后端地址，需与最终访问后端的 URL 一致
VITE_API_BASE="${VITE_API_BASE:-http://localhost:8000}"
# 模型路径(可选)，不设则 backend 使用默认
start_backend() {
  if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "请先安装后端依赖: pip install fastapi uvicorn"
    exit 1
  fi
  # 若 best.pt 依赖 EAW-Yolo11 自定义模块，需设置 PYTHONPATH
  if [ -d "$ROOT/EAW-Yolo11" ]; then
    export PYTHONPATH="${ROOT}/EAW-Yolo11:${PYTHONPATH:-}"
  fi
  echo "启动后端 API: http://0.0.0.0:${BACKEND_PORT}"
  cd "$ROOT/backend"
  exec python3 -m uvicorn app:app --host 0.0.0.0 --port "$BACKEND_PORT"
}

start_frontend_dev() {
  if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "正在安装前端依赖..."
    (cd "$ROOT/frontend" && npm install)
  fi
  echo "启动前端开发服务器: http://0.0.0.0:${FRONTEND_PORT}"
  cd "$ROOT/frontend"
  exec npm run dev
}

build_frontend() {
  if [ ! -d "$ROOT/frontend/node_modules" ]; then
    (cd "$ROOT/frontend" && npm install)
  fi
  echo "构建前端 (VITE_API_BASE=${VITE_API_BASE})"
  cd "$ROOT/frontend"
  VITE_API_BASE="$VITE_API_BASE" npm run build
  echo "构建完成: $ROOT/frontend/dist"
}

start_prod() {
  build_frontend
  # 后台启动后端（在 backend 目录下运行，以便 import predict）
  (
    cd "$ROOT/backend"
    export PYTHONPATH="${ROOT}/EAW-Yolo11:${PYTHONPATH:-}"
    python3 -m uvicorn app:app --host 0.0.0.0 --port "$BACKEND_PORT"
  ) &
  BKPID=$!
  sleep 2
  if command -v npx &>/dev/null; then
    echo "前端静态: http://0.0.0.0:${FRONTEND_PORT}"
    (cd "$ROOT/frontend/dist" && npx --yes serve -s -l "$FRONTEND_PORT") || (cd "$ROOT/frontend/dist" && python3 -m http.server "$FRONTEND_PORT")
  else
    (cd "$ROOT/frontend/dist" && python3 -m http.server "$FRONTEND_PORT")
  fi
  kill $BKPID 2>/dev/null || true
}

case "${1:-}" in
  backend)   start_backend ;;
  frontend)  start_frontend_dev ;;
  dev)
    echo "开发模式: 后端 ${BACKEND_PORT}，前端 ${FRONTEND_PORT}"
    start_backend &
    sleep 2
    start_frontend_dev
    ;;
  build)     build_frontend ;;
  prod)      start_prod ;;
  *)
    echo "用法: $0 {backend|frontend|dev|build|prod}"
    echo "  backend   - 仅启动后端 API (端口 BACKEND_PORT=${BACKEND_PORT})"
    echo "  frontend  - 仅启动前端开发服务器 (端口 FRONTEND_PORT=${FRONTEND_PORT})"
    echo "  dev       - 同时启动后端+前端(开发)"
    echo "  build     - 仅构建前端 (可设 VITE_API_BASE=${VITE_API_BASE})"
    echo "  prod      - 构建前端并启动后端+静态服务(同机部署)"
    exit 0
    ;;
esac
