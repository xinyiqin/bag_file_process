# 前后端分离部署

## 脚本 `deploy.sh`

在项目根目录执行：

```bash
./scripts/deploy.sh          # 打印用法
./scripts/deploy.sh backend  # 仅启动后端 API（默认 http://0.0.0.0:8000）
./scripts/deploy.sh frontend # 仅启动前端开发（默认 http://0.0.0.0:3000）
./scripts/deploy.sh dev      # 开发模式：同时启动后端 + 前端
./scripts/deploy.sh build    # 仅构建前端静态资源
./scripts/deploy.sh prod     # 同机部署：构建前端 + 启动后端 + 静态托管前端
```

## 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `BACKEND_PORT` | 后端端口 | 8000 |
| `FRONTEND_PORT` | 前端端口 | 3000 |
| `VITE_API_BASE` | 前端请求的后端地址（构建时注入） | http://localhost:8000 |
| `MODEL_PATH` | best.pt 模型路径 | 项目内 EAW-Yolo11/weights/best.pt |

生产环境示例：

```bash
# 构建时指定后端 API 地址（与用户访问的后端一致）
VITE_API_BASE=https://api.yourdomain.com ./scripts/deploy.sh build

# 或分别部署：后端用 uvicorn/nginx，前端用 nginx 托管 dist/
```

## 依赖

- 后端：`pip install fastapi uvicorn`，以及 `predict.py` 所需（ultralytics、opencv-python、pandas 等）
- 前端：`npm install`（在 frontend 目录）
- `prod` 模式静态托管：使用 `npx --yes serve` 或 Python 3。若出现 “Cannot copy server address to clipboard / xsel” 可忽略（仅影响复制链接，服务正常）。
