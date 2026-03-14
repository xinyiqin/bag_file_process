"""
HTTP API：上传视频，调用 best.pt 做轨迹跟踪与摄食分类，返回 JSON 给前端。
运行前请确保：1) 已安装 ultralytics；2) 若 best.pt 含自定义模块，设置 PYTHONPATH 包含 EAW-Yolo11。
启动: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import os
import tempfile
import threading
import queue
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from predict import run_analysis

app = FastAPI(title="鱼类跟踪与行为量化 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型路径：环境变量 MODEL_PATH 或默认项目内 EAW-Yolo11/weights/best.pt
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "EAW-Yolo11", "weights", "best.pt"),
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    tracker: str = "botsort.yaml",
    conf: float = 0.25,
    fps: float = 30.0,
    scale: float = 0.1,
    min_consecutive_frames: int = 3,
):
    """上传视频，运行轨迹跟踪与摄食分类，返回前端所需 JSON。min_consecutive_frames：连续多少帧为摄食才计为 1 次摄食。"""
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"模型文件不存在: {MODEL_PATH}，请设置环境变量 MODEL_PATH 或放置 best.pt")
    suffix = Path(video.filename or "video").suffix or ".mp4"
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            content = await video.read()
            f.write(content)
            path = f.name
        result = run_analysis(
            path,
            MODEL_PATH,
            tracker=tracker,
            conf=conf,
            iou=0.45,
            fps=fps,
            scale=scale,
            min_consecutive_frames=min_consecutive_frames,
        )
        # 返回可 JSON 序列化的部分（含预测视频 base64，若生成）
        out = {
            "coordinates": result["coordinates"],
            "trajectory": result["trajectory"],
            "feeding_stats": result["feeding_stats"],
            "speed": result["speed"],
            "acceleration": result["acceleration"],
            "angle": result["angle"],
            "zone_stats": result["zone_stats"],
            "zone_over_time": result.get("zone_over_time", []),
            "cumulative_feeding_events_over_time": result.get("cumulative_feeding_events_over_time", []),
            "heatmap": result["heatmap"],
            "feeding_frequency": result["feeding_frequency"],
            "frame_shape": result["frame_shape"],
        }
        if result.get("output_video_base64"):
            out["output_video_base64"] = result["output_video_base64"]
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except Exception:
                pass


def _build_out(result):
    """从 run_analysis 返回的 result 构建可 JSON 序列化的 out。"""
    out = {
        "coordinates": result["coordinates"],
        "trajectory": result["trajectory"],
        "feeding_stats": result["feeding_stats"],
        "speed": result["speed"],
        "acceleration": result["acceleration"],
        "angle": result["angle"],
        "zone_stats": result["zone_stats"],
        "zone_over_time": result.get("zone_over_time", []),
        "cumulative_feeding_events_over_time": result.get("cumulative_feeding_events_over_time", []),
        "heatmap": result["heatmap"],
        "feeding_frequency": result["feeding_frequency"],
        "frame_shape": result["frame_shape"],
    }
    if result.get("output_video_base64"):
        out["output_video_base64"] = result["output_video_base64"]
    return out


@app.post("/analyze-stream")
async def analyze_video_stream(
    video: UploadFile = File(...),
    tracker: str = "botsort.yaml",
    conf: float = 0.25,
    fps: float = 30.0,
    scale: float = 0.1,
    min_consecutive_frames: int = 3,
):
    """流式分析：先返回进度 NDJSON（type=progress），最后返回 type=result。前端可据此显示进度条。"""
    if not os.path.isfile(MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"模型文件不存在: {MODEL_PATH}")
    suffix = Path(video.filename or "video").suffix or ".mp4"
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            content = await video.read()
            f.write(content)
            path = f.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    progress_queue = queue.Queue()
    result_holder = []
    exc_holder = []

    def run():
        try:
            def on_progress(current, total):
                progress_queue.put({"type": "progress", "current": current, "total": total})

            def on_partial(partial_data):
                progress_queue.put({"type": "partial", "data": partial_data})

            result = run_analysis(
                path,
                MODEL_PATH,
                tracker=tracker,
                conf=conf,
                iou=0.45,
                fps=fps,
                scale=scale,
                min_consecutive_frames=min_consecutive_frames,
                progress_callback=on_progress,
                partial_callback=on_partial,
                partial_interval=15,
            )
            out = _build_out(result)
            progress_queue.put({"type": "result", "data": out})
        except Exception as e:
            exc_holder.append(e)
            progress_queue.put({"type": "error", "detail": str(e)})
        finally:
            progress_queue.put(None)
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except Exception:
                pass

    thread = threading.Thread(target=run)
    thread.start()

    async def ndjson_stream():
        loop = asyncio.get_event_loop()
        while True:
            msg = await loop.run_in_executor(None, progress_queue.get)
            if msg is None:
                break
            yield json.dumps(msg, ensure_ascii=False) + "\n"
            if isinstance(msg, dict) and msg.get("type") == "partial":
                await asyncio.sleep(0.05)

    return StreamingResponse(
        ndjson_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
