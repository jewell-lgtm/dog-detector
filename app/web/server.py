import asyncio
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.state import AppState

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Dog Detector")


_state: AppState | None = None


def set_state(state: AppState):
    global _state
    _state = state


class ROIRequest(BaseModel):
    points: list[list[int]]


class TriggerRequest(BaseModel):
    event: str  # "enter" or "leave"


@app.get("/stream")
async def stream():
    async def generate():
        while True:
            if _state is None:
                await asyncio.sleep(0.1)
                continue
            jpeg = _state.get_frame_jpeg()
            if jpeg is None:
                await asyncio.sleep(0.1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            await asyncio.sleep(0.05)  # ~20fps

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
async def get_state():
    if _state is None:
        return {}
    return _state.to_dict()


@app.get("/api/config")
async def get_config():
    if _state is None:
        return {}
    return {"roi_points": _state.roi_points}


@app.post("/api/roi")
async def set_roi(req: ROIRequest):
    if _state is None:
        return {"error": "not ready"}
    _state.set_roi_from_web([tuple(p) for p in req.points])
    return {"ok": True}


@app.post("/api/roi/clear")
async def clear_roi():
    if _state is None:
        return {"error": "not ready"}
    _state.clear_roi_from_web()
    return {"ok": True}


@app.post("/api/trigger")
async def trigger(req: TriggerRequest):
    if _state is None:
        return {"error": "not ready"}
    _state.set_trigger(req.event)
    _state.log_event(f"WEB TRIGGER: {req.event}")
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
