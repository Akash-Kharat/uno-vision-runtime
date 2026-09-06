"""Runtime API router for inference control and WebSockets."""

import asyncio
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from app.domain.runtime_state import InferenceResultSnapshot
from app.core.exceptions import AppError

router = APIRouter()

@router.get("/status")
async def get_status(request: Request):
    return request.app.state.inference_manager.get_status()

@router.post("/start")
async def start_inference(request: Request):
    request.app.state.inference_manager.start()
    return request.app.state.inference_manager.get_status()

@router.post("/stop")
async def stop_inference(request: Request):
    request.app.state.inference_manager.stop()
    return request.app.state.inference_manager.get_status()

@router.get("/result")
async def get_result(request: Request):
    res = request.app.state.inference_manager.get_latest_result()
    if not res:
        return {"success": False, "message": "No results yet"}
    return {"success": True, "result": res}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    inference_manager = websocket.app.state.inference_manager
    
    # Send initial state
    await websocket.send_json({
        "type": "runtime_state",
        "payload": inference_manager.get_status()
    })
    
    try:
        while True:
            # Pulse at ~30 FPS
            await asyncio.sleep(1.0 / 30.0)
            
            res = inference_manager.get_latest_tracked_result()
            if res:
                await websocket.send_json({
                    "type": "detection_result",
                    "sequence_id": res["sequence_id"],
                    "timestamp": res["detection_timestamp"],
                    "payload": res["payload"]
                })
            else:
                await websocket.send_json({
                    "type": "heartbeat",
                    "payload": inference_manager.get_status()
                })
    except WebSocketDisconnect:
        pass
