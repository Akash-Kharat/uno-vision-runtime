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

@router.post("/pause")
async def pause_inference(request: Request):
    request.app.state.inference_manager.pause()
    return request.app.state.inference_manager.get_status()

@router.post("/resume")
async def resume_inference(request: Request):
    request.app.state.inference_manager.resume()
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
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=10)
    
    # Send initial state
    await websocket.send_json({
        "type": "runtime_state",
        "payload": inference_manager.get_status()
    })
    
    def on_result(snapshot: InferenceResultSnapshot):
        # We must push to the asyncio queue safely from the background thread
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait, 
                snapshot
            )
        except asyncio.QueueFull:
            # If client is too slow, we just drop this frame (latest-result semantics)
            pass

    inference_manager.register_callback(on_result)
    
    try:
        while True:
            # We wait for either a new snapshot or periodically send a heartbeat
            try:
                snapshot = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_json({
                    "type": "detection_result",
                    "sequence_id": snapshot.sequence_id,
                    "timestamp": snapshot.timestamp,
                    "payload": snapshot.response.model_dump(mode='json')
                })
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "payload": inference_manager.get_status()
                })
    except WebSocketDisconnect:
        pass
    finally:
        inference_manager.unregister_callback(on_result)
