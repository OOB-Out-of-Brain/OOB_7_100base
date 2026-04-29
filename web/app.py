"""
뇌 CT 출혈 분석 웹 서버 (FastAPI + HTML)

실행:
    cd ~/OOB_7_100base
    ./venv/bin/python web/app.py
    또는
    ./venv/bin/uvicorn web.app:app --reload --port 7860

http://localhost:7860 에서 UI 접속
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import queue as _queue
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── 파이프라인 싱글톤 ──────────────────────────────────────────────────────────
_pipeline = None
_pipeline_error: Optional[str] = None

CLS_CKPT = Path(__file__).parent.parent / "checkpoints/classifier/best_classifier.pth"
SEG_CKPT = Path(__file__).parent.parent / "checkpoints/segmentor/best_segmentor.pth"
STATIC_DIR = Path(__file__).parent / "static"


def _get_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    if _pipeline_error:
        raise RuntimeError(_pipeline_error)
    if not CLS_CKPT.exists():
        _pipeline_error = f"분류 체크포인트 없음: {CLS_CKPT}"
        raise RuntimeError(_pipeline_error)
    if not SEG_CKPT.exists():
        _pipeline_error = f"분할 체크포인트 없음: {SEG_CKPT}"
        raise RuntimeError(_pipeline_error)
    from inference.pipeline import StrokePipeline
    _pipeline = StrokePipeline(
        classifier_ckpt=str(CLS_CKPT),
        segmentor_ckpt=str(SEG_CKPT),
    )
    return _pipeline


# ── 세션 상태 (단일 사용자 로컬 도구, 최대 20개 LRU) ────────────────────────
_SESSION_MAX = 20
_sessions: dict[str, dict] = {}


def _register_session(session_id: str, data: dict) -> None:
    """세션 등록 + 오래된 세션 자동 정리 (LRU cap)."""
    _sessions[session_id] = data
    if len(_sessions) > _SESSION_MAX:
        oldest = next(iter(_sessions))
        del _sessions[oldest]


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI(title="OOB 뇌 CT 출혈 분석")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ndarray_to_b64(image_np: np.ndarray, max_side: int = 800) -> str:
    img = Image.fromarray(image_np.astype(np.uint8))
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _result_to_dict(result) -> dict:
    return {
        "class_name": result.class_name,
        "class_idx": result.class_idx,
        "confidence": round(result.confidence, 4),
        "class_probs": {k: round(v, 4) for k, v in result.class_probs.items()},
        "decision_source": result.decision_source,
        "override_reason": result.override_reason,
        "lesion_area_px": result.lesion_area_px,
        "lesion_area_pct": round(result.lesion_area_pct, 2),
        "raw_lesion_area_pct": round(result.raw_lesion_area_pct, 2),
        "lesion_component_count": result.lesion_component_count,
        "kept_component_count": result.kept_component_count,
        "max_component_mean_prob": round(result.max_component_mean_prob, 4),
        "segmentation_confidence": round(result.segmentation_confidence, 4),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html을 찾을 수 없습니다.</h1>", status_code=500)


@app.get("/api/status")
async def status():
    ckpt_ok = CLS_CKPT.exists() and SEG_CKPT.exists()

    def _check_ollama() -> bool:
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False

    ollama_ok = await asyncio.to_thread(_check_ollama)
    return {
        "checkpoints": ckpt_ok,
        "cls_ckpt": str(CLS_CKPT),
        "seg_ckpt": str(SEG_CKPT),
        "ollama": ollama_ok,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """이미지 업로드 → 파이프라인 실행 → JSON + overlay base64 반환."""
    try:
        pipeline = _get_pipeline()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 읽기 실패: {e}")

    orig_np = np.array(img)

    try:
        result = pipeline.run(orig_np, make_overlay=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파이프라인 오류: {e}")

    session_id = str(uuid.uuid4())
    _register_session(session_id, {
        "result": result,
        "orig_np": orig_np,
        "chat_history": [],
    })

    resp = _result_to_dict(result)
    resp["session_id"] = session_id
    if result.overlay_image is not None:
        resp["overlay_b64"] = _ndarray_to_b64(result.overlay_image)
    resp["orig_b64"] = _ndarray_to_b64(orig_np)

    return JSONResponse(resp)


@app.get("/api/stream_llm/{session_id}")
async def stream_llm(session_id: str, model: str = "llama3.2-vision:11b"):
    """SSE: 세션의 파이프라인 결과에 대해 LLM 판독을 스트리밍."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="세션 없음 — 먼저 분석을 실행하세요.")

    result = session["result"]
    orig_np = session["orig_np"]

    async def generate():
        q: _queue.SimpleQueue = _queue.SimpleQueue()

        def _producer():
            try:
                import ollama
                from inference.llm_reporter import _ndarray_to_jpeg_b64, _build_user_prompt
                client = ollama.Client(host="http://localhost:11434")
                overlay_np = result.overlay_image if result.overlay_image is not None else orig_np
                image_b64 = _ndarray_to_jpeg_b64(overlay_np, max_side=512)
                prompt = _build_user_prompt(result, mode="balanced", image_used=True)
                stream = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
                    options={"temperature": 0.3, "num_predict": 600},
                    stream=True,
                )
                for chunk in stream:
                    q.put(("token", chunk["message"]["content"]))
                q.put(("done", None))
            except ImportError:
                q.put(("error", "ollama 패키지 없음: pip install ollama"))
            except Exception as e:
                q.put(("error", str(e)))

        threading.Thread(target=_producer, daemon=True).start()

        loop = asyncio.get_event_loop()
        while True:
            kind, val = await loop.run_in_executor(None, q.get)
            if kind == "token":
                yield f"data: {json.dumps({'token': val}, ensure_ascii=False)}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            else:
                yield f"data: {json.dumps({'error': val})}\n\n"
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat")
async def chat(request: Request):
    """SSE: 사용자 후속 질문에 대한 LLM 스트리밍 응답."""
    body = await request.json()
    session_id = body.get("session_id")
    question = body.get("question", "").strip()
    model = body.get("model", "llama3.2-vision:11b")

    if not question:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="세션 없음")

    result = session["result"]
    history = session.get("chat_history", [])

    SYSTEM = (
        "당신은 뇌 CT 출혈 분석 파이프라인의 결과를 설명하는 의료 연구 보조 챗봇입니다. "
        "분류 결과와 병변 정보만 바탕으로 답변하고, 의학적 진단을 내리지 마세요. "
        "모르는 것은 '의료진에게 문의하세요'라고 답하세요. 반드시 한국어로 답변하세요."
    )

    context = (
        f"[파이프라인 결과] 분류={result.class_name.upper()} "
        f"신뢰도={result.confidence:.1%} "
        f"병변면적={result.lesion_area_pct:.1f}% "
        f"병변컴포넌트={result.kept_component_count}/{result.lesion_component_count}"
    )

    messages = [{"role": "system", "content": SYSTEM + "\n\n" + context}]
    for h in history[-6:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": question})

    async def generate():
        q: _queue.SimpleQueue = _queue.SimpleQueue()

        def _producer():
            try:
                import ollama
                client = ollama.Client(host="http://localhost:11434")
                stream = client.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": 0.4, "num_predict": 400},
                    stream=True,
                )
                for chunk in stream:
                    q.put(("token", chunk["message"]["content"]))
                q.put(("done", None))
            except Exception as e:
                q.put(("error", str(e)))

        threading.Thread(target=_producer, daemon=True).start()

        loop = asyncio.get_event_loop()
        full_response = ""
        while True:
            kind, val = await loop.run_in_executor(None, q.get)
            if kind == "token":
                full_response += val
                yield f"data: {json.dumps({'token': val}, ensure_ascii=False)}\n\n"
            elif kind == "done":
                history.append({"user": question, "assistant": full_response})
                session["chat_history"] = history[-10:]
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            else:
                yield f"data: {json.dumps({'error': val})}\n\n"
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  OOB 뇌 CT 출혈 분석 웹 서버")
    print("  http://localhost:7860")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
