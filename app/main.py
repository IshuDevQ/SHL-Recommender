from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .catalog import load_catalog
from .llm import maybe_rephrase
from .policy import decide
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_recommender")


@asynccontextmanager
async def lifespan(application: FastAPI):
    n = len(load_catalog())
    from .retrieval import _bm25_index
    _bm25_index()
    from .embeddings import warm_embeddings   
    warm_embeddings()                         
    logger.info("Catalog loaded: %d assessments", n)
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    started = time.monotonic()
    try:
        # Hard cap at 8 turns (per assignment spec)
        if len(req.messages) > 8:
            return ChatResponse(
                reply=(
                    "We've reached the end of this conversation — to keep "
                    "refining, please start a new one."
                ),
                recommendations=[],
                end_of_conversation=True,
            )
        result = decide(req.messages)
        result.reply = maybe_rephrase(result.reply)
        return result
    except Exception:
        logger.exception("Unhandled error in /chat")
        return ChatResponse(
            reply=(
                "Something went wrong on my end — could you rephrase, or "
                "tell me the role or skill you're assessing for?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )
    finally:
        elapsed = time.monotonic() - started
        if elapsed > 5:
            logger.warning("Slow /chat call: %.2fs", elapsed)


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=200,
        content=ChatResponse(
            reply="Sorry, something went wrong. Could you try again?",
            recommendations=[],
            end_of_conversation=False,
        ).model_dump(),
    )
