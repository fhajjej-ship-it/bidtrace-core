# SPDX-License-Identifier: MPL-2.0
"""Host inference service. Start one worker bound to 127.0.0.1:8293."""
from __future__ import annotations
import base64
import binascii
import re
from typing import Any, Literal
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .models import ModelFailure, inference_context, normalize_image_sources, runtime, select_obligation_batch
from .obligations import OBLIGATIONS_PER_BATCH

app = FastAPI(title="BidTrace local original models")

@app.middleware("http")
async def localhost_only(request: Request, call_next):
    from starlette.responses import JSONResponse
    if request.client and request.client.host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return JSONResponse(status_code=403, content={"detail": "Local inference only"})
    context = {"run_id": request.headers.get("X-BidTrace-Run-ID"),
               "stage": request.headers.get("X-BidTrace-Stage")}
    if any(value is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)
           for value in context.values()):
        return JSONResponse(status_code=422, content={"detail": "Invalid inference correlation identifier"})
    token = inference_context.set(context)
    try:
        return await call_next(request)
    finally:
        inference_context.reset(token)

class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(max_length=128)
    is_query: bool = False

class RerankRequest(BaseModel):
    query: str = Field(max_length=16000)
    texts: list[str] = Field(max_length=128)

class ImagesRequest(BaseModel):
    images: list[str] = Field(max_length=32)

class QueryRequest(BaseModel):
    text: str = Field(max_length=16000)

class AssessmentImageSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["current", "previous", "evidence", "unknown"]
    title: str = Field(default="Unlabelled source image", min_length=1, max_length=500)
    document_id: str | None = Field(default=None, min_length=1, max_length=128)
    page: int | None = Field(default=None, ge=1, strict=True)

class AssessmentRequest(BaseModel):
    requirement: str | dict[str, Any]
    evidence: list[dict[str, Any]] = Field(max_length=20)
    images: list[str] = Field(default_factory=list, max_length=8)
    image_sources: list[AssessmentImageSource] | None = Field(default=None, max_length=8)

    @model_validator(mode="after")
    def aligned_image_sources(self):
        try:
            normalize_image_sources(len(self.images),
                [source.model_dump() for source in self.image_sources] if self.image_sources is not None else None,
                self.evidence)
        except ModelFailure as exc:
            raise ValueError(str(exc)) from exc
        return self

class AssessmentBatchRequest(AssessmentRequest):
    model_config = ConfigDict(extra="forbid")
    obligation_ids: list[str] = Field(min_length=1, max_length=OBLIGATIONS_PER_BATCH)

    @model_validator(mode="after")
    def canonical_obligation_selection(self):
        try:
            contract = select_obligation_batch(self.requirement, self.obligation_ids)
        except ModelFailure as exc:
            raise ValueError(str(exc)) from exc
        self.obligation_ids = [o["obligation_id"] for o in contract["obligations"]]
        return self

def decode_images(items):
    images = []
    for item in items:
        if len(item) > 20_000_000:
            raise HTTPException(413, "Page image exceeds local inference limit")
        try:
            images.append(base64.b64decode(item, validate=True))
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(422, "Invalid base64 image") from exc
    return images

def invoke(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except ModelFailure as exc:
        raise HTTPException(503, str(exc)) from exc

@app.get("/health")
def health():
    return runtime.status()

@app.post("/embed")
def embed(body: EmbeddingRequest):
    return {"vectors": invoke(runtime.embed, body.texts, body.is_query)}

@app.post("/rerank")
def rerank(body: RerankRequest):
    return {"scores": invoke(runtime.rerank, body.query, body.texts)}

@app.post("/visual/embed")
def visual_embed(body: ImagesRequest):
    return {"vectors": invoke(runtime.visual_embed, decode_images(body.images))}

@app.post("/visual/query")
def visual_query(body: QueryRequest):
    return {"vectors": invoke(runtime.visual_query, body.text)}

@app.post("/assess")
def assess(body: AssessmentRequest):
    return invoke(runtime.assess, body.requirement, body.evidence, decode_images(body.images),
        image_sources=[source.model_dump() for source in body.image_sources] if body.image_sources is not None else None)

@app.post("/unload")
def unload():
    runtime.unload()
    return {"unloaded": True}

@app.post("/assess/batch")
def assess_batch(body: AssessmentBatchRequest):
    return invoke(runtime.assess_batch, body.requirement, body.evidence, decode_images(body.images),
        image_sources=[source.model_dump() for source in body.image_sources] if body.image_sources is not None else None,
        obligation_ids=body.obligation_ids)
