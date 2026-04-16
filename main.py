from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import models
from database import engine
from routers import employees, signatures, validation, config as config_router, siamese as siamese_router, extract_signatures as extract_signatures_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables on startup
    models.Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Signature Validation API",
    description=(
        "API xác thực chữ ký viết tay cho phiếu giao hàng.\n\n"
        "**Luồng sử dụng:**\n"
        "1. Tạo nhân viên (`POST /employees`)\n"
        "2. Upload 2 chữ ký mẫu từ hợp đồng (`POST /employees/{code}/signatures`)\n"
        "3. Xác thực chữ ký trên phiếu giao hàng (`POST /validate`)\n"
        "4. Điều chỉnh ngưỡng và trọng số nếu cần (`PUT /config`)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(employees.router)
app.include_router(signatures.router)
app.include_router(validation.router)
app.include_router(config_router.router)
app.include_router(siamese_router.router)
app.include_router(extract_signatures_router.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}
