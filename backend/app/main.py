from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.chat import router as chat_router
from app.api.availability import router as availability_router
from app.api.stats import router as stats_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: ensure vector database index is initialized on startup."""
    try:
        from app.core.vector_store import get_vector_store
        vs = get_vector_store()
        vs.index_if_empty()
    except Exception as e:
        print(f"[Startup] Vector store auto-index notification: {e}")
    yield
    try:
        from app.core.vector_store import get_vector_store
        vs = get_vector_store()
        vs.close()
    except Exception:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    description="Full-stack AI-powered guest assistant backend for The Grand Azure Heritage Resort & Spa (Candolim, Goa)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware for Next.js frontend communication
cors_origins = list(settings.CORS_ORIGINS) if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS]
is_wildcard = "*" in cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if is_wildcard else cors_origins,
    allow_origin_regex=None if is_wildcard else r"https://.*\.vercel\.app",
    allow_credentials=not is_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat_router)
app.include_router(availability_router)
app.include_router(stats_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "docs": "/docs",
        "mock_mode": settings.MOCK_LLM
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/hotel-data")
def get_hotel_data():
    from pathlib import Path
    import json
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    hotel_data_path = Path(__file__).resolve().parent / "data" / "hotel_data.json"
    if not hotel_data_path.exists():
        raise HTTPException(status_code=404, detail="Hotel data file not found")
    with open(hotel_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/api/vector-store/stats")
def get_vector_store_stats():
    """Returns vector database status, collection details, and point counts."""
    from app.core.vector_store import get_vector_store
    store = get_vector_store()
    return {
        "collection_name": store.collection_name,
        "point_count": store.get_point_count(),
        "vector_dimension": store.embedder.dimension,
        "embedding_provider": store.embedder.provider_name,
        "storage_mode": store.storage_mode,
        "status": "ready" if store.get_point_count() > 0 else "empty"
    }

from pydantic import BaseModel, Field

class HyDEGenerateRequest(BaseModel):
    query: str = Field(..., description="Guest question to expand with hypothetical document passage")

@app.get("/api/hyde/stats")
def get_hyde_stats():
    """Returns HyDE generator configuration, model, and execution statistics."""
    from app.core.hyde import get_hyde_generator
    generator = get_hyde_generator()
    return generator.get_stats()

@app.post("/api/hyde/generate")
async def generate_hyde_passage(req: HyDEGenerateRequest):
    """Generates a hypothetical hotel handbook passage for a guest query using HyDE."""
    from app.core.hyde import get_hyde_generator
    generator = get_hyde_generator()
    passage = await generator.agenerate_hypothetical_document(req.query)
    return {
        "query": req.query,
        "hypothetical_passage": passage,
        "is_fallback": passage.strip() == req.query.strip(),
        "stats": generator.get_stats()
    }

def start():
    """Entrypoint to launch uvicorn directly via 'uv run backend' or Render."""
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=settings.DEBUG)

if __name__ == "__main__":
    start()

