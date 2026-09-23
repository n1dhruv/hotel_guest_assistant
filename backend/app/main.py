from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.chat import router as chat_router
from app.api.availability import router as availability_router
from app.api.stats import router as stats_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Full-stack AI-powered guest assistant backend for The Grand Azure Heritage Resort & Spa (Candolim, Goa)",
    version="1.0.0"
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

def start():
    """Entrypoint to launch uvicorn directly via 'uv run backend' or Render."""
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=settings.DEBUG)

if __name__ == "__main__":
    start()

