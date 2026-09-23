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
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
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
    """Entrypoint to launch uvicorn directly via 'uv run backend'."""
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()

