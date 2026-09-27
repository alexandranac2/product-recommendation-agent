import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings

# Import route modules
from api.routes import recommendations, health, routes_list


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store and compile the graph once, before serving traffic."""
    recommendations.initialize_recommendation_system()
    yield


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
)

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Public, read-only API with no cookies or auth, so credentials stay off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(recommendations.router, tags=["Recommendations"])
app.include_router(routes_list.router, tags=["Routes"])


@app.get("/")
async def root():
    return {
        "message": "Product Recommendation System API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "routes": "/api/routes"
    }

