from fastapi import APIRouter, HTTPException

from api.routes import recommendations

router = APIRouter(prefix="/api/health", tags=["Health"])


@router.get("/")
def health_check() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "healthy"}


@router.get("/ready")
def readiness_check() -> dict[str, str]:
    """Readiness: the index is loaded and the graph is compiled."""
    if recommendations._recommend is None:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready"}
