from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.routing import APIRoute

router = APIRouter(prefix="/api/routes", tags=["Routes"])


def get_route_info(app: FastAPI) -> list[dict[str, str]]:
    """Extract route information from FastAPI app"""
    routes_info = []

    for route in app.routes:
        if isinstance(route, APIRoute):
            methods = ", ".join(route.methods)
            routes_info.append(
                {
                    "path": route.path,
                    "methods": methods,
                    "name": route.name,
                    "summary": route.summary or "No description",
                    "tags": ", ".join(route.tags) if route.tags else "No tags",
                }
            )

    return routes_info


@router.get("/")
async def list_routes(request: Request) -> dict[str, Any]:
    """
    List all available API routes and their details.

    Returns information about each endpoint including:
    - Path
    - HTTP methods
    - Description
    - Tags
    """
    app: FastAPI = request.app
    routes = get_route_info(app)

    return {"total_routes": len(routes), "routes": routes, "base_url": str(request.base_url).rstrip("/")}


@router.get("/summary")
async def routes_summary(request: Request) -> dict[str, Any]:
    """Get a summary of all routes grouped by tags"""
    app: FastAPI = request.app
    routes = get_route_info(app)

    # Group by tags
    grouped = {}
    for route in routes:
        tag = route["tags"] if route["tags"] != "No tags" else "default"
        if tag not in grouped:
            grouped[tag] = []
        grouped[tag].append({"path": route["path"], "methods": route["methods"], "summary": route["summary"]})

    return {"total_routes": len(routes), "routes_by_tag": grouped}
