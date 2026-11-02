"""FastAPI application entrypoint for Nanodynamics Astrodynamics & OrbitGuard AI Engine."""

from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from api.routers.conjunction import router as conjunction_router
from api.routers.health import router as health_router
from api.routers.simulation import router as simulation_router

app = FastAPI(
    title="Nanodynamics & OrbitGuard AI",
    description="High-precision orbital mechanics propagation, perturbation simulation, and AI collision avoidance system.",
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable permissive CORS for cross-domain orchestration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://nanodynamics.co.za",
        "https://www.nanodynamics.co.za",
        "https://api.nanodynamics.co.za",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core sub-routers
app.include_router(health_router)
app.include_router(simulation_router)
app.include_router(conjunction_router)


@app.get("/", response_class=JSONResponse)
async def root():
    """Root platform discovery endpoint."""
    return {
        "platform": "Nanodynamics & OrbitGuard AI",
        "version": "2.4.0",
        "documentation": "/docs",
        "health_check": "/api/health",
        "modules": {
            "simulation": "/api/simulation",
            "conjunction": "/api/conjunction",
            "health": "/api/health",
        },
        "endpoints": {
            "propagate_numerical": "POST /api/simulation/propagate",
            "propagate_tle": "POST /api/simulation/propagate-tle",
            "screen_conjunction": "POST /api/conjunction/screen",
            "evaluate_risk": "POST /api/conjunction/evaluate",
            "generate_cdm": "POST /api/conjunction/cdm",
            "avoidance_maneuver": "POST /api/conjunction/maneuver",
        },
    }
