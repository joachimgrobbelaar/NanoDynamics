"""OrbitGuard AI — Main FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from orbitguard.api.routers import alerts, conjunctions, maneuvers, satellites, simulation
from orbitguard.config import settings
from orbitguard.db.session import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orbitguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info("Initializing OrbitGuard AI Database and Ephemeris Engine...")
    await init_db()
    logger.info("OrbitGuard AI Core Engine online on port %d", settings.PORT)
    yield
    logger.info("OrbitGuard AI shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous Orbital Conjunction Screening & Maneuver Recommendation Copilot for CubeSats & Small Satellites.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(satellites.router, prefix="/api/v1")
app.include_router(conjunctions.router, prefix="/api/v1")
app.include_router(maneuvers.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")
app.include_router(simulation.router, prefix="/api/v1")


@app.get("/api/v1/health")
async def health_check():
    """System health and astrodynamics engine telemetry."""
    return {
        "status": "healthy",
        "service": "OrbitGuard AI",
        "version": settings.VERSION,
        "astrodynamics_engine": {
            "integrator": "Runge-Kutta 4(5) Cowell",
            "perturbations": ["J2 Earth Oblateness", "Exponential Atmospheric Drag"],
            "screening_horizon_hours": settings.SCREENING_WINDOW_HOURS,
            "conjunction_trigger_km": settings.CONJUNCTION_THRESHOLD_KM,
            "maneuver_solver": "VNC Phasing & Out-of-Plane Tsiolkovsky Engine",
        },
    }


@app.get("/")
async def root():
    """Mission Control overview and quick-start links."""
    return {
        "message": "OrbitGuard AI Mission Control Copilot is active.",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "satellites": "/api/v1/satellites",
            "conjunctions": "/api/v1/conjunctions",
            "screen_horizon": "/api/v1/conjunctions/screen",
            "maneuvers": "/api/v1/maneuvers",
            "alerts": "/api/v1/alerts",
            "simulate_threat": "/api/v1/demo/simulate-threat",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orbitguard.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
