"""Database engine and session management for OrbitGuard AI with graceful memory fallback."""

from collections.abc import AsyncGenerator
import logging
from typing import Any, Optional

from orbitguard.config import settings

logger = logging.getLogger(__name__)

try:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from orbitguard.db.models import Base, Satellite, Tenant

    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        future=True,
    )

    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    HAS_SQLALCHEMY = True
except ImportError:
    engine = None
    AsyncSessionLocal = None
    HAS_SQLALCHEMY = False
    logger.info("SQLAlchemy not installed; operating in zero-dependency in-memory mode.")


# In-memory store fallback for environments without SQLAlchemy
IN_MEMORY_STORE = {
    "tenants": [
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Stanford University Space Rendezvous Lab",
            "slug": "stanford-srl",
            "tier": "academic",
            "contact_email": "cubesat-ops@stanford.edu",
            "telegram_chat_id": settings.DEFAULT_TELEGRAM_CHAT_ID,
            "webhook_url": "http://127.0.0.1:8045/api/v1/alerts/webhook-demo",
        }
    ],
    "satellites": [
        {
            "id": "00000000-0000-0000-0000-000000000002",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "norad_cat_id": 43015,
            "name": "AeroCube-3U (Stanford)",
            "international_designator": "2017-071F",
            "sat_class": "cubesat_3u",
            "dry_mass_kg": 4.0,
            "fuel_mass_kg": 0.5,
            "cross_section_area_m2": 0.03,
            "cd_drag_coefficient": 2.2,
            "hard_body_radius_m": 0.5,
            "thruster_type": "cold_gas",
            "thrust_n": 0.050,
            "isp_s": 70.0,
            "status": "operational",
            "created_at": "2026-11-01T00:00:00Z",
        }
    ],
    "conjunctions": [],
    "maneuvers": [],
    "alerts": [],
}

_db_initialized = False


async def init_db() -> None:
    """Initialize database tables or verify in-memory store."""
    global _db_initialized
    if HAS_SQLALCHEMY and engine is not None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            res = await session.execute(select(Tenant).limit(1))
            existing_tenant = res.scalar_one_or_none()

            if not existing_tenant:
                default_tenant = Tenant(
                    id="00000000-0000-0000-0000-000000000001",
                    name="Stanford University Space Rendezvous Lab",
                    slug="stanford-srl",
                    api_key_hash="og_live_academic_token_hash_stanford",
                    tier="academic",
                    contact_email="cubesat-ops@stanford.edu",
                    telegram_chat_id=settings.DEFAULT_TELEGRAM_CHAT_ID,
                    webhook_url="http://127.0.0.1:8045/api/v1/alerts/webhook-demo",
                )
                session.add(default_tenant)

                cubesat = Satellite(
                    id="00000000-0000-0000-0000-000000000002",
                    tenant_id=default_tenant.id,
                    norad_cat_id=43015,
                    name="AeroCube-3U (Stanford)",
                    international_designator="2017-071F",
                    sat_class="cubesat_3u",
                    dry_mass_kg=4.0,
                    fuel_mass_kg=0.5,
                    cross_section_area_m2=0.03,
                    cd_drag_coefficient=2.2,
                    hard_body_radius_m=0.5,
                    thruster_type="cold_gas",
                    thrust_n=0.050,
                    isp_s=70.0,
                    status="operational",
                )
                session.add(cubesat)
                await session.commit()
                logger.info("Database initialized and seeded.")
    else:
        logger.info("In-memory store ready with default Stanford AeroCube-3U.")
    _db_initialized = True


async def get_db() -> AsyncGenerator[Any, None]:
    """Dependency that provides database session or memory fallback."""
    global _db_initialized
    if not _db_initialized:
        await init_db()

    if HAS_SQLALCHEMY and AsyncSessionLocal is not None:
        async with AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    else:
        yield IN_MEMORY_STORE
