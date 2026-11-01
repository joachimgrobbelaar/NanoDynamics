"""SQLAlchemy ORM models mirroring the OrbitGuard PostgreSQL / Supabase schema."""

import datetime
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    tier: Mapped[str] = mapped_column(String(50), default="academic")
    contact_email: Mapped[str] = mapped_column(String(255), nullable=False)
    webhook_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    webhook_secret: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    telegram_chat_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    settings: Mapped[Dict[str, Any]] = mapped_column(SQLiteJSON, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    satellites = relationship("Satellite", back_populates="tenant", cascade="all, delete-orphan")


class Satellite(Base):
    __tablename__ = "satellites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    norad_cat_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    international_designator: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sat_class: Mapped[str] = mapped_column(String(50), default="cubesat_3u")

    dry_mass_kg: Mapped[float] = mapped_column(Float, default=4.0)
    fuel_mass_kg: Mapped[float] = mapped_column(Float, default=0.5)
    cross_section_area_m2: Mapped[float] = mapped_column(Float, default=0.03)
    cd_drag_coefficient: Mapped[float] = mapped_column(Float, default=2.2)
    hard_body_radius_m: Mapped[float] = mapped_column(Float, default=0.5)

    thruster_type: Mapped[str] = mapped_column(String(50), default="cold_gas")
    thrust_n: Mapped[float] = mapped_column(Float, default=0.050)
    isp_s: Mapped[float] = mapped_column(Float, default=70.0)

    status: Mapped[str] = mapped_column(String(50), default="operational")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    tenant = relationship("Tenant", back_populates="satellites")
    conjunctions = relationship("ConjunctionScreening", back_populates="primary_satellite", cascade="all, delete-orphan")


class TLERecord(Base):
    __tablename__ = "tle_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    norad_cat_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    satellite_name: Mapped[str] = mapped_column(String(150), nullable=False)
    line1: Mapped[str] = mapped_column(String(70), nullable=False)
    line2: Mapped[str] = mapped_column(String(70), nullable=False)
    epoch_timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    bstar: Mapped[float] = mapped_column(Float, default=0.0)
    inclination_deg: Mapped[float] = mapped_column(Float, nullable=False)
    raan_deg: Mapped[float] = mapped_column(Float, nullable=False)
    eccentricity: Mapped[float] = mapped_column(Float, nullable=False)
    arg_perigee_deg: Mapped[float] = mapped_column(Float, nullable=False)
    mean_anomaly_deg: Mapped[float] = mapped_column(Float, nullable=False)
    mean_motion_rev_day: Mapped[float] = mapped_column(Float, nullable=False)
    revolution_number: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(50), default="celestrak")
    ingested_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))


class DebrisCatalog(Base):
    __tablename__ = "debris_catalog"

    norad_cat_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    rcs_size: Mapped[str] = mapped_column(String(20), default="MEDIUM")
    hard_body_radius_m: Mapped[float] = mapped_column(Float, default=1.0)
    country_code: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")


class ConjunctionScreening(Base):
    __tablename__ = "conjunction_screenings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    primary_sat_id: Mapped[str] = mapped_column(String(36), ForeignKey("satellites.id", ondelete="CASCADE"), nullable=False)
    secondary_norad_id: Mapped[int] = mapped_column(Integer, nullable=False)
    secondary_object_name: Mapped[str] = mapped_column(String(150), nullable=False)

    screening_window_start: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    screening_window_end: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    screening_window_hours: Mapped[int] = mapped_column(Integer, default=72)
    propagation_method: Mapped[str] = mapped_column(String(50), default="J2_drag_numerical")

    tca_epoch: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False, index=True)
    miss_distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    radial_miss_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    in_track_miss_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cross_track_miss_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relative_speed_km_s: Mapped[float] = mapped_column(Float, nullable=False)

    combined_hard_body_radius_m: Mapped[float] = mapped_column(Float, default=1.5)
    collision_probability: Mapped[float] = mapped_column(Float, default=0.0)
    risk_tier: Mapped[str] = mapped_column(String(20), default="LOW")
    status: Mapped[str] = mapped_column(String(30), default="screened")

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    primary_satellite = relationship("Satellite", back_populates="conjunctions")
    maneuvers = relationship("ManeuverRecommendation", back_populates="conjunction", cascade="all, delete-orphan")


class ManeuverRecommendation(Base):
    __tablename__ = "maneuver_recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conjunction_id: Mapped[str] = mapped_column(String(36), ForeignKey("conjunction_screenings.id", ondelete="CASCADE"), nullable=False)
    satellite_id: Mapped[str] = mapped_column(String(36), ForeignKey("satellites.id", ondelete="CASCADE"), nullable=False)

    maneuver_strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    planned_burn_epoch: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    lead_time_hours: Mapped[float] = mapped_column(Float, nullable=False)

    dv_velocity_v_m_s: Mapped[float] = mapped_column(Float, default=0.0)
    dv_normal_n_m_s: Mapped[float] = mapped_column(Float, default=0.0)
    dv_conormal_c_m_s: Mapped[float] = mapped_column(Float, default=0.0)
    dv_total_magnitude_m_s: Mapped[float] = mapped_column(Float, nullable=False)

    burn_duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    propellant_consumed_kg: Mapped[float] = mapped_column(Float, nullable=False)
    remaining_fuel_kg: Mapped[float] = mapped_column(Float, nullable=False)
    remaining_delta_v_m_s: Mapped[float] = mapped_column(Float, nullable=False)

    post_maneuver_miss_distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    post_maneuver_collision_prob: Mapped[float] = mapped_column(Float, default=0.0)
    new_perigee_altitude_km: Mapped[float] = mapped_column(Float, nullable=False)
    new_apogee_altitude_km: Mapped[float] = mapped_column(Float, nullable=False)

    executive_brief_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation_status: Mapped[str] = mapped_column(String(30), default="recommended")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    conjunction = relationship("ConjunctionScreening", back_populates="maneuvers")


class AlertDispatch(Base):
    __tablename__ = "alert_dispatches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conjunction_id: Mapped[str] = mapped_column(String(36), ForeignKey("conjunction_screenings.id", ondelete="CASCADE"), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    channel: Mapped[str] = mapped_column(String(30), nullable=False)
    recipient_target: Mapped[str] = mapped_column(String(255), nullable=False)
    payload_json: Mapped[Dict[str, Any]] = mapped_column(SQLiteJSON, default=dict)
    dispatch_status: Mapped[str] = mapped_column(String(20), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    delivered_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(100), default="orbitguard_autonomous_agent")
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    details: Mapped[Dict[str, Any]] = mapped_column(SQLiteJSON, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
