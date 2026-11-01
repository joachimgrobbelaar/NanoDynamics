"""Pydantic schemas for OrbitGuard AI API validation and serialization."""

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SatelliteBase(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "AeroCube-3U"})
    norad_cat_id: int = Field(..., json_schema_extra={"example": 43015})
    international_designator: Optional[str] = Field("2017-071F", json_schema_extra={"example": "2017-071F"})
    sat_class: str = Field("cubesat_3u", json_schema_extra={"example": "cubesat_3u"})
    dry_mass_kg: float = Field(4.0, json_schema_extra={"example": 4.0})
    fuel_mass_kg: float = Field(0.5, json_schema_extra={"example": 0.5})
    cross_section_area_m2: float = Field(0.03, json_schema_extra={"example": 0.03})
    cd_drag_coefficient: float = Field(2.2, json_schema_extra={"example": 2.2})
    hard_body_radius_m: float = Field(0.5, json_schema_extra={"example": 0.5})
    thruster_type: str = Field("cold_gas", json_schema_extra={"example": "cold_gas"})
    thrust_n: float = Field(0.050, json_schema_extra={"example": 0.050})
    isp_s: float = Field(70.0, json_schema_extra={"example": 70.0})


class SatelliteCreate(SatelliteBase):
    tenant_id: str


class SatelliteResponse(SatelliteBase):
    id: str
    tenant_id: str
    status: str
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class ConjunctionResponse(BaseModel):
    id: str
    tenant_id: str
    primary_sat_id: str
    secondary_norad_id: int
    secondary_object_name: str
    tca_epoch: datetime.datetime
    miss_distance_km: float
    radial_miss_km: Optional[float]
    in_track_miss_km: Optional[float]
    cross_track_miss_km: Optional[float]
    relative_speed_km_s: float
    collision_probability: float
    risk_tier: str
    status: str

    model_config = ConfigDict(from_attributes=True)


class ManeuverResponse(BaseModel):
    id: str
    conjunction_id: str
    satellite_id: str
    maneuver_strategy: str
    planned_burn_epoch: datetime.datetime
    lead_time_hours: float
    dv_velocity_v_m_s: float
    dv_normal_n_m_s: float
    dv_conormal_c_m_s: float
    dv_total_magnitude_m_s: float
    burn_duration_s: float
    propellant_consumed_kg: float
    remaining_fuel_kg: float
    remaining_delta_v_m_s: float
    post_maneuver_miss_distance_km: float
    post_maneuver_collision_prob: float
    new_perigee_altitude_km: float
    new_apogee_altitude_km: float
    executive_brief_markdown: str
    recommendation_status: str

    model_config = ConfigDict(from_attributes=True)


class AlertResponse(BaseModel):
    id: str
    conjunction_id: str
    channel: str
    recipient_target: str
    dispatch_status: str
    delivered_at: Optional[datetime.datetime]
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class ScreeningTriggerRequest(BaseModel):
    tenant_id: Optional[str] = None
    horizon_hours: float = 72.0


class ThreatSimulationRequest(BaseModel):
    satellite_name: str = "AeroCube-3U"
    threat_debris_name: str = "COSMOS-2251 DEBRIS [CAT# 34123]"
    miss_distance_target_km: float = 0.85
    tca_lead_hours: float = 18.5
