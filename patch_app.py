import re

with open("visualization_3d/app.py", "r") as f:
    content = f.read()

# 1. Add fields to SatelliteParams
target_fields = """    raan_deg: float = Field(..., ge=0.0, lt=360.0, description="RAAN in degrees")
    # Atmosphere configuration"""
replace_fields = """    raan_deg: float = Field(..., ge=0.0, lt=360.0, description="RAAN in degrees")
    arg_periapsis_deg: float = Field(0.0, ge=0.0, lt=360.0, description="Argument of Periapsis in degrees")
    true_anomaly_deg: float = Field(0.0, ge=0.0, lt=360.0, description="True Anomaly (Phase) in degrees")
    # Atmosphere configuration"""
content = content.replace(target_fields, replace_fields)

# 2. Update OrbitalElements instantiation
target_inst = """        orbit = OrbitalElements(
            a=a_m,
            e=params.eccentricity,
            i=np.radians(params.inclination_deg),
            raan=np.radians(params.raan_deg),
            arg_pe=0.0,
            nu=0.0,
        )"""
replace_inst = """        orbit = OrbitalElements(
            a=a_m,
            e=params.eccentricity,
            i=np.radians(params.inclination_deg),
            raan=np.radians(params.raan_deg),
            arg_pe=np.radians(params.arg_periapsis_deg),
            nu=np.radians(params.true_anomaly_deg),
        )"""
content = content.replace(target_inst, replace_inst)

with open("visualization_3d/app.py", "w") as f:
    f.write(content)
