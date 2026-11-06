# OrbitGuard operator notes

OrbitGuard is the LEO monitoring workspace next to NanoDynamics. The 3D
simulator, its numerical propagator, and its existing routes stay in
place. This page is a first version: it shows what was retrieved, what
was derived, and what is not known. It does not command a spacecraft.

## What is running

1. **Ingest.** One public CelesTrak request for the `stations` group, as
   OMM JSON:
   `https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=JSON`.
   The client caches the batch and does not ask again for two hours,
   which is the update interval CelesTrak documents. HTTP 301, 403, and
   404 are not retried. Records are validated, hashed, and stored once.
   The same hash is a duplicate. The same catalog number and epoch with
   a different hash is a conflict and is not overwritten. Fields that
   look like credentials are rejected. Legacy TLE lines can be checked
   for a checksum and for a strict Alpha-5 catalog field (`I` and `O`
   are rejected; the ceiling is `Z9999` = 339999).
2. **Propagation of catalog elements.** Mean elements go through SGP4
   with the WGS-72 constants and the improved operation mode. Output is
   TEME. They are not handed to the numerical integrator.
3. **Display transform.** TEME is rotated by Greenwich mean sidereal
   time into an Earth-fixed display frame, then converted to WGS-84
   geodetic coordinates. UTC is used in place of UT1. Polar motion and
   the equation of equinoxes are not applied, so the frame is not ITRF.
4. **Numerical simulation.** `leo_simulator.OrbitPropagator` runs only
   when the caller labels the state `osculating_cartesian` in the
   `nanodynamics_eci` frame. A regression test compares a drag-free,
   J2-free circular orbit with the closed-form Keplerian position.
   A separate check compares the sign of nodal drift with the first-order
   J2 secular rate. Osculating elements are not that secular rate.
   Neither check is a PINN accuracy claim.
5. **SGP4 wiring check.** The epoch state of the standard verification
   element (catalog 00005 in the python-sgp4 `tcppver.out` vector) matches
   that published TEME coordinate. That element is not a current object
   and the match is not an operational error budget.
6. **Environment.** When the download succeeds, F10.7 comes from
   `https://services.swpc.noaa.gov/json/f107_cm_flux.json` (solar flux
   units, 2800 MHz) and planetary K from
   `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json`
   (dimensionless). On 2026-11-06,
   `https://services.swpc.noaa.gov/json/planetary_k_index.json` returned
   HTTP 404, so OrbitGuard does not use it. TEC and thermospheric density
   are left empty.
7. **Geometry.** A screen reports the closest sampled separation in one
   frame. It does not report a collision probability.

## What is not running

- No 2D Pc, no Hall 3D Nc, and no Monte Carlo collision count.
- Nc, as defined by Hall (Journal of Spacecraft and Rockets, 2021,
  https://doi.org/10.2514/1.A34919), is an expected number of collisions.
  The paper states that Nc equals Pc for an isolated encounter under its
  assumptions, and that Nc can exceed Pc when encounters repeat. OrbitGuard
  does not compute Nc and does not relabel it as Pc. 3D Nc is not a
  fallback for a missing 2D input, and it does not repair a bad covariance.
  The NASA software record is https://software.nasa.gov/software/GSC-18593-1.
  The CARA briefing is
  https://ntrs.nasa.gov/api/citations/20205007226/downloads/CARA_3D-NC%20and%20BFMC%20Presentation_ATrainMOWG_Fall2020_Final.pdf.
  The conjunction-assessment handbook appendix distributed by NTRS is
  https://ntrs.nasa.gov/api/citations/20240003468/downloads/CA_Hanbook_Appendix.pdf.
- No covariance propagation. A supplied matrix is checked for shape,
  finite values, symmetry, positive-semidefiniteness, epoch, and frame.
  There is no state-transition matrix and no process noise. Public GP
  has no calibrated covariance. Absence is not shown as zero or safe.
- No uncertainty ellipsoid or Gaussian splat. A drawing would not be a
  probability.
- NRLMSISE-00 (https://ccmc.gsfc.nasa.gov/models/NRLMSIS~00/) and JB2008
  (https://ccmc.gsfc.nasa.gov/models/JB2008~2008/) are not evaluated.
- No GNSS TEC product and no in-situ plasma product. TEC is an integrated
  column (TECU = 10^16 electrons m^-2), not a local density. Kp and F10.7
  are not turned into either quantity.
- No optical, SLR, or lidar feed. Optical data would be lines of sight
  and brightness, not ranges. Cooperative satellite laser ranging and
  uncooperative debris ranging are different, and both need an external
  network (see https://ilrs.gsfc.nasa.gov/lw19/docs/2014/Papers/3009_Kirchner_paper.pdf).
  Debris-ranging distance and accuracy depend on the station and the
  target; none is stated here. A conventional lidar does not map LEO.
  Transmitting a laser would require the aviation and laser-safety
  approvals that apply at the site, and a partner who can perform the
  ranging. This software does not treat a U.S. Laser Clearinghouse
  process as the rule for work in South Africa, and it does not point
  or fire a laser.
- Space-Track (https://www.space-track.org/documentation) is not called.
  If `SPACETRACK_USER` and `SPACETRACK_PASSWORD` exist in the server
  environment, the status flag says only that they are configured. The
  values are not returned. This preview has no account and no login.
- No maneuver solver and no command path.

## Ages

Element epoch age and retrieval age are different. Epoch age is the age
of the mean elements. Retrieval age is the time since this process last
stored a successful download. The default labels use 72 hours for the
epoch and 6 hours for the retrieval. Both numbers are display settings
for this workspace. CelesTrak's own "oldest" table flag (3.5 days for
active satellites) is a data-review aid, not a safety rule, and OrbitGuard
does not adopt it as one. A state inside the display window is not a
statement that the orbit is safe.

## Sample mode

The simulation sample is two synthetic element sets named
`OG-SAMPLE-LEO-A` and `OG-SAMPLE-LEO-B`. Their integers exist only so
SGP4 can start. They are not NORAD identities. States are shown at the
element epoch (2024-06-01), not slid forward to today and labelled live.
The banner and every object row say simulation sample.

## Frames and constants

| Quantity | Value used here |
| --- | --- |
| Catalog theory | SGP4, WGS-72 constants, TEME, UTC |
| Display | TEME rotated by GMST; WGS-84 geodetic |
| Numerical frame | `nanodynamics_eci` (J2 along that model's z axis) |
| GMST at 2000-01-01 12:00 UTC | 280.46061837 degrees |

GP JSON omits `CENTER_NAME`, `REF_FRAME`, `TIME_SYSTEM`, and
`MEAN_ELEMENT_THEORY`. CelesTrak's formats page states the omitted
values as EARTH, TEME, UTC, and SGP4:
https://celestrak.org/NORAD/documentation/gp-data-formats.php.

## Interfaces

Passive ingest, normalization, SGP4 propagation, numerical propagation,
environment, geometry, and the page are separate modules under
`orbitguard/monitor/`. The HTTP surface is `/api/orbitguard/`. The older
copilot package under `orbitguard/physics` and `ai/orbitguard.py` is not
used by this workspace and is not a validated probability service.
