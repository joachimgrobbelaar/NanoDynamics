"""Sensor classes OrbitGuard can describe. None of them are connected.

Optical angles are not ranges. Cooperative satellite laser ranging and
uncooperative debris laser ranging are different, and both need an
external network. A conventional lidar does not map the LEO environment.
"""

from __future__ import annotations

from typing import Any


def sensor_catalog() -> dict[str, Any]:
    disconnected = "disconnected"
    return {
        "feed_status": disconnected,
        "mode_label": "No sensor feed is connected. Rows describe limits, not observations.",
        "tasking": (
            "This workspace does not identify surveillance targets and does not "
            "task, point, or fire a sensor or a laser."
        ),
        "sensors": [
            {
                "id": "optical_track",
                "name": "Optical angles and brightness",
                "status": disconnected,
                "measures": "Line of sight and brightness, when a telescope feed exists.",
                "does_not_measure": "Slant range. An optical detection is not a measured distance.",
                "current_values": None,
            },
            {
                "id": "cooperative_slr",
                "name": "Cooperative satellite laser ranging",
                "status": disconnected,
                "measures": "Two-way range to a satellite carrying retroreflectors, from a qualified station.",
                "does_not_measure": "Uncooperative debris. ILRS participation is an external network, not a local lidar sweep.",
                "references": ["https://ilrs.gsfc.nasa.gov/lw19/docs/2014/Papers/3009_Kirchner_paper.pdf"],
                "current_values": None,
            },
            {
                "id": "debris_laser_ranging",
                "name": "Uncooperative debris laser ranging",
                "status": "not_integrated",
                "measures": "Range to a non-cooperative object only with specialized equipment and a suitable target.",
                "does_not_measure": (
                    "A universal LEO debris map. Range and accuracy depend on the "
                    "station, the laser, the atmosphere, and the target. No single "
                    "distance or accuracy is stated here."
                ),
                "approvals": (
                    "Any real transmit activity needs the aviation and laser-safety "
                    "approvals that apply at the site, plus a partner who can actually "
                    "range the object. A U.S. Laser Clearinghouse process is not "
                    "treated as the rule for South African operations."
                ),
                "current_values": None,
            },
            {
                "id": "conventional_lidar",
                "name": "Conventional lidar",
                "status": disconnected,
                "measures": "Nothing in this build.",
                "does_not_measure": "The LEO environment as a whole. A mapping lidar is not an orbital catalog.",
                "current_values": None,
            },
            {
                "id": "gnss_tec",
                "name": "GNSS total electron content",
                "status": disconnected,
                "measures": "Integrated electron column along a signal path, in TECU, when a TEC product is connected.",
                "does_not_measure": "Local 3D plasma density.",
                "current_values": None,
            },
            {
                "id": "in_situ_plasma",
                "name": "In-situ plasma instrument",
                "status": disconnected,
                "measures": "Local charged-particle properties at a vehicle, when that vehicle's product is ingested.",
                "does_not_measure": "A column TEC value, and not thermospheric neutral density.",
                "current_values": None,
            },
        ],
    }
