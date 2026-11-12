"""External product links.

``ORBITGUARD_URL`` is the only address for the OrbitGuard site. It is read
from the environment and is empty until that site has a real URL. Callers
must hide the link when this value is empty. Do not substitute a placeholder.
"""

import os

ORBITGUARD_URL = os.environ.get("ORBITGUARD_URL", "").strip()
