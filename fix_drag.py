import re

with open("visualization_3d/static/index.html", "r") as f:
    content = f.read()

target = "const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), new THREE.Vector3(0,0,0));"
replacement = "const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()).negate(), dragOrbitStartPos);"
content = content.replace(target, replacement)

# Let's also fix the maneuver handle drag sensitivity and logic
# Pushing mouse up/right increases, down/left decreases
target_drag = "const delta = (-dy + dx) * 0.5; // Sensitivity factor"
replacement_drag = "const delta = (-dy + dx) * 2.0; // Increased sensitivity"
content = content.replace(target_drag, replacement_drag)

with open("visualization_3d/static/index.html", "w") as f:
    f.write(content)
