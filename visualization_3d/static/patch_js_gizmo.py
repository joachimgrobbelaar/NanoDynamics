import re

with open("index.html", "r") as f:
    content = f.read()

gizmo_patch = """
    // --- Maneuver Handles (KSP Style) ---
    const handleGeo = new THREE.SphereGeometry(60, 16, 16);
    const handleMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.2 });
    
    const handlePrograde = new THREE.Mesh(handleGeo, handleMat);
    handlePrograde.position.set(300, 0, 0);
    handlePrograde.userData = { axis: 'prograde', color: 0x4ade80 };
    
    const handleRadial = new THREE.Mesh(handleGeo, handleMat);
    handleRadial.position.set(0, 300, 0);
    handleRadial.userData = { axis: 'radial', color: 0x38bdf8 };
    
    const handleNormal = new THREE.Mesh(handleGeo, handleMat);
    handleNormal.position.set(0, 0, 300);
    handleNormal.userData = { axis: 'normal', color: 0xe879f9 };
    
    nodeGizmoGroup.add(handlePrograde);
    nodeGizmoGroup.add(handleRadial);
    nodeGizmoGroup.add(handleNormal);
    
    const maneuverHandles = [handlePrograde, handleRadial, handleNormal];
"""

content = content.replace("nodeGizmoGroup.add(arrowNormal);", "nodeGizmoGroup.add(arrowNormal);\n" + gizmo_patch)

with open("index.html", "w") as f:
    f.write(content)
