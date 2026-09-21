import re

with open("index.html", "r") as f:
    content = f.read()

animate_patch = """
        // --- Radial Camera Lock & Stat Box Update ---
        const radialLockCheckbox = document.getElementById('camera-radial-lock');
        const statBox = document.getElementById('floating-stat-box');
        let statBoxActive = false;

        if (selectedSatIndex >= 0 && activeSatellites[selectedSatIndex]) {
            const sat = activeSatellites[selectedSatIndex];
            
            // 3D Stat Box Placement
            statBoxActive = true;
            statBox.style.display = 'block';
            document.getElementById('fs-name').innerText = sat.name || 'Sat';
            document.getElementById('fs-alt').innerText = document.getElementById('t-alt').innerText;
            document.getElementById('fs-vel').innerText = document.getElementById('t-vel').innerText;
            document.getElementById('fs-inc').innerText = document.getElementById('t-inc').innerText;
            
            // Project 3D pos to 2D screen
            const tempV = sat.mesh.position.clone();
            tempV.project(camera);
            const rect = renderer.domElement.getBoundingClientRect();
            const x = (tempV.x *  .5 + .5) * rect.width;
            const y = (tempV.y * -.5 + .5) * rect.height;
            statBox.style.left = `${x + 15}px`;
            statBox.style.top = `${y - 15}px`;

            // Radial Lock Camera Adjustment
            if (radialLockCheckbox && radialLockCheckbox.checked && controls) {
                // Vector from planet center to sat
                const planetPos = sat.parentBody === 'Moon' ? moon.position : new THREE.Vector3(0,0,0);
                const radialDir = new THREE.Vector3().subVectors(sat.mesh.position, planetPos).normalize();
                
                // Determine current distance to maintain it
                const dist = camera.position.distanceTo(sat.mesh.position);
                const targetCamPos = sat.mesh.position.clone().add(radialDir.multiplyScalar(dist));
                
                // Lerp for smoothness instead of harsh snapping
                camera.position.lerp(targetCamPos, 0.05);
            }
        } else {
            statBox.style.display = 'none';
        }
        // --------------------------------------------
"""

# Insert this block right after the camera focus modes block ends in animate()
content = content.replace("updateTelemetry(null, 0);\n        }", "updateTelemetry(null, 0);\n        }\n" + animate_patch)

with open("index.html", "w") as f:
    f.write(content)
