import re

with open("index.html", "r") as f:
    content = f.read()

# 1. HTML Patches
content = content.replace(
    '<optgroup id="sat-focus-group" label="Satellites"></optgroup>\n                </select>\n            </div>',
    '<optgroup id="sat-focus-group" label="Satellites"></optgroup>\n                </select>\n            </div>\n            <div class="row" style="margin-top:4px;">\n                <label style="font-size:10px; color:#94a3b8;"><input type="checkbox" id="camera-radial-lock" checked> Radial View Lock (Inward)</label>\n            </div>'
)

stat_box = """
    <!-- Floating 3D Stat Box -->
    <div id="floating-stat-box" style="display:none; position:absolute; pointer-events:none; background:rgba(15,23,42,0.85); border:1px solid #3b82f6; border-radius:6px; padding:8px; color:white; font-family:monospace; font-size:11px; z-index:100; text-shadow:0 1px 2px black; box-shadow:0 4px 6px rgba(0,0,0,0.5); backdrop-filter:blur(2px);">
        <div style="color:#60a5fa; font-weight:bold; margin-bottom:4px; font-size:12px; border-bottom:1px solid #3b82f6; padding-bottom:2px;" id="fs-name">Sat-Name</div>
        <div>Alt: <span id="fs-alt" style="color:#fcd34d;">-- km</span></div>
        <div>Vel: <span id="fs-vel" style="color:#a7f3d0;">-- km/s</span></div>
        <div>Inc: <span id="fs-inc" style="color:#c4b5fd;">-- °</span></div>
    </div>
"""
content = content.replace('<div id="time-hud">', stat_box + '\n    <div id="time-hud">')

drag_btn = '<button type="button" class="btn" style="background:#8b5cf6;" onclick="toggleDragOrbitMode()">✨ Drag to Add Orbit</button>\n                '
content = content.replace('<button type="button" class="btn" onclick="openSatModal()">+ Add Satellite</button>', drag_btn + '<button type="button" class="btn" onclick="openSatModal()">+ Add Satellite</button>')

# 2. Gizmo Patch
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

# 3. Interactions Patch
interaction_code = """
    // ── INTERACTIVE UX: DRAG-TO-ORBIT & MANEUVER NODES ───────────────────────
    
    // UI State
    let dragOrbitMode = false;
    let dragOrbitStartPos = null;
    let dragOrbitPreviewLine = null;
    
    // Maneuver Gizmo Drag State
    const uxRaycaster = new THREE.Raycaster();
    const uxMouse = new THREE.Vector2();
    let draggingHandle = null;
    let dragStartMouse = new THREE.Vector2();
    let dragStartValue = 0;
    
    window.toggleDragOrbitMode = function() {
        dragOrbitMode = !dragOrbitMode;
        if (dragOrbitMode) {
            document.body.style.cursor = 'crosshair';
            alert('Click on Earth and drag outwards to define an orbit. Release to add satellite.');
        } else {
            document.body.style.cursor = 'default';
            if (dragOrbitPreviewLine) {
                scene.remove(dragOrbitPreviewLine);
                dragOrbitPreviewLine.geometry.dispose();
                dragOrbitPreviewLine.material.dispose();
                dragOrbitPreviewLine = null;
            }
        }
    };
    
    function onPointerDown(event) {
        if (event.button !== 0) return; // Only left click
        
        // Raycast setup
        const rect = renderer.domElement.getBoundingClientRect();
        uxMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        uxMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        uxRaycaster.setFromCamera(uxMouse, camera);
        
        // 1. Maneuver Handle Dragging
        if (nodeGizmoGroup.visible && typeof maneuverHandles !== 'undefined') {
            const intersects = uxRaycaster.intersectObjects(maneuverHandles);
            if (intersects.length > 0) {
                controls.enabled = false; // Disable camera orbit
                draggingHandle = intersects[0].object;
                dragStartMouse.set(event.clientX, event.clientY);
                
                const axis = draggingHandle.userData.axis;
                dragStartValue = parseFloat(document.getElementById(`burn-${axis}`).value) || 0;
                draggingHandle.material.opacity = 0.8;
                return;
            }
        }
        
        // 2. Drag-to-Orbit Creation
        if (dragOrbitMode) {
            const earthIntersect = uxRaycaster.intersectObject(earth);
            if (earthIntersect.length > 0) {
                controls.enabled = false;
                dragOrbitStartPos = earthIntersect[0].point.clone();
                
                // Create preview line
                const geo = new THREE.BufferGeometry().setFromPoints([dragOrbitStartPos, dragOrbitStartPos]);
                const mat = new THREE.LineBasicMaterial({ color: 0x8b5cf6, linewidth: 2 });
                dragOrbitPreviewLine = new THREE.Line(geo, mat);
                scene.add(dragOrbitPreviewLine);
            }
        }
    }
    
    function onPointerMove(event) {
        // 1. Maneuver Handle Dragging
        if (draggingHandle) {
            const dy = event.clientY - dragStartMouse.y;
            const dx = event.clientX - dragStartMouse.x;
            
            // Map screen movement to delta V
            // Pushing mouse up/right increases, down/left decreases
            const delta = (-dy + dx) * 0.5; // Sensitivity factor
            
            let newVal = dragStartValue + delta;
            const axis = draggingHandle.userData.axis;
            
            // Round to nearest whole number
            newVal = Math.round(newVal);
            
            const numInput = document.getElementById(`burn-${axis}`);
            const slideInput = document.getElementById(`burn-${axis}-slider`);
            if (numInput && slideInput) {
                numInput.value = newVal;
                slideInput.value = newVal;
                // Trigger preview update
                window.syncBurnSliders(axis);
            }
            return;
        }
        
        // 2. Drag-to-Orbit Creation Preview
        if (dragOrbitMode && dragOrbitStartPos) {
            const rect = renderer.domElement.getBoundingClientRect();
            uxMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            uxMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            uxRaycaster.setFromCamera(uxMouse, camera);
            
            // Intersect with a plane at Earth's center facing camera
            const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), new THREE.Vector3(0,0,0));
            const intersectPoint = new THREE.Vector3();
            uxRaycaster.ray.intersectPlane(plane, intersectPoint);
            
            if (intersectPoint && dragOrbitPreviewLine) {
                const posAttr = dragOrbitPreviewLine.geometry.attributes.position;
                posAttr.setXYZ(1, intersectPoint.x, intersectPoint.y, intersectPoint.z);
                posAttr.needsUpdate = true;
            }
        }
    }
    
    function onPointerUp(event) {
        if (draggingHandle) {
            controls.enabled = true;
            draggingHandle.material.opacity = 0.2;
            draggingHandle = null;
        }
        
        if (dragOrbitMode && dragOrbitStartPos && dragOrbitPreviewLine) {
            controls.enabled = true;
            
            // Calculate orbital parameters based on drag vector
            const endPos = new THREE.Vector3().fromArray(dragOrbitPreviewLine.geometry.attributes.position.array, 3);
            const dragVector = new THREE.Vector3().subVectors(endPos, dragOrbitStartPos);
            
            const altitudeKm = Math.max(200, (dragOrbitStartPos.length() - R_EARTH_KM) + dragVector.length() * 0.5); // Heuristic
            const inclination = Math.abs(Math.atan2(dragVector.y, Math.sqrt(dragVector.x*dragVector.x + dragVector.z*dragVector.z)) * 180 / Math.PI);
            
            // Reset state
            scene.remove(dragOrbitPreviewLine);
            dragOrbitPreviewLine.geometry.dispose();
            dragOrbitPreviewLine.material.dispose();
            dragOrbitPreviewLine = null;
            dragOrbitStartPos = null;
            toggleDragOrbitMode(); // Turn off
            
            // Open modal and pre-fill
            document.getElementById('m-alt').value = Math.round(altitudeKm);
            document.getElementById('m-inc').value = Math.round(inclination);
            document.getElementById('m-ecc').value = 0.0;
            openSatModal();
        }
    }
    
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
"""
content = content.replace("</script>\n</body>", interaction_code + "\n</script>\n</body>")

# 4. Animate Patch - Inserted cleanly at the end of the camera modes block
target_block = """        } else {
            // Earth focus default
            if (controls) controls.target.set(0, 0, 0);
            updateTelemetry(null, 0);
        }"""

animate_patch = """        } else {
            // Earth focus default
            if (controls) controls.target.set(0, 0, 0);
            updateTelemetry(null, 0);
        }

        // --- Radial Camera Lock & Stat Box Update ---
        const radialLockCheckbox = document.getElementById('camera-radial-lock');
        const statBox = document.getElementById('floating-stat-box');
        
        if (selectedSatIndex >= 0 && activeSatellites[selectedSatIndex]) {
            const sat = activeSatellites[selectedSatIndex];
            
            // 3D Stat Box Placement
            if (statBox) {
                statBox.style.display = 'block';
                const fName = document.getElementById('fs-name');
                if(fName) {
                    fName.innerText = sat.name || 'Sat';
                    document.getElementById('fs-alt').innerText = document.getElementById('t-alt').innerText;
                    document.getElementById('fs-vel').innerText = document.getElementById('t-vel').innerText;
                    document.getElementById('fs-inc').innerText = document.getElementById('t-inc').innerText;
                }
                
                // Project 3D pos to 2D screen
                const tempV = sat.mesh.position.clone();
                tempV.project(camera);
                const rect = renderer.domElement.getBoundingClientRect();
                const x = (tempV.x *  .5 + .5) * rect.width;
                const y = (tempV.y * -.5 + .5) * rect.height;
                statBox.style.left = `${x + 15}px`;
                statBox.style.top = `${y - 15}px`;
            }

            // Radial Lock Camera Adjustment
            if (radialLockCheckbox && radialLockCheckbox.checked && controls) {
                const planetPos = sat.parentBody === 'Moon' ? moon.position : new THREE.Vector3(0,0,0);
                const radialDir = new THREE.Vector3().subVectors(sat.mesh.position, planetPos).normalize();
                const dist = camera.position.distanceTo(sat.mesh.position);
                const targetCamPos = sat.mesh.position.clone().add(radialDir.multiplyScalar(dist));
                camera.position.lerp(targetCamPos, 0.05);
            }
        } else {
            if (statBox) statBox.style.display = 'none';
        }
        // --------------------------------------------"""

content = content.replace(target_block, animate_patch)

with open("index.html", "w") as f:
    f.write(content)
