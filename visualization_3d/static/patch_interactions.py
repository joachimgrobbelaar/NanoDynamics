import re

with open("index.html", "r") as f:
    content = f.read()

interaction_code = """
    // ── INTERACTIVE UX: DRAG-TO-ORBIT & MANEUVER NODES ───────────────────────
    
    // UI State
    let dragOrbitMode = false;
    let dragOrbitStartPos = null;
    let dragOrbitPreviewLine = null;
    
    // Maneuver Gizmo Drag State
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
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
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        
        // 1. Maneuver Handle Dragging
        if (nodeGizmoGroup.visible) {
            const intersects = raycaster.intersectObjects(maneuverHandles);
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
            const earthIntersect = raycaster.intersectObject(earth);
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
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            
            // Intersect with a plane at Earth's center facing camera
            const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), new THREE.Vector3(0,0,0));
            const intersectPoint = new THREE.Vector3();
            raycaster.ray.intersectPlane(plane, intersectPoint);
            
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

with open("index.html", "w") as f:
    f.write(content)
