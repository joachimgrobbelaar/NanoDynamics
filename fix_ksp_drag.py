import re

with open("visualization_3d/static/index.html", "r") as f:
    content = f.read()

# Replace variables in UI State
state_target = """    let draggingHandle = null;
    let dragStartMouse = new THREE.Vector2();
    let dragStartValue = 0;"""
state_replace = """    let draggingHandle = null;
    let dragStartValue = 0;
    let dragPlane = null;
    let dragStartPoint3D = null;"""
content = content.replace(state_target, state_replace)


# Replace pointerDown maneuver handle logic
down_target = """                controls.enabled = false; // Disable camera orbit
                draggingHandle = intersects[0].object;
                dragStartMouse.set(event.clientX, event.clientY);
                
                const axis = draggingHandle.userData.axis;
                dragStartValue = parseFloat(document.getElementById(`burn-${axis}`).value) || 0;
                draggingHandle.material.opacity = 0.8;
                return;"""
down_replace = """                controls.enabled = false; // Disable camera orbit
                draggingHandle = intersects[0].object;
                const axis = draggingHandle.userData.axis;
                dragStartValue = parseFloat(document.getElementById(`burn-${axis}`).value) || 0;
                draggingHandle.material.opacity = 0.8;
                
                // Set up drag plane facing camera at handle pos
                dragPlane = new THREE.Plane().setFromNormalAndCoplanarPoint(
                    camera.getWorldDirection(new THREE.Vector3()).negate(),
                    intersects[0].point
                );
                const pt = new THREE.Vector3();
                uxRaycaster.ray.intersectPlane(dragPlane, pt);
                dragStartPoint3D = pt.clone();
                return;"""
content = content.replace(down_target, down_replace)


# Replace pointerMove maneuver handle logic
move_target = """        // 1. Maneuver Handle Dragging
        if (draggingHandle) {
            const dy = event.clientY - dragStartMouse.y;
            const dx = event.clientX - dragStartMouse.x;
            
            // Map screen movement to delta V
            // Increased sensitivity
            const delta = (-dy + dx) * 2.0; // Increased sensitivity
            
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
        }"""
move_replace = """        // 1. Maneuver Handle Dragging (KSP 3D Projection)
        if (draggingHandle) {
            const rect = renderer.domElement.getBoundingClientRect();
            uxMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            uxMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            uxRaycaster.setFromCamera(uxMouse, camera);
            
            const pt = new THREE.Vector3();
            uxRaycaster.ray.intersectPlane(dragPlane, pt);
            if (pt && dragStartPoint3D) {
                const diff = new THREE.Vector3().subVectors(pt, dragStartPoint3D);
                
                const axisName = draggingHandle.userData.axis;
                const localDir = new THREE.Vector3();
                if (axisName === 'prograde') localDir.set(1,0,0);
                else if (axisName === 'radial') localDir.set(0,1,0);
                else if (axisName === 'normal') localDir.set(0,0,1);
                
                const worldDir = localDir.clone().transformDirection(nodeGizmoGroup.matrixWorld).normalize();
                
                // Dot product gives the drag distance along the 3D axis vector
                const deltaMeters = diff.dot(worldDir);
                
                // Scale scene distance to Delta V (m/s)
                const deltaV = deltaMeters * 5.0;
                
                let newVal = dragStartValue + deltaV;
                newVal = Math.round(newVal);
                
                const numInput = document.getElementById(`burn-${axisName}`);
                const slideInput = document.getElementById(`burn-${axisName}-slider`);
                if (numInput && slideInput) {
                    numInput.value = newVal;
                    slideInput.value = newVal;
                    window.syncBurnSliders(axisName);
                }
            }
            return;
        }"""
content = content.replace(move_target, move_replace)

with open("visualization_3d/static/index.html", "w") as f:
    f.write(content)
