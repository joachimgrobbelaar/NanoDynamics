import re

with open("visualization_3d/static/index.html", "r") as f:
    content = f.read()

target = """    async function confirmAddSatellite() {
        const btn = document.getElementById('m-add-btn');
        const errBox = document.getElementById('m-error');
        btn.disabled = true;
        btn.innerText = 'Calculating...';
        errBox.innerText = '';
        
        const isMoon = document.getElementById('m-body-moon').checked;
        const colorInput = document.getElementById('m-color').value;

        const payload = {
            name: document.getElementById('m-name').value,
            parent_body: isMoon ? 'Moon' : 'Earth',
            mass: parseFloat(document.getElementById('m-mass').value),
            drag_area: parseFloat(document.getElementById('m-area').value),
            cd: parseFloat(document.getElementById('m-cd').value),
            altitude_km: parseFloat(document.getElementById('m-alt').value),
            eccentricity: parseFloat(document.getElementById('m-ecc').value),
            inclination_deg: parseFloat(document.getElementById('m-inc').value),
            raan_deg: parseFloat(document.getElementById('m-raan').value) || 0.0,
            atmosphere_type: document.getElementById('m-atm').value,
            density_scale: parseFloat(document.getElementById('m-dens').value),
            include_j2: document.getElementById('m-j2').checked,
            include_drag: document.getElementById('m-drag').checked,
            include_moon: document.getElementById('m-moon-grav').checked,
            color: colorInput,
            icon: document.getElementById('m-icon').value,
            t_start: simTime
        };"""

replace = """    async function confirmAddSatellite() {
        const btn = document.getElementById('m-add-btn');
        const errBox = document.getElementById('m-error');
        btn.disabled = true;
        btn.innerText = 'Calculating...';
        errBox.innerText = '';
        
        const isMoon = document.getElementById('m-body-moon').checked;
        const colorInput = document.getElementById('m-color').value;
        const baseName = document.getElementById('m-name').value;
        const isCluster = document.getElementById('m-cluster-enable') && document.getElementById('m-cluster-enable').checked;
        
        const N = isCluster ? parseInt(document.getElementById('m-cluster-n').value) || 1 : 1;
        const dPhase = isCluster ? parseFloat(document.getElementById('m-cluster-dphase').value) || 0.0 : 0.0;
        const dRaan = isCluster ? parseFloat(document.getElementById('m-cluster-draan').value) || 0.0 : 0.0;
        const dAlt = isCluster ? parseFloat(document.getElementById('m-cluster-dalt').value) || 0.0 : 0.0;
        const dInc = isCluster ? parseFloat(document.getElementById('m-cluster-dinc').value) || 0.0 : 0.0;

        let payloads = [];
        for(let i = 0; i < N; i++) {
            payloads.push({
                name: N > 1 ? `${baseName}-${i+1}` : baseName,
                parent_body: isMoon ? 'Moon' : 'Earth',
                mass: parseFloat(document.getElementById('m-mass').value),
                drag_area: parseFloat(document.getElementById('m-area').value),
                cd: parseFloat(document.getElementById('m-cd').value),
                altitude_km: parseFloat(document.getElementById('m-alt').value) + (i * dAlt),
                eccentricity: parseFloat(document.getElementById('m-ecc').value),
                inclination_deg: (parseFloat(document.getElementById('m-inc').value) + (i * dInc)) % 180,
                raan_deg: (parseFloat(document.getElementById('m-raan').value || 0.0) + (i * dRaan)) % 360,
                arg_periapsis_deg: 0.0,
                true_anomaly_deg: (i * dPhase) % 360,
                atmosphere_type: document.getElementById('m-atm').value,
                density_scale: parseFloat(document.getElementById('m-dens').value),
                include_j2: document.getElementById('m-j2').checked,
                include_drag: document.getElementById('m-drag').checked,
                include_moon: document.getElementById('m-moon-grav').checked,
                color: colorInput,
                icon: document.getElementById('m-icon').value,
                t_start: simTime
            });
        }
        
        let allSuccess = true;
        let clusterCount = 0;
"""
content = content.replace(target, replace)

# 2. Modify the fetch loop
target_fetch = """        try {
            const res = await fetch('/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Simulation failed');
            
            payload.initial_state = data.new_state;
            addSatelliteToScene(payload, data.trajectory);
            
            document.getElementById('sat-modal').style.display = 'none';
        } catch (err) {
            errBox.innerText = err.message;
        } finally {
            btn.disabled = false;
            btn.innerText = 'Deploy Satellite';
        }"""
replace_fetch = """        try {
            for(let payload of payloads) {
                const res = await fetch('/simulate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Simulation failed');
                
                payload.initial_state = data.new_state;
                addSatelliteToScene(payload, data.trajectory);
                clusterCount++;
            }
            document.getElementById('sat-modal').style.display = 'none';
        } catch (err) {
            errBox.innerText = err.message + (clusterCount > 0 ? ` (after adding ${clusterCount} sats)` : '');
            allSuccess = false;
        } finally {
            btn.disabled = false;
            btn.innerText = 'Deploy Satellite(s)';
        }"""
content = content.replace(target_fetch, replace_fetch)

with open("visualization_3d/static/index.html", "w") as f:
    f.write(content)
