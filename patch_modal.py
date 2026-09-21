import re

with open("visualization_3d/static/index.html", "r") as f:
    content = f.read()

# 1. UI changes
modal_html = """
            <div class="row" style="margin-top:10px; border-top:1px solid #334155; padding-top:10px;">
                <label style="color:#e2e8f0; font-weight:bold; font-size: 14px; cursor: pointer;"><input type="checkbox" id="m-cluster-enable" onchange="toggleClusterUI()"> 🌌 Generate Cluster / Constellation</label>
            </div>
            <div id="cluster-options" style="display:none; background:#0f172a; padding:12px; border-radius:6px; margin-top:8px; border:1px solid #3b82f6;">
                <div class="form-group"><label style="color:#93c5fd;">Satellites (N):</label><input type="number" id="m-cluster-n" value="5" min="2" max="100"></div>
                <div class="form-group"><label style="color:#93c5fd;">Δ Phase (°):</label><input type="number" id="m-cluster-dphase" value="10.0" step="1.0"></div>
                <div class="form-group"><label style="color:#93c5fd;">Δ RAAN (°):</label><input type="number" id="m-cluster-draan" value="0.0" step="1.0"></div>
                <div class="form-group"><label style="color:#93c5fd;">Δ Alt (km):</label><input type="number" id="m-cluster-dalt" value="0.0" step="1.0"></div>
                <div class="form-group"><label style="color:#93c5fd;">Δ Inc (°):</label><input type="number" id="m-cluster-dinc" value="0.0" step="1.0"></div>
                <div style="font-size:11px; color:#64748b; margin-top:6px;">Generates N satellites spaced out sequentially by these parameters to form Walker stars, strings-of-pearls, or custom local offsets.</div>
            </div>
"""

content = content.replace(
    '<div class="modal-buttons">',
    modal_html + '\n            <div class="modal-buttons">'
)

toggle_js = """
    window.toggleClusterUI = function() {
        const checked = document.getElementById('m-cluster-enable').checked;
        document.getElementById('cluster-options').style.display = checked ? 'block' : 'none';
    }
"""
content = content.replace('function closeModal() {', toggle_js + '\n    function closeModal() {')


# 2. Logic changes for submitNewSatellite
target = """    async function submitNewSatellite() {
        const btn = document.getElementById('launch-btn');
        btn.disabled = true;
        btn.innerText = 'Launching...';
        
        const parentBody = document.getElementById('m-body').value;
        const colorInput = document.getElementById('m-color').value;

        const payload = {
            name: document.getElementById('m-name').value,
            parent_body: parentBody,
            mass: parseFloat(document.getElementById('m-mass').value),
            drag_area: parseFloat(document.getElementById('m-area').value),
            cd: parseFloat(document.getElementById('m-cd').value),
            altitude_km: parseFloat(document.getElementById('m-alt').value),
            eccentricity: parseFloat(document.getElementById('m-ecc').value),
            inclination_deg: parseFloat(document.getElementById('m-inc').value),
            raan_deg: parseFloat(document.getElementById('m-raan').value) || 0.0,
            atmosphere_type: document.getElementById('m-atm-type').value,
            density_scale: parseFloat(document.getElementById('m-density-scale').value),
            include_j2: document.getElementById('m-j2').checked,
            include_drag: document.getElementById('m-drag').checked,
            include_moon: document.getElementById('m-moon').checked,
            color: colorInput,
            icon: document.getElementById('m-icon').value,
            t_start: simTime
        };"""

replace = """    async function submitNewSatellite() {
        const btn = document.getElementById('launch-btn');
        btn.disabled = true;
        btn.innerText = 'Launching...';
        
        const parentBody = document.getElementById('m-body').value;
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
                parent_body: parentBody,
                mass: parseFloat(document.getElementById('m-mass').value),
                drag_area: parseFloat(document.getElementById('m-area').value),
                cd: parseFloat(document.getElementById('m-cd').value),
                altitude_km: parseFloat(document.getElementById('m-alt').value) + (i * dAlt),
                eccentricity: parseFloat(document.getElementById('m-ecc').value),
                inclination_deg: (parseFloat(document.getElementById('m-inc').value) + (i * dInc)) % 180,
                raan_deg: (parseFloat(document.getElementById('m-raan').value || 0.0) + (i * dRaan)) % 360,
                arg_periapsis_deg: 0.0,
                true_anomaly_deg: (i * dPhase) % 360,
                atmosphere_type: document.getElementById('m-atm-type').value,
                density_scale: parseFloat(document.getElementById('m-density-scale').value),
                include_j2: document.getElementById('m-j2').checked,
                include_drag: document.getElementById('m-drag').checked,
                include_moon: document.getElementById('m-moon').checked,
                color: colorInput,
                icon: document.getElementById('m-icon').value,
                t_start: simTime
            });
        }
"""
content = content.replace(target, replace)

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
            
            closeModal();
        } catch (err) {
            alert("Launch Failed: " + err.message);
        } finally {
            btn.disabled = false;
            btn.innerText = 'Launch';
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
            }
            closeModal();
        } catch (err) {
            alert("Launch Failed: " + err.message);
        } finally {
            btn.disabled = false;
            btn.innerText = 'Launch';
        }"""
content = content.replace(target_fetch, replace_fetch)

with open("visualization_3d/static/index.html", "w") as f:
    f.write(content)
