        const M_TO_KM = 1 / 1000;
        const earthRadius = 6378.137;

        // --- State Management ---
        let activeSatellites = []; // { name, params, trajectory, mesh, line, points, times, color }
        let moonPoints = [];
        let selectedSatIndex = -1;
        let globalTimeIdx = 0;
        let floatTimeIdx = 0;

        // --- Three.js Setup ---
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2000000);
        let renderer;
        try {
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
        } catch (e) {
            const errorBanner = document.createElement('div');
            errorBanner.style.position = 'absolute';
            errorBanner.style.top = '50%';
            errorBanner.style.left = '50%';
            errorBanner.style.transform = 'translate(-50%, -50%)';
            errorBanner.style.color = 'white';
            errorBanner.style.background = 'red';
            errorBanner.style.padding = '20px';
            errorBanner.style.zIndex = '1000';
            errorBanner.innerText = 'WebGL Initialization Failed: ' + e.message;
            document.body.appendChild(errorBanner);
            console.error("WebGL failed to initialize:", e);
            // We cannot proceed with 3D rendering, but we shouldn't crash the rest of the JS.
        }

        let controls;
        if (renderer) {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
        }

        // Lighting & Textures
        const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
        sunLight.position.set(50000, 10000, 20000); 
        scene.add(sunLight);
        scene.add(new THREE.AmbientLight(0x222222)); 

        const textureLoader = new THREE.TextureLoader();
        const earthColor = textureLoader.load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg');
        const earthBump = textureLoader.load('https://unpkg.com/three-globe/example/img/earth-topology.png');
        const moonColor = textureLoader.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_1024.jpg'); 

        // Earth
        const earth = new THREE.Mesh(
            new THREE.SphereGeometry(earthRadius, 64, 64),
            new THREE.MeshPhongMaterial({ map: earthColor, bumpMap: earthBump, bumpScale: 50.0, specular: new THREE.Color('grey'), shininess: 5 })
        );
        scene.add(earth);

        // Moon
        const moon = new THREE.Mesh(
            new THREE.SphereGeometry(1737.4, 32, 32),
            new THREE.MeshPhongMaterial({ map: moonColor })
        );
        scene.add(moon);
        scene.add(new THREE.AxesHelper(earthRadius * 1.5));

        camera.position.set(earthRadius * 3, earthRadius * 2, earthRadius * 3);
        controls.target.copy(earth.position);

        // --- GUI Setup ---
        const gui = new lil.GUI({ title: 'View Settings', container: document.getElementById('gui-container') });
        const simSettings = { playbackSpeed: 5, cameraTarget: 'Earth' };
        gui.add(simSettings, 'playbackSpeed', 0, 20, 1).name('Playback Speed');
        // NOTE: lil-gui's .options() DESTROYS the controller and returns a
        // fresh one without handlers, so this must be re-assignable and the
        // handler re-attached on every refresh (see updateCameraDropdown).
        let cameraTargetCtrl = gui.add(simSettings, 'cameraTarget', ['Earth', 'Moon']).name('Camera Focus');
        // Snap the camera to a vantage point near the Moon; the animation
        // loop then keeps the target glued to it (follow-cam).
        function focusMoon() {
            selectedSatIndex = -2;
            if (moonPoints.length > 0) {
                const mp = moonPoints[globalTimeIdx % moonPoints.length];
                controls.target.copy(mp);
                camera.position.set(mp.x + 9000, mp.y + 5000, mp.z + 9000);
            }
            renderSidebar();
        }
        function handleFocusChange(v) {
            if(v === 'Earth') { selectedSatIndex = -1; controls.target.set(0,0,0); }
            else if(v === 'Moon') focusMoon();
            else selectedSatIndex = activeSatellites.findIndex(s => s.name === v);
            renderSidebar();
        }
        cameraTargetCtrl.onChange(handleFocusChange);

        // --- Fetch Moon ---
        let moonOrbitLine = null;
        fetch(`/moon_track?dt=60&n=1440`).then(r => r.json()).then(data => {
            if(data.error) return;
            moonPoints = data.map(p => new THREE.Vector3(p.x * M_TO_KM, p.y * M_TO_KM, p.z * M_TO_KM));
            // Closed orbit path (the track spans a full 24 h revolution).
            if (moonPoints.length > 1) {
                if (moonOrbitLine) scene.remove(moonOrbitLine);
                moonOrbitLine = new THREE.LineLoop(
                    new THREE.BufferGeometry().setFromPoints(moonPoints),
                    new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.5 })
                );
                scene.add(moonOrbitLine);
                moon.position.copy(moonPoints[0]);
            }
        });

        // --- Core Functions ---
        function addSatelliteToScene(satData) {
            const pts = satData.trajectory.map(p => new THREE.Vector3(p.x * M_TO_KM, p.y * M_TO_KM, p.z * M_TO_KM));
            const tms = satData.trajectory.map(p => p.t);
            const color = satData.color || Math.random() * 0xffffff;

            const mesh = new THREE.Mesh(new THREE.SphereGeometry(150, 16, 16), new THREE.MeshBasicMaterial({ color }));
            scene.add(mesh);

            const line = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(pts), 
                new THREE.LineBasicMaterial({ color, linewidth: 2, transparent: true, opacity: 0.8 })
            );
            scene.add(line);

            activeSatellites.push({
                name: satData.name, params: satData.params, trajectory: satData.trajectory,
                period_s: satData.period_s || 5400, // default 90m if missing
                mesh, line, points: pts, times: tms, color
            });

            updateCameraDropdown();
            renderSidebar();
        }

        function clearSatellites() {
            activeSatellites.forEach(sat => { scene.remove(sat.mesh); scene.remove(sat.line); });
            activeSatellites = [];
            selectedSatIndex = -1;
            updateCameraDropdown();
            renderSidebar();
        }

        function newProject() {
            clearSatellites();
            document.getElementById('save-name').value = '';
            globalTimeIdx = 0;
            floatTimeIdx = 0;
        }

        function exportData() {
            if (activeSatellites.length === 0) {
                alert("No active satellites to export.");
                return;
            }

            let csvContent = "Satellite,Time(s),X(m),Y(m),Z(m),VX(m/s),VY(m/s),VZ(m/s)\n";

            activeSatellites.forEach(sat => {
                const T = sat.period_s;
                const step = T / 6.0;
                
                const times = sat.times;
                if (times.length === 0) return;
                
                const t_end = times[times.length - 1];
                const t_start = Math.max(0, t_end - 10 * T);
                
                let current_t = t_start;
                
                while (current_t <= t_end) {
                    // Find closest index
                    let closestIdx = 0;
                    let minDiff = Infinity;
                    for (let i = 0; i < times.length; i++) {
                        let diff = Math.abs(times[i] - current_t);
                        if (diff < minDiff) {
                            minDiff = diff;
                            closestIdx = i;
                        }
                    }
                    
                    const pt = sat.trajectory[closestIdx];
                    csvContent += `${sat.name},${pt.t.toFixed(2)},${pt.x.toFixed(2)},${pt.y.toFixed(2)},${pt.z.toFixed(2)},${pt.vx.toFixed(4)},${pt.vy.toFixed(4)},${pt.vz.toFixed(4)}\n`;
                    
                    current_t += step;
                }
            });

            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", "trajectory_export.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function updateCameraDropdown() {
            const opts = ['Earth', 'Moon', ...activeSatellites.map(s => s.name)];
            // Keep a valid selection, then rebuild: .options() destroys the
            // old controller, so re-attach the focus handler to the new one.
            if (!opts.includes(simSettings.cameraTarget)) {
                simSettings.cameraTarget = 'Earth';
                selectedSatIndex = -1;
            }
            if (cameraTargetCtrl && typeof cameraTargetCtrl.options === 'function') {
                const fresh = cameraTargetCtrl.options(opts);
                if (fresh && fresh !== cameraTargetCtrl) {
                    cameraTargetCtrl = fresh;
                    cameraTargetCtrl.onChange(handleFocusChange);
                }
            }
        }

        function renderSidebar() {
            const list = document.getElementById('sat-list');
            list.innerHTML = '';
            activeSatellites.forEach((sat, idx) => {
                const li = document.createElement('li');
                li.innerText = sat.name;
                li.style.borderLeftColor = '#' + Math.floor(sat.color).toString(16).padStart(6, '0');
                if (idx === selectedSatIndex) li.classList.add('active');
                li.onclick = () => {
                    selectedSatIndex = idx;
                    simSettings.cameraTarget = sat.name;
                    updateCameraDropdown();
                    renderSidebar();
                };
                list.appendChild(li);
            });
        }

        // --- API Interactions ---
        async function submitNewSatellite() {
            const payload = {
                name: document.getElementById('m-name').value || 'Sat',
                mass: parseFloat(document.getElementById('m-mass').value),
                drag_area: parseFloat(document.getElementById('m-area').value),
                cd: parseFloat(document.getElementById('m-cd').value),
                altitude_km: parseFloat(document.getElementById('m-alt').value),
                eccentricity: parseFloat(document.getElementById('m-ecc').value),
                inclination_deg: parseFloat(document.getElementById('m-inc').value),
                raan_deg: parseFloat(document.getElementById('m-raan').value)
            };
            
            document.getElementById('modal-buttons').innerHTML = "<em>Computing Orbit...</em>";
            try {
                const res = await fetch('/simulate', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) {
                    alert("Simulation failed: " + (data.detail || res.status));
                    closeModal();
                    return;
                }
                addSatelliteToScene(data);
            } catch(e) { alert("Simulation failed: " + e.message); }
            closeModal();
        }

        async function saveSim() {
            const name = document.getElementById('save-name').value;
            if(!name) return alert("Enter a name");
            const payload = {
                filename: name,
                satellites: activeSatellites.map(s => ({name: s.name, params: s.params, period_s: s.period_s, trajectory: s.trajectory, color: s.color}))
            };
            await fetch('/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            alert("Saved!");
            populateSaves();
        }

        async function loadSim() {
            const name = document.getElementById('load-dropdown').value;
            if(!name) return;
            const res = await fetch(`/load/${name}`);
            const data = await res.json();
            clearSatellites();
            data.satellites.forEach(s => addSatelliteToScene(s));
            document.getElementById('save-name').value = name;
        }

        async function populateSaves() {
            const res = await fetch('/list_saves');
            const data = await res.json();
            const sel = document.getElementById('load-dropdown');
            sel.innerHTML = '<option value="">Select File...</option>';
            data.files.forEach(f => {
                const opt = document.createElement('option');
                opt.value = opt.innerText = f;
                sel.appendChild(opt);
            });
        }

        // --- Modals ---
        function openModal() { 
            document.getElementById('modal-overlay').style.display = 'flex'; 
            document.getElementById('modal-buttons').innerHTML = '<button onclick="closeModal()">Cancel</button><button onclick="submitNewSatellite()" style="background:#007bff;color:white;border:none;">Launch</button>';
        }
        function closeModal() { document.getElementById('modal-overlay').style.display = 'none'; }

        // --- Animation Loop ---
        function animate() {
            requestAnimationFrame(animate);
            earth.rotation.y += 0.0002; 

            // Time progression based on fastest array length approx
            const maxLen = activeSatellites.reduce((max, s) => Math.max(max, s.points.length), 0) || (moonPoints.length ? moonPoints.length : 1440);
            floatTimeIdx += (simSettings.playbackSpeed / 5); 
            globalTimeIdx = Math.floor(floatTimeIdx) % maxLen;

            // Update Moon
            if(moonPoints.length > 0) moon.position.copy(moonPoints[globalTimeIdx % moonPoints.length]);

            // Update Satellites
            activeSatellites.forEach(sat => {
                const idx = globalTimeIdx % sat.points.length;
                sat.mesh.position.copy(sat.points[idx]);
            });

            // Camera Target (Moon selection rides along: preserve the
            // viewer's offset while tracking the Moon's motion).
            if(selectedSatIndex === -2) {
                camera.position.add(moon.position.clone().sub(controls.target));
                controls.target.copy(moon.position);
            }
            else if(selectedSatIndex >= 0 && activeSatellites[selectedSatIndex]) {
                controls.target.copy(activeSatellites[selectedSatIndex].mesh.position);
            } else {
                controls.target.set(0,0,0);
            }

            // Update Telemetry if focused
            if (selectedSatIndex === -2) {
                document.getElementById('telemetry').innerHTML =
                    `Focus: Moon<br>Time: ${(globalTimeIdx*60).toFixed(1)} s<br>Distance: ${(moon.position.length()).toFixed(1)} km<br>Velocity: -- km/s`;
            } else if (selectedSatIndex >= 0 && activeSatellites[selectedSatIndex]) {
                const sat = activeSatellites[selectedSatIndex];
                const idx = globalTimeIdx % sat.points.length;
                const pos = sat.points[idx];
                const alt = (pos.length() - earthRadius).toFixed(2);
                
                const ptData = sat.trajectory[idx];
                const vMag = Math.sqrt(ptData.vx*ptData.vx + ptData.vy*ptData.vy + ptData.vz*ptData.vz) / 1000;
                
                document.getElementById('telemetry').innerHTML = 
                    `Focus: ${sat.name}<br>Time: ${ptData.t.toFixed(1)} s<br>Altitude: ${alt} km<br>Velocity: ${vMag.toFixed(3)} km/s`;
            } else {
                document.getElementById('telemetry').innerHTML = `Time: ${(globalTimeIdx*60).toFixed(1)} s<br>Altitude: -- km<br>Velocity: -- km/s`;
            }

            if (controls) controls.update();
            if (renderer) renderer.render(scene, camera);
        }
        
        populateSaves();
        if (renderer) animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            if (renderer) renderer.setSize(window.innerWidth, window.innerHeight);
        });
