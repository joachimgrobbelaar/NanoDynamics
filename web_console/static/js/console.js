// Nanodynamics-v3 — Photorealistic Three.js Rayleigh/Mie Atmosphere, Cislunar CR3BP & PINN Ribbon Engine

let scene, camera, renderer, earthMesh, atmoMesh, moonMesh, truthLine, pinnLine, cr3bpLine;
let camAngle = 0.0;
let camMode = 'leo';

function initThreeScene() {
  const container = document.getElementById('webgl-viewport');
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 2000);
  camera.position.set(0, 14, 28);

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  container.appendChild(renderer.domElement);

  const sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
  sunLight.position.set(45, 22, 35);
  scene.add(sunLight);
  scene.add(new THREE.AmbientLight(0x1e293b, 0.65));

  // Earth Photosphere (Normalized R_E = 6.378 units)
  const earthGeo = new THREE.SphereGeometry(6.378, 64, 64);
  const earthMat = new THREE.MeshStandardMaterial({
    color: 0x0b3b68,
    roughness: 0.45,
    metalness: 0.25,
    emissive: 0x041226,
  });
  earthMesh = new THREE.Mesh(earthGeo, earthMat);
  scene.add(earthMesh);

  // Custom GLSL Rayleigh/Mie Atmospheric Limb Halo
  const atmoGeo = new THREE.SphereGeometry(6.68, 64, 64);
  const atmoMat = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vNormal;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      void main() {
        float intensity = pow(0.66 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.6);
        gl_FragColor = vec4(0.0, 0.85, 1.0, 1.0) * intensity * 1.35;
      }
    `,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true,
  });
  atmoMesh = new THREE.Mesh(atmoGeo, atmoMat);
  scene.add(atmoMesh);

  // Moon & L1/L2 Cislunar Gateways
  const moonGeo = new THREE.SphereGeometry(1.737, 32, 32);
  const moonMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.85 });
  moonMesh = new THREE.Mesh(moonGeo, moonMat);
  moonMesh.position.set(38.44, 0, 0);
  scene.add(moonMesh);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  camAngle += 0.0025;
  if (earthMesh) earthMesh.rotation.y += 0.0018;

  if (camMode === 'leo') {
    camera.position.set(Math.cos(camAngle) * 20, 9.5, Math.sin(camAngle) * 20);
    camera.lookAt(0, 0, 0);
  } else {
    camera.position.set(19 + Math.cos(camAngle) * 32, 16, Math.sin(camAngle) * 32);
    camera.lookAt(19, 0, 0);
  }
  renderer.render(scene, camera);
}

function setCameraMode(mode) {
  camMode = mode;
}

function updateUiLabels() {
  document.getElementById('val-alt').textContent = document.getElementById('inp-alt').value + ' km';
  document.getElementById('val-f107').textContent = Number(document.getElementById('inp-f107').value).toFixed(1);
  document.getElementById('val-kp').textContent = Number(document.getElementById('inp-kp').value).toFixed(1);
  document.getElementById('val-b').textContent = Number(document.getElementById('inp-b').value).toFixed(4);
}

async function loadEmpiricalStatus() {
  const res = await fetch('/api/empirical');
  const data = await res.json();
  document.getElementById('emp-eof-count').textContent = `${data.esa_sentinel_1a_eof.length} OSVs (ITRF->GCRF)`;
  document.getElementById('emp-sp3-swarm').textContent = `${data.nasa_sp3_swarm_a.length} Epochs (Swarm-A)`;
  document.getElementById('emp-sp3-grace').textContent = `${data.nasa_sp3_grace_fo.length} Epochs (GRACE-FO)`;
  document.getElementById('hw-usb-badge').textContent = data.esp32_gateway_usb ? '● /dev/ttyUSB0 ONLINE' : '● HIL STANDBY';
}

async function runPropagationCompare() {
  const payload = {
    altitude_km: parseFloat(document.getElementById('inp-alt').value),
    f107: parseFloat(document.getElementById('inp-f107').value),
    kp: parseFloat(document.getElementById('inp-kp').value),
    b_coeff: parseFloat(document.getElementById('inp-b').value),
    dt_s: 25.0,
    steps: 72,
  };
  const res = await fetch('/api/propagate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  // Render 3D Truth vs PINN Orbit Ribbons (1e-6 scale: 1000 km = 1.0 unit)
  if (truthLine) scene.remove(truthLine);
  if (pinnLine) scene.remove(pinnLine);
  if (cr3bpLine) scene.remove(cr3bpLine);

  const truthPts = data.truth_r_m.map(p => new THREE.Vector3(p[0] * 1e-6, p[2] * 1e-6, -p[1] * 1e-6));
  const pinnPts = data.pinn_r_m.map(p => new THREE.Vector3(p[0] * 1e-6, p[2] * 1e-6, -p[1] * 1e-6));
  const cr3bpPts = data.cr3bp_states_nd.map(s => new THREE.Vector3(s[0] * 38.44, s[2] * 38.44, -s[1] * 38.44));

  truthLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(truthPts),
    new THREE.LineBasicMaterial({ color: 0x00E5FF, linewidth: 2 })
  );
  pinnLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pinnPts),
    new THREE.LineBasicMaterial({ color: 0x00FF66, linewidth: 2 })
  );
  cr3bpLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(cr3bpPts),
    new THREE.LineBasicMaterial({ color: 0xFFD21E, linewidth: 2 })
  );
  scene.add(truthLine);
  scene.add(pinnLine);
  scene.add(cr3bpLine);

  const posErr = data.pos_residual_m;
  const hamErr = data.energy_rel_drift;
  const latMs = data.infer_times_ms;

  document.getElementById('lbl-pos-err').textContent = posErr[posErr.length - 1].toFixed(3) + ' m';
  document.getElementById('lbl-ham-err').textContent = hamErr[hamErr.length - 1].toExponential(2);
  document.getElementById('lbl-lat-ms').textContent = latMs[latMs.length - 1].toFixed(3) + ' ms';
  document.getElementById('val-jacobi').textContent = Number(data.cr3bp_jacobi_drift).toExponential(2);

  drawSpark('spark-pos', posErr, '#00E5FF');
  drawSpark('spark-ham', hamErr, '#00FF66');
  drawSpark('spark-lat', latMs, '#FFD21E');
}

function drawSpark(id, arr, color) {
  const c = document.getElementById(id);
  if (!c || !arr || arr.length < 2) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const minV = Math.min(...arr), maxV = Math.max(...arr);
  const span = Math.max(maxV - minV, 1e-12);
  ctx.beginPath();
  arr.forEach((v, i) => {
    const x = (i / (arr.length - 1)) * (W - 16) + 8;
    const y = H - 10 - ((v - minV) / span) * (H - 20);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

window.addEventListener('DOMContentLoaded', async () => {
  initThreeScene();
  await loadEmpiricalStatus();
  await runPropagationCompare();
});
