/**
 * NanoDynamics Agentic Grader — Autonomous 3D Particle Cycle
 * Cycles smoothly between:
 *   1. Microchip (Silicon core, bus traces, IC pins)
 *   2. Neural Mesh (Synaptic network sphere with interconnected pathways)
 *   3. 3U Satellite (3U CubeSat bus, deployed solar panels, dipole antenna)
 */

(function () {
  const canvas = document.getElementById('nanodynamics-particle-canvas');
  if (!canvas || typeof THREE === 'undefined') return;

  const PARTICLE_COUNT = 2400;
  const MORPH_DURATION = 2.4; // seconds to transition
  const HOLD_DURATION = 4.5;  // seconds to hold shape

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
  camera.position.z = 175;

  const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // Particle positions buffers
  const currentPositions = new Float32Array(PARTICLE_COUNT * 3);
  const targetMicrochip = new Float32Array(PARTICLE_COUNT * 3);
  const targetNeuralMesh = new Float32Array(PARTICLE_COUNT * 3);
  const targetSatellite = new Float32Array(PARTICLE_COUNT * 3);
  const particleColors = new Float32Array(PARTICLE_COUNT * 3);

  // Color Palette: Cyan (0, 240, 255), Violet/Indigo (120, 80, 255), Cobalt (0, 112, 243)
  const colorCyan = new THREE.Color(0x00f0ff);
  const colorIndigo = new THREE.Color(0x7928ca);
  const colorCobalt = new THREE.Color(0x0070f3);

  // 1. Generate Microchip Coordinates
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const idx = i * 3;
    const r = Math.random();
    if (r < 0.45) {
      // Central square CPU die [-32, 32] x [-32, 32]
      targetMicrochip[idx] = (Math.random() - 0.5) * 64;
      targetMicrochip[idx + 1] = (Math.random() - 0.5) * 64;
      targetMicrochip[idx + 2] = (Math.random() - 0.5) * 4;
    } else if (r < 0.80) {
      // Bus traces extending out along axes
      const axis = Math.floor(Math.random() * 4);
      const span = 35 + Math.random() * 45;
      const spread = (Math.random() - 0.5) * 16;
      if (axis === 0) { targetMicrochip[idx] = span; targetMicrochip[idx + 1] = spread; }
      else if (axis === 1) { targetMicrochip[idx] = -span; targetMicrochip[idx + 1] = spread; }
      else if (axis === 2) { targetMicrochip[idx] = spread; targetMicrochip[idx + 1] = span; }
      else { targetMicrochip[idx] = spread; targetMicrochip[idx + 1] = -span; }
      targetMicrochip[idx + 2] = (Math.random() - 0.5) * 2;
    } else {
      // Outer peripheral contact pads / pins
      const angle = Math.random() * Math.PI * 2;
      const radius = 70 + Math.random() * 14;
      targetMicrochip[idx] = Math.cos(angle) * radius;
      targetMicrochip[idx + 1] = Math.sin(angle) * radius;
      targetMicrochip[idx + 2] = (Math.random() - 0.5) * 3;
    }

    // Set initial position & colors
    currentPositions[idx] = targetMicrochip[idx];
    currentPositions[idx + 1] = targetMicrochip[idx + 1];
    currentPositions[idx + 2] = targetMicrochip[idx + 2];

    const mix = Math.random();
    const c = mix < 0.6 ? colorCyan.clone().lerp(colorCobalt, mix) : colorCyan.clone().lerp(colorIndigo, mix);
    particleColors[idx] = c.r;
    particleColors[idx + 1] = c.g;
    particleColors[idx + 2] = c.b;
  }

  // 2. Generate Neural Mesh Coordinates (Fibonacci Sphere with synaptic clusters)
  const phi = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const idx = i * 3;
    const y = 1 - (i / (PARTICLE_COUNT - 1)) * 2;
    const radius = Math.sqrt(1 - y * y);
    const theta = phi * i;

    const baseRadius = 65 + (Math.sin(theta * 4) * 8);
    targetNeuralMesh[idx] = Math.cos(theta) * radius * baseRadius;
    targetNeuralMesh[idx + 1] = y * baseRadius;
    targetNeuralMesh[idx + 2] = Math.sin(theta) * radius * baseRadius;
  }

  // 3. Generate 3U Satellite Coordinates (3U Chassis + Deployed Wings + Antenna)
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const idx = i * 3;
    const r = Math.random();
    if (r < 0.40) {
      // 3U CubeSat Body: 22 x 66 x 22 mm proportion (scaled: 16 x 50 x 16)
      targetSatellite[idx] = (Math.random() - 0.5) * 20;
      targetSatellite[idx + 1] = (Math.random() - 0.5) * 60;
      targetSatellite[idx + 2] = (Math.random() - 0.5) * 20;
    } else if (r < 0.88) {
      // Deployed Solar Panel Wings on Left (-X) and Right (+X)
      const side = Math.random() < 0.5 ? -1 : 1;
      const wingX = side * (18 + Math.random() * 52);
      const wingY = (Math.random() - 0.5) * 44;
      const wingZ = (Math.random() - 0.5) * 4;
      targetSatellite[idx] = wingX;
      targetSatellite[idx + 1] = wingY;
      targetSatellite[idx + 2] = wingZ;
    } else {
      // Nadir Antenna / Sensor Boom extending from top (+Y)
      targetSatellite[idx] = (Math.random() - 0.5) * 3;
      targetSatellite[idx + 1] = 30 + Math.random() * 40;
      targetSatellite[idx + 2] = (Math.random() - 0.5) * 3;
    }
  }

  // Build Three.js Geometry & Material
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(currentPositions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));

  // Particle texture sprite
  const spriteCanvas = document.createElement('canvas');
  spriteCanvas.width = 32;
  spriteCanvas.height = 32;
  const sCtx = spriteCanvas.getContext('2d');
  const grad = sCtx.createRadialGradient(16, 16, 0, 16, 16, 16);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.3, 'rgba(0,240,255,0.8)');
  grad.addColorStop(0.7, 'rgba(0,112,243,0.3)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  sCtx.fillStyle = grad;
  sCtx.fillRect(0, 0, 32, 32);
  const texture = new THREE.CanvasTexture(spriteCanvas);

  const material = new THREE.PointsMaterial({
    size: 2.8,
    vertexColors: true,
    map: texture,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const particleSystem = new THREE.Points(geometry, material);
  scene.add(particleSystem);

  // State Machine for Autonomous Particle Cycle
  // Cycle order: 0 (Microchip) -> 1 (Neural Mesh) -> 2 (3U Satellite) -> 0
  const states = [targetMicrochip, targetNeuralMesh, targetSatellite];
  const stateLabels = ['Silicon Microchip', 'Living Neural Mesh', '3U Satellite Orbital Swarm'];
  let currentShapeIdx = 0;
  let targetShapeIdx = 1;
  let phase = 'hold'; // 'hold' or 'morph'
  let phaseTimer = 0;

  // Mouse Parallax
  let mouseX = 0, mouseY = 0;
  let targetRotationX = 0, targetRotationY = 0;

  window.addEventListener('mousemove', function (e) {
    mouseX = (e.clientX - window.innerWidth / 2) * 0.0005;
    mouseY = (e.clientY - window.innerHeight / 2) * 0.0005;
  });

  window.addEventListener('resize', function () {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // Easing function (smooth cubic)
  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  const clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    phaseTimer += delta;

    if (phase === 'hold') {
      if (phaseTimer >= HOLD_DURATION) {
        phase = 'morph';
        phaseTimer = 0;
        targetShapeIdx = (currentShapeIdx + 1) % states.length;
        const badge = document.getElementById('particle-state-badge');
        if (badge) {
          badge.textContent = "Morphing: " + stateLabels[targetShapeIdx];
        }
      }
    } else if (phase === 'morph') {
      const progress = Math.min(1.0, phaseTimer / MORPH_DURATION);
      const ease = easeInOutCubic(progress);

      const sourceArray = states[currentShapeIdx];
      const destArray = states[targetShapeIdx];
      const pos = geometry.attributes.position.array;

      for (let i = 0; i < PARTICLE_COUNT * 3; i++) {
        pos[i] = sourceArray[i] + (destArray[i] - sourceArray[i]) * ease;
      }
      geometry.attributes.position.needsUpdate = true;

      if (progress >= 1.0) {
        currentShapeIdx = targetShapeIdx;
        phase = 'hold';
        phaseTimer = 0;
        const badge = document.getElementById('particle-state-badge');
        if (badge) {
          badge.textContent = "Active State: " + stateLabels[currentShapeIdx];
        }
      }
    }

    // Gentle 3D continuous rotation with mouse parallax
    targetRotationY += 0.003;
    targetRotationX = mouseY * 0.5;

    particleSystem.rotation.y += (targetRotationY - particleSystem.rotation.y) * 0.05;
    particleSystem.rotation.x += (targetRotationX - particleSystem.rotation.x) * 0.05;

    renderer.render(scene, camera);
  }

  animate();
})();
