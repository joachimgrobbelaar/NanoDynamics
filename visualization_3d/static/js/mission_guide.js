/**
 * NANODYNAMICS // MISSION CONTROL INTERACTIVE SPOTLIGHT GUIDE (< 200 lines)
 * Zero-dependency SVG mask + dynamic clip-path spotlight engine covering 7 GNC tour steps.
 */
(function () {
    'use strict';

    const TOUR_STEPS = [
        {
            badge: '01 / 07 // COMMAND DECK & TIME WARP',
            selector: '#top-nav',
            summary: 'Mission Elapsed Time (MET) clock, real-time vs time-acceleration warp selector (1× to 10,000×), and simulation pause/resume controls (`Space` key).'
        },
        {
            badge: '02 / 07 // CAMERA FOCUS & RADIAL LOCKING',
            selector: '#camera-focus-bar, #sidebar',
            summary: 'Switch camera tracking between Earth, Moon, Cislunar System, and active Satellite Tracker, with radial nadir view locking for orbital plane inspection.'
        },
        {
            badge: '03 / 07 // TELEMETRY & AEROTHERMAL MONITORING',
            selector: '#telemetry-panel',
            summary: 'Live ECI/ECEF state vectors (r, v), osculating Keplerian elements (a, e, i, Ω, ω, ν), Hamiltonian energy balance (E_k vs E_target), and Sutton-Graves aero-thermal heating (T_aero).'
        },
        {
            badge: '04 / 07 // VNC THRUSTER BURN NODES',
            selector: '#cmd-burn-btn, #maneuver-panel',
            summary: 'Interactive 3D maneuver node placement along the orbit with Prograde (V), Normal (N), and Radial (C) Δv handles and real-time trajectory preview branching.'
        },
        {
            badge: '05 / 07 // ONBOARD OPTICAL PIP CAMERA',
            selector: '#cmd-cam-btn, #sat-camera-pip',
            summary: 'Resizable Picture-in-Picture optical viewport supporting Nadir Earth Observation, Velocity Horizon, and Zenith Star Tracker modes with PNG snapshot export.'
        },
        {
            badge: '06 / 07 // ISL LASER COMMUNICATION MESH',
            selector: '#cmd-isl-btn, #isl-routing-panel',
            summary: 'Inter-Satellite Laser (ISL) optical mesh routing with Earth-limb line-of-sight occlusion checking and Dijkstra shortest-path telemetry relay.'
        },
        {
            badge: '07 / 07 // EDGE TINYML HARDWARE',
            selector: '#cmd-config-btn, #hil-benchmark-panel',
            summary: 'Dual-ESP32 FreeRTOS EdgeLink management (Node 1 Wi-Fi/OTA Gateway + Node 2 TFLM INT8 PINN & Symplectic Hamiltonian Manifold projection over ESP-NOW).'
        }
    ];

    let currentStep = 0;
    let isActive = false;
    let overlayEl = null;
    let ringEl = null;
    let cardEl = null;

    function resolveTarget(selectorList) {
        const parts = selectorList.split(',').map(s => s.trim());
        for (let i = 0; i < parts.length; i++) {
            const el = document.querySelector(parts[i]);
            if (el) {
                const r = el.getBoundingClientRect();
                if (r.width > 4 && r.height > 4) return el;
            }
        }
        return document.querySelector(parts[0]) || document.body;
    }

    function buildEvenOddClipPath(vw, vh, rect, pad) {
        const x1 = Math.max(0, Math.round(rect.left - pad));
        const y1 = Math.max(0, Math.round(rect.top - pad));
        const x2 = Math.min(vw, Math.round(rect.right + pad));
        const y2 = Math.min(vh, Math.round(rect.bottom + pad));
        return `path(evenodd, "M 0 0 H ${vw} V ${vh} H 0 Z M ${x1} ${y1} H ${x2} V ${y2} H ${x1} Z")`;
    }

    function ensureOverlayDOM() {
        if (overlayEl) return;
        overlayEl = document.createElement('div');
        overlayEl.id = 'mission-guide-overlay';
        overlayEl.style.cssText = 'position:fixed;inset:0;z-index:9998;pointer-events:none;display:none;';

        const maskSvg = document.createElement('div');
        maskSvg.id = 'mission-guide-mask';
        maskSvg.style.cssText = 'position:fixed;inset:0;background:rgba(5, 7, 11, 0.85);pointer-events:none;transition:clip-path 180ms cubic-bezier(0.16,1,0.3,1);';
        overlayEl.appendChild(maskSvg);

        ringEl = document.createElement('div');
        ringEl.id = 'mission-guide-spotlight-ring';
        ringEl.style.cssText = 'position:fixed;border:1.5px solid #00E5FF;box-shadow:0 0 18px rgba(0,229,255,0.45),inset 0 0 12px rgba(0,229,255,0.18);border-radius:4px;pointer-events:none;transition:all 180ms cubic-bezier(0.16,1,0.3,1);';
        overlayEl.appendChild(ringEl);

        cardEl = document.createElement('div');
        cardEl.id = 'mission-guide-card';
        cardEl.className = 'sx-glass-panel';
        cardEl.style.cssText = 'position:fixed;width:360px;padding:14px 16px;border-radius:6px;background:rgba(5,7,11,0.95);border:1px solid rgba(0,229,255,0.45);box-shadow:0 12px 32px rgba(0,0,0,0.85);color:#E2E8F0;pointer-events:auto;z-index:9999;font-family:ui-monospace,"JetBrains Mono","Roboto Mono",monospace;';
        cardEl.innerHTML = `
            <div id="mg-step-badge" class="sx-telemetry-header" style="color:#00E5FF;font-size:10px;font-weight:700;letter-spacing:0.14em;margin-bottom:8px;"></div>
            <div id="mg-step-summary" style="font-size:12px;line-height:1.55;color:#E2E8F0;margin-bottom:14px;font-family:system-ui,-apple-system,sans-serif;"></div>
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;border-top:1px solid rgba(56,189,248,0.18);padding-top:10px;">
                <button id="mg-prev-btn" class="sx-cmd-btn" type="button">[← PREV]</button>
                <div style="display:flex;gap:6px;">
                    <button id="mg-next-btn" class="sx-cmd-btn" type="button" style="border-color:#00E5FF !important;color:#00E5FF !important;">[NEXT →]</button>
                    <button id="mg-exit-btn" class="sx-cmd-btn" type="button" style="border-color:#FF1744 !important;color:#FF1744 !important;">[✕ EXIT]</button>
                </div>
            </div>
        `;
        overlayEl.appendChild(cardEl);
        document.body.appendChild(overlayEl);

        document.getElementById('mg-prev-btn').addEventListener('click', () => window.MissionGuide.prev());
        document.getElementById('mg-next-btn').addEventListener('click', () => window.MissionGuide.next());
        document.getElementById('mg-exit-btn').addEventListener('click', () => window.MissionGuide.exit());
        window.addEventListener('resize', () => { if (isActive) renderStep(); });
    }

    function renderStep() {
        ensureOverlayDOM();
        const step = TOUR_STEPS[currentStep];
        const target = resolveTarget(step.selector);
        const rect = target.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const pad = 6;

        const maskEl = document.getElementById('mission-guide-mask');
        maskEl.style.clipPath = buildEvenOddClipPath(vw, vh, rect, pad);

        ringEl.style.left = Math.max(0, rect.left - pad) + 'px';
        ringEl.style.top = Math.max(0, rect.top - pad) + 'px';
        ringEl.style.width = Math.min(vw, rect.width + pad * 2) + 'px';
        ringEl.style.height = Math.min(vh, rect.height + pad * 2) + 'px';

        document.getElementById('mg-step-badge').textContent = step.badge;
        document.getElementById('mg-step-summary').textContent = step.summary;

        let cardLeft = Math.min(vw - 380, Math.max(16, rect.left));
        let cardTop = rect.bottom + 14;
        if (cardTop + 185 > vh) {
            cardTop = Math.max(48, rect.top - 185);
            if (cardTop < 48) cardLeft = Math.min(vw - 380, Math.max(16, rect.right + 16));
        }
        cardEl.style.left = cardLeft + 'px';
        cardEl.style.top = cardTop + 'px';
    }

    window.MissionGuide = {
        start: function (stepIndex = 0) {
            ensureOverlayDOM();
            currentStep = Math.max(0, Math.min(TOUR_STEPS.length - 1, stepIndex));
            isActive = true;
            overlayEl.style.display = 'block';
            renderStep();
        },
        next: function () {
            if (!isActive) return;
            currentStep = (currentStep + 1) % TOUR_STEPS.length;
            renderStep();
        },
        prev: function () {
            if (!isActive) return;
            currentStep = (currentStep - 1 + TOUR_STEPS.length) % TOUR_STEPS.length;
            renderStep();
        },
        exit: function () {
            isActive = false;
            if (overlayEl) overlayEl.style.display = 'none';
        },
        getStep: function () { return currentStep; },
        isActive: function () { return isActive; },
        steps: TOUR_STEPS
    };

    window.addEventListener('keydown', function (e) {
        const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
        if (e.key === '?' || e.key === 'h' || e.key === 'H') {
            e.preventDefault();
            if (isActive) window.MissionGuide.exit();
            else window.MissionGuide.start(0);
        } else if (e.key === 'Escape' && isActive) {
            e.preventDefault();
            window.MissionGuide.exit();
        } else if (e.key === 'ArrowRight' && isActive) {
            window.MissionGuide.next();
        } else if (e.key === 'ArrowLeft' && isActive) {
            window.MissionGuide.prev();
        }
    });
})();
