import re

with open("index.html", "r") as f:
    content = f.read()

# 1. Add Radial Lock UI
content = content.replace(
    '<select id="camera-focus-select" onchange="changeCameraFocus()" style="flex:1;">',
    '<select id="camera-focus-select" onchange="changeCameraFocus()" style="flex:1;">'
)
content = content.replace(
    '<optgroup id="sat-focus-group" label="Satellites"></optgroup>\n                </select>\n            </div>',
    '<optgroup id="sat-focus-group" label="Satellites"></optgroup>\n                </select>\n            </div>\n            <div class="row" style="margin-top:4px;">\n                <label style="font-size:10px; color:#94a3b8;"><input type="checkbox" id="camera-radial-lock" checked> Radial View Lock</label>\n            </div>'
)

# 2. Add Floating Stat Box UI (3D label)
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

# 3. Add Drag-to-Orbit Mode Button
drag_btn = '<button type="button" class="btn" style="background:#8b5cf6;" onclick="toggleDragOrbitMode()">✨ Drag to Add Orbit</button>\n                '
content = content.replace('<button type="button" class="btn" onclick="openSatModal()">+ Add Satellite</button>', drag_btn + '<button type="button" class="btn" onclick="openSatModal()">+ Add Satellite</button>')

with open("index.html", "w") as f:
    f.write(content)
