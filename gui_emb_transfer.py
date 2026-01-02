from flask import Flask, render_template_string, request, jsonify
import numpy as np
from adf.manipulator import Manipulator
from adf.mano_hand import ManoHand
import threading
import meshcat


import signal
import sys

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\n\n" + "="*70)
    print("🛑 Shutting down gracefully...")
    print("="*70)
    try:
        # Close meshcat viewers
        if hasattr(shared_viewer, 'close'):
            shared_viewer.close()
        print("✓ Cleaned up resources")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")
    print("👋 Goodbye!\n")
    sys.exit(0)

app = Flask(__name__)

# Initialize
print("Initializing...")
shared_viewer = meshcat.Visualizer()
shared_viewer.open()
shared_viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 4.0)


dof_mano = 15

manip_a = Manipulator(Manipulator.names[-1], verbose=False, fixed_base=True, viewer=shared_viewer)
mano_hand = ManoHand(flat_hand=True, use_pca=True, n_comp=dof_mano, viewer=shared_viewer, verbose=False)

dof_robot = manip_a.dof_tendons
print(f"Robot DOF: {dof_robot}, MANO DOF: {dof_mano}")

# State
current_joints_robot = np.zeros(dof_robot)
current_joints_mano = np.zeros(dof_mano)
enable_ik = True
ik_mode = "robot_to_hand"  # "robot_to_hand" or "hand_to_robot"
lock = threading.Lock()

# Enhanced HTML with dual sliders
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Bidirectional Control - Robot ⇄ Hand</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 24px;
        }
        .meshcat-links {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
        }
        .meshcat-links a {
            color: #1976D2;
            text-decoration: none;
            font-weight: bold;
            margin-right: 20px;
        }
        .meshcat-links a:hover { text-decoration: underline; }
        .status {
            padding: 12px;
            background: #4CAF50;
            color: white;
            border-radius: 6px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }
        .btn-reset { background: #ff9800; color: white; }
        .btn-reset:hover { background: #f57c00; }
        .btn-save { background: #4CAF50; color: white; }
        .btn-save:hover { background: #45a049; }
        .btn-load { background: #2196F3; color: white; }
        .btn-load:hover { background: #1976D2; }
        .btn-random { background: #9C27B0; color: white; }
        .btn-random:hover { background: #7B1FA2; }
        .btn-export { background: #00BCD4; color: white; }
        .btn-export:hover { background: #0097A7; }
        button:active { transform: scale(0.95); }
        label {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 15px;
            background: white;
            border-radius: 6px;
            cursor: pointer;
        }
        input[type="checkbox"], input[type="radio"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        /* Mode Selector */
        .mode-selector {
            display: flex;
            gap: 15px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #ffc107;
        }
        .mode-selector label {
            background: transparent;
            padding: 5px 10px;
        }
        .mode-selector input[type="radio"]:checked + span {
            font-weight: bold;
            color: #ff6f00;
        }
        
        /* Panel Layout */
        .panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .panel {
            background: #fafafa;
            padding: 15px;
            border-radius: 8px;
        }
        .panel h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: #555;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .panel.active {
            background: #e8f5e9;
            border: 2px solid #4CAF50;
        }
        .panel.inactive {
            opacity: 0.6;
        }
        
        .sliders {
            display: grid;
            gap: 10px;
            max-height: 450px;
            overflow-y: auto;
            padding: 10px;
            background: white;
            border-radius: 6px;
        }
        .slider-row {
            display: grid;
            grid-template-columns: 60px 1fr 70px;
            gap: 10px;
            align-items: center;
            padding: 8px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .slider-label {
            font-weight: bold;
            color: #555;
            font-size: 13px;
        }
        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
            background: linear-gradient(to right, #667eea, #764ba2);
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: white;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        input[type="range"]::-moz-range-thumb {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: white;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            border: none;
        }
        .slider-value {
            font-family: 'Courier New', monospace;
            font-weight: bold;
            color: #666;
            text-align: right;
        }
        
        /* Responsive */
        @media (max-width: 1100px) {
            .panels {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 ⇄ 👋 Bidirectional Control Panel</h1>
        <div class="meshcat-links">
            📺 <strong>3D Visualizations:</strong>
            <a href="http://localhost:7005/static/" target="_blank">Manipulator</a>
            <a href="http://localhost:7006/static/" target="_blank">MANO Hand</a>
        </div>
        <div class="status" id="status">Ready - Robot: {{ dof_robot }} DOF, MANO: {{ dof_mano }} DOF</div>
        
        <div class="mode-selector">
            <strong>🔄 IK Mode:</strong>
            <label>
                <input type="radio" name="mode" value="robot_to_hand" checked onchange="changeMode(this.value)">
                <span>🤖 → 👋 Robot to Hand</span>
            </label>
            <label>
                <input type="radio" name="mode" value="hand_to_robot" onchange="changeMode(this.value)">
                <span>👋 → 🤖 Hand to Robot</span>
            </label>
            <label>
                <input type="checkbox" id="ik" checked onchange="toggleIK()">
                <span>Enable IK Transfer</span>
            </label>
        </div>
        
        <div class="controls">
            <button class="btn-reset" onclick="reset()">↺ Reset All</button>
            <button class="btn-save" onclick="save()">💾 Save</button>
            <button class="btn-load" onclick="load()">📂 Load</button>
            <button class="btn-random" onclick="random()">🎲 Random</button>
            <button class="btn-export" onclick="exportRefBase()">📤 Export Reference Base</button>
        </div>
        
        <div class="panels">
            <div class="panel" id="robot-panel">
                <h2>🤖 Manipulator ({{ dof_robot }} DOF)</h2>
                <div class="sliders" id="robot-sliders"></div>
            </div>
            <div class="panel" id="mano-panel">
                <h2>👋 MANO Hand ({{ dof_mano }} PCA)</h2>
                <div class="sliders" id="mano-sliders"></div>
            </div>
        </div>
    </div>
    
    <script>
        const dofRobot = {{ dof_robot }};
        const dofMano = {{ dof_mano }};
        let valsRobot = new Array(dofRobot).fill(0);
        let valsMano = new Array(dofMano).fill(0);
        let updateTimeout = null;
        let currentMode = 'robot_to_hand';
        
        // Generate robot sliders
        const robotContainer = document.getElementById('robot-sliders');
        for (let i = 0; i < dofRobot; i++) {
            const row = document.createElement('div');
            row.className = 'slider-row';
            row.innerHTML = `
                <span class="slider-label">J${i.toString().padStart(2, '0')}</span>
                <input type="range" min="0" max="1" step="0.01" value="0" 
                       id="robot_${i}" oninput="updateRobotSlider(${i}, this.value)">
                <span class="slider-value" id="robot_v${i}">0.00</span>
            `;
            robotContainer.appendChild(row);
        }
        
        // Generate MANO sliders
        const manoContainer = document.getElementById('mano-sliders');
        for (let i = 0; i < dofMano; i++) {
            const row = document.createElement('div');
            row.className = 'slider-row';
            row.innerHTML = `
                <span class="slider-label">PC${i.toString().padStart(2, '0')}</span>
                <input type="range" min="-2" max="2" step="0.01" value="0" 
                       id="mano_${i}" oninput="updateManoSlider(${i}, this.value)">
                <span class="slider-value" id="mano_v${i}">0.00</span>
            `;
            manoContainer.appendChild(row);
        }
        
        function updatePanelStates() {
            const robotPanel = document.getElementById('robot-panel');
            const manoPanel = document.getElementById('mano-panel');
            
            if (currentMode === 'robot_to_hand') {
                robotPanel.classList.add('active');
                robotPanel.classList.remove('inactive');
                manoPanel.classList.remove('active');
                manoPanel.classList.add('inactive');
            } else {
                manoPanel.classList.add('active');
                manoPanel.classList.remove('inactive');
                robotPanel.classList.remove('active');
                robotPanel.classList.add('inactive');
            }
        }
        
        function updateRobotSlider(i, value) {
            valsRobot[i] = parseFloat(value);
            document.getElementById(`robot_v${i}`).textContent = parseFloat(value).toFixed(2);
            
            clearTimeout(updateTimeout);
            updateTimeout = setTimeout(() => {
                fetch('/update_robot', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({joints: valsRobot})
                });
            }, 50);
        }
        
        function updateManoSlider(i, value) {
            valsMano[i] = parseFloat(value);
            document.getElementById(`mano_v${i}`).textContent = parseFloat(value).toFixed(2);
            
            clearTimeout(updateTimeout);
            updateTimeout = setTimeout(() => {
                fetch('/update_mano', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({joints: valsMano})
                });
            }, 50);
        }
        
        function changeMode(mode) {
            currentMode = mode;
            fetch('/set_mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            })
            .then(() => {
                updatePanelStates();
                const arrow = mode === 'robot_to_hand' ? '🤖 → 👋' : '👋 → 🤖';
                document.getElementById('status').textContent = `Mode: ${arrow}`;
                setTimeout(() => {
                    document.getElementById('status').textContent = `Ready - Robot: ${dofRobot} DOF, MANO: ${dofMano} DOF`;
                }, 2000);
            });
        }
        
        function reset() {
            for (let i = 0; i < dofRobot; i++) {
                valsRobot[i] = 0;
                document.getElementById(`robot_${i}`).value = 0;
                document.getElementById(`robot_v${i}`).textContent = '0.00';
            }
            for (let i = 0; i < dofMano; i++) {
                valsMano[i] = 0;
                document.getElementById(`mano_${i}`).value = 0;
                document.getElementById(`mano_v${i}`).textContent = '0.00';
            }
            updateRobotSlider(0, 0);
            updateManoSlider(0, 0);
            document.getElementById('status').textContent = '↺ Reset all joints';
        }
        
        function save() {
            fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    robot: valsRobot,
                    mano: valsMano
                })
            })
            .then(r => r.json())
            .then(d => {
                document.getElementById('status').textContent = d.msg;
                setTimeout(() => {
                    document.getElementById('status').textContent = `Ready`;
                }, 2000);
            });
        }
        
        function load() {
            fetch('/load')
            .then(r => r.json())
            .then(d => {
                if (d.robot) {
                    for (let i = 0; i < dofRobot; i++) {
                        valsRobot[i] = d.robot[i];
                        document.getElementById(`robot_${i}`).value = valsRobot[i];
                        document.getElementById(`robot_v${i}`).textContent = valsRobot[i].toFixed(2);
                    }
                }
                if (d.mano) {
                    for (let i = 0; i < dofMano; i++) {
                        valsMano[i] = d.mano[i];
                        document.getElementById(`mano_${i}`).value = valsMano[i];
                        document.getElementById(`mano_v${i}`).textContent = valsMano[i].toFixed(2);
                    }
                }
                updateRobotSlider(0, 0);
                updateManoSlider(0, 0);
                document.getElementById('status').textContent = d.msg;
            });
        }
        
        function random() {
            if (currentMode === 'robot_to_hand') {
                for (let i = 0; i < dofRobot; i++) {
                    const v = Math.random();
                    valsRobot[i] = v;
                    document.getElementById(`robot_${i}`).value = v;
                    document.getElementById(`robot_v${i}`).textContent = v.toFixed(2);
                }
                updateRobotSlider(0, 0);
            } else {
                for (let i = 0; i < dofMano; i++) {
                    const v = (Math.random() * 4) - 2; // -2 to 2
                    valsMano[i] = v;
                    document.getElementById(`mano_${i}`).value = v;
                    document.getElementById(`mano_v${i}`).textContent = v.toFixed(2);
                }
                updateManoSlider(0, 0);
            }
            document.getElementById('status').textContent = '🎲 Random pose generated';
        }
        
        function exportRefBase() {
            fetch('/export_ref_base', {method: 'POST'})
            .then(r => r.json())
            .then(d => document.getElementById('status').textContent = d.msg);
        }
        
        function toggleIK() {
            const enabled = document.getElementById('ik').checked;
            fetch('/ik', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: enabled})
            })
            .then(() => {
                document.getElementById('status').textContent = `IK Transfer: ${enabled ? 'ON' : 'OFF'}`;
            });
        }
        
        // Initialize panel states
        updatePanelStates();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML, dof_robot=dof_robot, dof_mano=dof_mano)

@app.route('/update_robot', methods=['POST'])
def update_robot():
    global current_joints_robot
    with lock:
        current_joints_robot = np.array(request.json['joints'])
        print(f"🤖 Robot joints: {current_joints_robot}")
        manip_a.forward_kinematic(current_joints_robot, normalized=True)
        manip_a.vis_model()
        
        if enable_ik and ik_mode == "robot_to_hand":
            anchors = manip_a.get_anchor()
            mano_hand.inverse_kinematic(
                anchors, niter=2000, hotstart=False, temporal_smoothing=False, 
                floating_base=True, th_loss=0.0001, focus_tip=True, 
                visualize=True, lr=8e-2
            )
            mano_hand.vis_model()
    return jsonify({'status': 'ok'})

@app.route('/update_mano', methods=['POST'])
def update_mano():
    global current_joints_mano
    with lock:
        current_joints_mano = np.array(request.json['joints'])
        print(f"👋 MANO joints: {current_joints_mano}")
        
        # Update MANO hand with PCA coefficients
        mano_hand.forward_kinematic(current_joints_mano, normalized=False)
        mano_hand.vis_model()
        
        if enable_ik and ik_mode == "hand_to_robot":
            # Get anchor points from MANO hand
            mano_keypoints = mano_hand.get_mano_keypoints()
            print("got the keypoints")
            # Now visualize these mano key

            # Solve IK for the manipulator
            manip_a.inverse_kinematic(
                mano_keypoints, visualize=True)
            manip_a.vis_model()
    return jsonify({'status': 'ok'})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global ik_mode
    ik_mode = request.json['mode']
    print(f"🔄 IK Mode: {ik_mode}")
    return jsonify({'status': 'ok'})

@app.route('/ik', methods=['POST'])
def toggle_ik():
    global enable_ik
    enable_ik = request.json['enabled']
    print(f"IK enabled: {enable_ik}")
    return jsonify({'status': 'ok'})

@app.route('/save', methods=['POST'])
def save():
    data = request.json
    np.savez('saved_pose.npz', 
             robot=np.array(data['robot']), 
             mano=np.array(data['mano']))
    print(f"💾 Saved both poses")
    return jsonify({'msg': '✓ Both poses saved successfully'})

@app.route('/load')
def load():
    try:
        data = np.load('saved_pose.npz')
        robot_joints = data['robot']
        mano_joints = data['mano']
        print(f"📂 Loaded poses - Robot: {robot_joints}, MANO: {mano_joints}")
        return jsonify({
            'robot': robot_joints.tolist(),
            'mano': mano_joints.tolist(),
            'msg': '✓ Both poses loaded successfully'
        })
    except FileNotFoundError:
        return jsonify({'msg': '✗ No saved pose found'})

@app.route('/export_ref_base', methods=['POST'])
def export_ref_base():
    mano_hand.save_calib_eef()
    return jsonify({'msg': '✓ Reference base exported successfully'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Bidirectional ADF Control Server is running!")
    print("   ssh -L 5000:localhost:5000 -L 7005:localhost:7005 -L 7006:localhost:7006 user@host")
    print("="*70 + "\n")
    
    # Initial visualization
    manip_a.forward_kinematic(current_joints_robot, normalized=False)
    manip_a.vis_model()
    mano_hand.forward_kinematic(current_joints_mano, normalized=False)
    mano_hand.vis_model()
    
    try:
        app.run(host='0.0.0.0', port=4000, debug=False, threaded=True)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
