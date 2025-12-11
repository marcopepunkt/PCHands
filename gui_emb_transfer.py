from flask import Flask, render_template_string, request, jsonify
import numpy as np
from adf.manipulator import Manipulator
from adf.mano_hand import ManoHand
import threading
import meshcat

app = Flask(__name__)

# Initialize
print("Initializing...")
shared_viewer = meshcat.Visualizer()
shared_viewer.open()
shared_viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 4.0)


manip_a = Manipulator(Manipulator.names[-1], verbose=False, fixed_base=True, viewer = shared_viewer)
mano_hand = ManoHand(flat_hand=True, use_pca=True, n_comp=10, viewer = shared_viewer, verbose=False)
dof = manip_a.dof_tendons
print("the dof is:", dof)
# State
current_joints = np.zeros(dof)
enable_ik = True
lock = threading.Lock()

# Compact HTML with sliders
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Manipulator Control - {{ dof }} DOF</title>
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
            max-width: 1200px;
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
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        .sliders {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 12px;
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            background: #fafafa;
            border-radius: 8px;
        }
        .slider-row {
            display: grid;
            grid-template-columns: 60px 1fr 70px;
            gap: 10px;
            align-items: center;
            padding: 10px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Manipulator Control Panel</h1>
        <div class="meshcat-links">
            📺 <strong>3D Visualizations:</strong>
            <a href="http://localhost:7005/static/" target="_blank">Manipulator</a>
            <a href="http://localhost:7006/static/" target="_blank">MANO Hand</a>
            <span style="color:#666;">(Open in new tabs)</span>
        </div>
        <div class="status" id="status">Ready - {{ dof }} DOF</div>
        <div class="controls">
            <button class="btn-reset" onclick="reset()">↺ Reset</button>
            <button class="btn-save" onclick="save()">💾 Save</button>
            <button class="btn-load" onclick="load()">📂 Load</button>
            <button class="btn-random" onclick="random()">🎲 Random</button>
            <button class="btn-export" onclick="exportRefBase()">📤 Export Reference Base</button>
            <label>
                <input type="checkbox" id="ik" checked onchange="toggleIK()">
                <span>Enable IK Transfer</span>
            </label>
        </div>
        <div class="sliders" id="sliders"></div>
    </div>
    <script>
        const dof = {{ dof }};
        let vals = new Array(dof).fill(0);
        let updateTimeout = null;
        
        // Generate sliders
        const container = document.getElementById('sliders');
        for (let i = 0; i < dof; i++) {
            const row = document.createElement('div');
            row.className = 'slider-row';
            row.innerHTML = `
                <span class="slider-label">J${i.toString().padStart(2, '0')}</span>
                <input type="range" min="0" max="1" step="0.01" value="0" 
                       id="s${i}" oninput="updateSlider(${i}, this.value)">
                <span class="slider-value" id="v${i}">0.00</span>
            `;
            container.appendChild(row);
        }
        
        function updateSlider(i, value) {
            vals[i] = parseFloat(value);
            document.getElementById(`v${i}`).textContent = parseFloat(value).toFixed(2);
            
            // Debounce updates for performance
            clearTimeout(updateTimeout);
            updateTimeout = setTimeout(() => {
                fetch('/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({joints: vals})
                });
            }, 50);
        }
        
        function reset() {
            for (let i = 0; i < dof; i++) {
                vals[i] = 0;
                document.getElementById(`s${i}`).value = 0;
                document.getElementById(`v${i}`).textContent = '0.00';
            }
            updateSlider(0, 0);
            document.getElementById('status').textContent = '↺ Reset all joints';
        }
        
        function save() {
            fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({joints: vals})
            })
            .then(r => r.json())
            .then(d => {
                document.getElementById('status').textContent = d.msg;
                setTimeout(() => {
                    document.getElementById('status').textContent = `Ready - ${dof} DOF`;
                }, 2000);
            });
        }
        
        function load() {
            fetch('/load')
            .then(r => r.json())
            .then(d => {
                if (d.joints) {
                    for (let i = 0; i < dof; i++) {
                        vals[i] = d.joints[i];
                        document.getElementById(`s${i}`).value = vals[i];
                        document.getElementById(`v${i}`).textContent = vals[i].toFixed(2);
                    }
                    updateSlider(0, 0);
                }
                document.getElementById('status').textContent = d.msg;
                setTimeout(() => {
                    document.getElementById('status').textContent = `Ready - ${dof} DOF`;
                }, 2000);
            });
        }
        
        function random() {
            for (let i = 0; i < dof; i++) {
                const v = Math.random(); // Range 0 to 1
                vals[i] = v;
                document.getElementById(`s${i}`).value = v;
                document.getElementById(`v${i}`).textContent = v.toFixed(2);
            }
            updateSlider(0, 0);
            document.getElementById('status').textContent = '🎲 Random pose generated';
        }
        
        function exportRefBase() {
            fetch('/export_ref_base', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({joints: vals})
            })
            .then(r => r.json())
            .then(d => {
                document.getElementById('status').textContent = d.msg;
                setTimeout(() => {
                    document.getElementById('status').textContent = `Ready - ${dof} DOF`;
                }, 2000);
            });
        }
        
        function toggleIK() {
            const enabled = document.getElementById('ik').checked;
            fetch('/ik', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: enabled})
            })
            .then(() => {
                const status = enabled ? 'ON' : 'OFF';
                document.getElementById('status').textContent = `IK Transfer: ${status}`;
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML, dof=dof)

@app.route('/update', methods=['POST'])
def update():
    global current_joints
    with lock:
        current_joints = np.array(request.json['joints'])
        print(f"🔧 Updated joints: {current_joints}")
        manip_a.forward_kinematic(current_joints, normalized=True)
        manip_a.vis_model()
        
        if enable_ik:
            anchors = manip_a.get_anchor()
            mano_hand.inverse_kinematic(
                anchors, niter=2000, hotstart=False,temporal_smoothing=False, floating_base=True,
                th_loss=0.0001, focus_tip=True, visualize=True, lr=1e-2
            )
            mano_hand.vis_model()
    return jsonify({'status': 'ok'})

@app.route('/ik', methods=['POST'])
def toggle_ik():
    global enable_ik
    enable_ik = request.json['enabled']
    return jsonify({'status': 'ok'})

@app.route('/save', methods=['POST'])
def save():
    joints = np.array(request.json['joints'])
    np.save('saved_pose.npy', joints)
    print(f"💾 Saved pose: {joints}")
    return jsonify({'msg': '✓ Pose saved successfully'})

@app.route('/load')
def load():
    try:
        joints = np.load('saved_pose.npy')
        print(f"📂 Loaded pose: {joints}")
        return jsonify({'joints': joints.tolist(), 'msg': '✓ Pose loaded successfully'})
    except FileNotFoundError:
        return jsonify({'msg': '✗ No saved pose found'})

@app.route('/export_ref_base', methods=['POST'])
def export_ref_base():
    """Callback for exporting reference base"""
    # joints = np.array(request.json['joints'])
    
    mano_hand.save_calib_eef()
    
    return jsonify({'msg': '✓ Reference base exported successfully'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print(f"🤖 Manipulator Control Server")
    print("="*70)
    print(f"   DOF: {dof}")
    print(f"   Control Panel: http://localhost:5000")
    print("="*70)
    print("\n📡 SSH Port Forwarding:")
    print("   ssh -L 5000:localhost:5000 -L 7005:localhost:7005 -L 7006:localhost:7006 user@host")
    print("="*70 + "\n")
    
    # Initial visualization
    manip_a.forward_kinematic(current_joints, normalized=False)
    manip_a.vis_model()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)