#!/usr/bin/env python3
"""
Lightweight GUI to explore principal components for a manipulator.

Usage:
    python examine_pcs.py --manip ergocub_hand_right --n 6

Requirements:
    - Matplotlib
    - numpy
    - the repository's Python path (run from project root)
    - `adf` PCA artifacts (run `adf/calib_eef.py` if missing)

This script creates sliders for the first N principal components and updates
an inline 3D scatter of the 22 anchor points. A button sends the current
anchor positions to the manipulator: `inverse_kinematic` is attempted and
`vis_model` is opened to inspect the resulting robot pose.
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ensure repo package imports work when running from project root
from adf.manipulator import Manipulator


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--manip', type=str, default=None,
                   help='Manipulator name from Manipulator.names (or index)')
    p.add_argument('--n', type=int, default=6,
                   help='Number of principal components to expose as sliders')
    return p


def main():
    args = build_parser().parse_args()

    # choose manipulator
    if args.manip is None:
        print('Available manipulators:')
        print(Manipulator.names)
        print('Run with --manip <name> or index to select one.')
        return

    # allow numeric index
    if args.manip.isdigit():
        manip_name = Manipulator.names[int(args.manip)]
    else:
        manip_name = args.manip

    manip = Manipulator(manip_name, verbose=False)

    if manip.pca is None or manip.stats is None:
        print('PCA artifacts not found for manipulator "{}".'.format(manip_name))
        print('Run `cd adf && python calib_eef.py` to generate `pca.pth` and `stats.npy`.')
        return

    n_components = manip.pca.num_components
    n_sliders = min(args.n, n_components)

    # initial principal components: zeros (mean)
    pc = np.zeros(n_components, dtype=float)

    # plot setup
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # leave room for dynamic sliders on the right and buttons on the left
    plt.subplots_adjust(left=0.25, bottom=0.05, top=0.92)

    anchors = manip.pc_to_anchor(pc[:n_components])
    scat = ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], c='C0', s=40)
    # annotate points with their index
    text_objs = []
    for i in range(anchors.shape[0]):
        text_objs.append(ax.text(anchors[i, 0], anchors[i, 1], anchors[i, 2], str(i)))

    ax.set_title(f'Anchors (manip: {manip_name})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # sliders area (dynamic layout)
    slider_axes = []
    sliders = []
    # preferred slider height/gap (will shrink if too many sliders requested)
    pref_slider_height = 0.03
    gap = 0.01
    # maximum vertical area reserved for sliders
    max_slider_area = 0.75  # fraction of figure height
    total_needed = n_sliders * (pref_slider_height + gap) - gap
    if total_needed > max_slider_area:
        # shrink slider height to fit into max area
        slider_height = (max_slider_area - (n_sliders - 1) * gap) / n_sliders
        if slider_height <= 0:
            slider_height = pref_slider_height * 0.5
    else:
        slider_height = pref_slider_height

    start_y = 0.85  # top y position for first slider
    # default slider range in std-dev units [-3, 3]
    for i in range(n_sliders):
        y = start_y - i * (slider_height + gap)
        # ensure slider stays above the bottom margin
        if y < 0.06:
            y = 0.06
        ax_slider = fig.add_axes([0.25, y, 0.65, slider_height])
        s = Slider(ax_slider, f'PC{i}', -3.0, 3.0, valinit=0.0)
        sliders.append(s)


    def update_plot(val=None):
        # construct full pc vector padded with zeros
        pc_local = np.zeros(n_components, dtype=float)
        pc_local[:n_sliders] = [s.val for s in sliders]
        try:
            anchors = manip.pc_to_anchor(pc_local)
        except Exception as e:
            print('Error converting PC to anchors:', e)
            return
        
        # Save the updated anchors to the watch file if in watch mode
        if 'watch_file' in globals():
            try:
                np.save(watch_file, anchors)
            except Exception as e:
                print('Error saving anchors to watch file:', e) 
                
        # update scatter positions
        scat._offsets3d = (anchors[:, 0], anchors[:, 1], anchors[:, 2])
        for i, t in enumerate(text_objs):
            t.set_position((anchors[i, 0], anchors[i, 1]))
            # z-order update via set_3d_properties
            try:
                t.set_3d_properties(anchors[i, 2])
            except Exception:
                pass
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update_plot)


    # Buttons: send to klampt, randomize, reset
    ax_button_send = fig.add_axes([0.025, 0.25, 0.15, 0.06])
    btn_send = Button(ax_button_send, 'Send → Klampt')
    ax_button_send_watch = fig.add_axes([0.025, 0.32, 0.15, 0.06])
    btn_send_watch = Button(ax_button_send_watch, 'Send → Klampt (watch)')

    ax_button_rand = fig.add_axes([0.025, 0.18, 0.15, 0.06])
    btn_rand = Button(ax_button_rand, 'Random')

    ax_button_reset = fig.add_axes([0.025, 0.11, 0.15, 0.06])
    btn_reset = Button(ax_button_reset, 'Reset')


    def on_random(event):
        for s in sliders:
            s.set_val(np.random.uniform(-2.0, 2.0))

    def on_reset(event):
        for s in sliders:
            s.set_val(0.0)

    def on_send(event):
        # build pc and anchors, write to temp file and spawn separate process
        import tempfile
        import subprocess
        import shlex

        pc_local = np.zeros(n_components, dtype=float)
        pc_local[:n_sliders] = [s.val for s in sliders]
        anchors = manip.pc_to_anchor(pc_local)

        # write anchors to a temp npy file
        tmp = tempfile.NamedTemporaryFile(prefix='anchors_', suffix='.npy', delete=False)
        np.save(tmp.name, anchors)
        tmp.close()

        cmd = f"python launch_klampt.py --manip {shlex.quote(manip_name)} --anchors {shlex.quote(tmp.name)}"
        try:
            # spawn detached process so GUI remains responsive
            subprocess.Popen(cmd, shell=True)
        except Exception as e:
            print('Failed to spawn klampt launcher:', e)


    def on_send_watch(event):
        import tempfile
        import subprocess
        import shlex

        pc_local = np.zeros(n_components, dtype=float)
        pc_local[:n_sliders] = [s.val for s in sliders]
        anchors = manip.pc_to_anchor(pc_local)

        tmp = tempfile.NamedTemporaryFile(prefix='anchors_', suffix='.npy', delete=False)
        np.save(tmp.name, anchors)
        tmp.close()

        cmd = f"python launch_klampt.py --manip {shlex.quote(manip_name)} --anchors {shlex.quote(tmp.name)} --watch"
        global watch_file
        watch_file = tmp.name  # store for later updates
        try:
            subprocess.Popen(cmd, shell=True)
        except Exception as e:
            print('Failed to spawn klampt watcher:', e)


    btn_rand.on_clicked(on_random)
    btn_reset.on_clicked(on_reset)
    btn_send.on_clicked(on_send)
    btn_send_watch.on_clicked(on_send_watch)

    plt.show()


if __name__ == '__main__':
    main()
