#!/usr/bin/env python3
"""
Small launcher to open a Klampt visualization in a separate process.

This script is intended to be invoked from GUI tools (like `examine_pcs.py`)
so the Qt QApplication is created in the process' main thread and avoids
warnings/errors about creating QApplication from another thread.
"""
import argparse
import numpy as np
import os
import time
from klampt import vis
from klampt.math import se3
from klampt import GeometricPrimitive
from adf.manipulator import Manipulator


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--manip', required=True, help='manipulator name')
    p.add_argument('--anchors', required=True, help='path to .npy anchors file')
    p.add_argument('--watch', action='store_true', help='poll the anchors file and update visualization in realtime')
    p.add_argument('--interval', type=float, default=0.5, help='polling interval (seconds) when --watch is used')
    p.add_argument('--focus-tip', action='store_true', help='use focus_tip in IK')
    return p


def main():
    args = build_parser().parse_args()
    if not os.path.exists(args.anchors):
        raise FileNotFoundError(args.anchors)
    anchors = np.load(args.anchors)
    manip = Manipulator(args.manip, verbose=False)

    # If watch mode, create a Klampt viewport once and poll for updates
    if args.watch:
        vis.setWindowTitle(f"Klampt - {args.manip}")
        vis.setBackgroundColor(1, 1, 1)
        vis.add('world', se3.identity(), fancy=True, length=0.05, width=0.004, hide_label=True)
        vis.add('robot', manip.robot, hide_label=True)
        vis.setColor('robot', 0.7, 0.6, 0.6)

        # add anchor spheres
        anchor_names = []
        for i in range(anchors.shape[0]):
            name = f'A_{i:02d}'
            anc = GeometricPrimitive()
            anc.setSphere(anchors[i], 0.005)
            vis.add(name, anc, hide_label=True)
            # use manipulator palette if available
            try:
                vis.setColor(name, *manip.colors[i])
            except Exception:
                pass
            anchor_names.append(name)

        # show non-blocking
        vis.show()

        last_mtime = None
        try:
            while True:
                print('Checking for anchor updates...')
                try:
                    mtime = os.path.getmtime(args.anchors)
                except Exception:
                    mtime = None
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        anchors = np.load(args.anchors)
                    except Exception as e:
                        print('Failed loading anchors file:', e)
                        time.sleep(args.interval)
                        continue
                    # run IK and update robot config
                    try:
                        manip.inverse_kinematic(anchors, focus_tip=args.focus_tip)
                    except Exception as e:
                        print('inverse_kinematic failed:', e)
                    # update anchor visuals
                    for i in range(anchors.shape[0]):
                        name = anchor_names[i]
                        try:
                            vis.remove(name)
                        except Exception:
                            pass
                        anc = GeometricPrimitive()
                        anc.setSphere(anchors[i], 0.005)
                        vis.add(name, anc, hide_label=True)
                        try:
                            vis.setColor(name, *manip.colors[i])
                        except Exception:
                            pass
                    # request a redraw
                    try:
                        vis.show()
                    except Exception:
                        pass
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print('Exiting watch mode')
        return

    # non-watch mode: one-shot visualize and exit
    try:
        manip.inverse_kinematic(anchors, focus_tip=args.focus_tip)
    except Exception as e:
        print('inverse_kinematic failed:', e)
    try:
        manip.vis_model()
    except Exception as e:
        print('vis_model failed:', e)


if __name__ == '__main__':
    main()
