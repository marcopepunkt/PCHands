from adf.manipulator import Manipulator
from adf.mano_hand import ManoHand

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ref_manip = ['orca_v1']  # Fixed-base manipulator as reference
    target_manips_names = ['mano_hand']#'orca_v1']
 
    # Set all calib values to 0 for the target manipulators
    target_manips = []
    for name in target_manips_names:
        if name == 'mano_hand':
            manip = ManoHand(flat_hand=True, calibrated=False, use_pca=True, n_comp=7)
        else:
            manip = Manipulator(name, fixed_base=False)
            # print(f"Original {manip.name} calib: {manip.get_base()}")
            manip.save_calib_eef([0,0,0], [0,0,0])  # No translation, no rotation
            manip.cleanup()  # Clean up temporary files
            del manip
            manip = Manipulator(name, fixed_base=False) # Reload to apply zero calib

        target_manips.append(manip)

    # target_manips = [Manipulator(name, fixed_base=False) for name in target_manips_names]
    ref_manip = Manipulator(ref_manip[0], fixed_base=True) 

    len_traj = 10
    q = np.linspace(0.1, 0.6, num=len_traj)
    ref_manip_dof = ref_manip.dof

    ref_manip_trajectory = np.zeros((len_traj, ref_manip_dof))
    for i in range(ref_manip_dof):
        ref_manip_trajectory[:, i] = q  

    # Compute the Anchor Trajectory
    anchors_ref_list = []
    for i in range(len_traj):
        ref_manip_trajectory[i, :] = ref_manip.denormalize_joint(ref_manip_trajectory[i, :])
        ref_manip.forward_kinematic(ref_manip_trajectory[i, :])
        anchors_ref = ref_manip.get_anchor()
        anchors_ref_list.append(anchors_ref)
        ref_manip.vis_model(save=f"data/retargeter/ref_manip_{i:02d}.png")
    

    for manip in target_manips:
        print(f"Retargeting to {manip.name}...")
        base_pos_list = []
        base_rot_list = []
        for i in range(len_traj):
            print(f"Step {i}...")
            manip.inverse_kinematic(anchors_ref_list[i], focus_tip=False, floating_base=True, visual=True, niter = 2000)
            vis = manip.vis_model(save=f"data/retargeter/{manip.name}_step_{i:02d}.png")
            
            
            
            pose = manip.get_base()
            base_rot_list.append(pose[0:3])
            base_pos_list.append(pose[3:6])
        base_pos_array = np.array(base_pos_list)
        base_rot_array = np.array(base_rot_list)
        # Compute the mean base position and rotation
        mean_base_pos = np.mean(base_pos_array, axis=0)
        mean_base_rot = np.mean(base_rot_array, axis=0)


        # Visualize the positions as a scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(base_pos_array[:, 0], base_pos_array[:, 1], base_pos_array[:, 2], c='b', marker='o')
        ax.scatter(mean_base_pos[0], mean_base_pos[1], mean_base_pos[2], c='r', marker='^', s=100)
        ax.set_title(f'Base Positions for {manip.name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(f"data/retargeter/{manip.name}_base_positions.png")
        plt.close()

        # Visualize the rotations as arrows in 3D space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(base_rot_array.shape[0]):
            ax.quiver(mean_base_pos[0], mean_base_pos[1], mean_base_pos[2],
                      base_rot_array[i, 0], base_rot_array[i, 1], base_rot_array[i, 2],
                      length=0.1, color='b', alpha=0.5)
        ax.quiver(mean_base_pos[0], mean_base_pos[1], mean_base_pos[2],
                  mean_base_rot[0], mean_base_rot[1], mean_base_rot[2],
                  length=0.2, color='r', alpha=1.0)
        ax.set_title(f'Base Rotations for {manip.name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')  
        plt.savefig(f"data/retargeter/{manip.name}_base_rotations.png")
        plt.close()

        # Save the base positions and rotations to calib_eef.yaml file
        manip.save_calib_eef(mean_base_pos, mean_base_rot) 

        