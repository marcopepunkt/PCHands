import numpy as np
from adf.manipulator import Manipulator
from adf.mano_hand import ManoHand


# list all available manipulator names
print(Manipulator.names)
# create manipulator instance

manip_a = Manipulator(Manipulator.names[-2], verbose=False, fixed_base=True)
mano_hand = ManoHand(flat_hand=True, use_pca=True, n_comp=10)
manip_b = Manipulator(Manipulator.names[-2], verbose=False, fixed_base=False)

import h5py

# Path to your HDF5 file
# h5_file = "/home/marco/srl_il_lfd/data_test/orca/0004.h5"
h5_file = "/home/marco/srl_il_lfd/data_test/mimic/0000.h5"

with h5py.File(h5_file, 'r') as f:
    actions_hand = f['actions_hand'][:]  # shape: (108, 16)

# Number of samples you want
full_len = actions_hand.shape[0]
len_traj = full_len

# Compute evenly spaced indices
indices = np.linspace(0, full_len - 1, len_traj, dtype=int)

# Sample the trajectory
trajectory_a = actions_hand[indices, :]

print("Sampled trajectory shape:", trajectory_a.shape)
# q = np.linspace(0, 0.5, num=len_traj)
# Load a trajctory for for manipulator A with hdf5 file


# manip_a_dof = manip_a.dof_tendons

# trajectory_a = np.zeros((len_traj, manip_a_dof))
# for i in range(manip_a_dof):
#     trajectory_a[:, i] = q  

path = "data/cross_emb_transfer/"

import cv2
import os

video_name_a = os.path.join(path, 'manip_a_video.mp4')
video_name_b = os.path.join(path, 'manip_b_video.mp4')
video_name_mano = os.path.join(path, 'mano_video.mp4')

width = 2700
height = 1500

video_a = cv2.VideoWriter(video_name_a, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
video_b = cv2.VideoWriter(video_name_b, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
video_mano = cv2.VideoWriter(video_name_mano, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))

reconstructed_trajectory_b = np.zeros((len_traj, manip_b.dof_tendons))

for i in range(len_traj):
    
    # trajectory_a[i, :] = manip_a.denormalize_joint(trajectory_a[i, :])
    manip_a.forward_kinematic(trajectory_a[i, :], normalized=False)
    anchors_a =manip_a.get_anchor()
    img_manip_a = manip_a.vis_model(return_image=True)
    # img_manip_a.save(path+f"temp_a{i}.png")

    img_manip_a = cv2.cvtColor(np.array(img_manip_a), cv2.COLOR_RGBA2BGR)
    if img_manip_a.shape[1] != width or img_manip_a.shape[0] != height:
        img_manip_a = cv2.resize(img_manip_a, (width, height), interpolation=cv2.INTER_LINEAR)
    video_a.write(img_manip_a)

    print(f"Transferring step {i}...")
    mano_pose = mano_hand.inverse_kinematic(anchors_a, niter=2000, hotstart=True, floating_base=True, th_loss=0.00007, focus_tip=True, visualize=False, lr = 2e-2, temporal_smoothing = False)
    img_mano = mano_hand.vis_model(return_image=True)
    # img_mano.save(path+f"temp_mano{i}.png")

    img_mano = cv2.cvtColor(np.array(img_mano), cv2.COLOR_RGBA2BGR)
    if img_mano.shape[1] != width or img_mano.shape[0] != height:
        img_mano = cv2.resize(img_mano, (width, height), interpolation=cv2.INTER_LINEAR)
    video_mano.write(img_mano)

    mano_keypoints = mano_hand.get_mano_keypoints()

    manip_b_pose = manip_b.inverse_kinematic(mano_keypoints)#, th_loss = 0.0, visualize = True, base_pos_weight=0.1, base_rot_weight=0.1, niter=1000, focus_tip=True)
    img_manip_b = manip_b.vis_model(return_image=True)
    # img_manip_b.save(path+f"temp_b{i}.png")
    img_manip_b = cv2.cvtColor(np.array(img_manip_b), cv2.COLOR_RGBA2BGR)
        
    if img_manip_b.shape[1] != width or img_manip_b.shape[0] != height:
        img_manip_b = cv2.resize(img_manip_b, (width, height), interpolation=cv2.INTER_LINEAR)

    video_b.write(img_manip_b)
    reconstructed_trajectory_b[i, :] = manip_b.get_joint(use_scheme = True)



# Create a Video out of the saved images
video_a.release() 
video_b.release()   
video_mano.release()

print("Videos saved.")

# Save the reconstructed and base trajectories as .npy files
np.save(path+'trajectory_a.npy', np.array(trajectory_a))
np.save(path+'reconstructed_trajectory_b.npy', reconstructed_trajectory_b)


# Plot the trajectories
import matplotlib.pyplot as plt
for j in range(manip_a.dof_tendons):
    plt.figure()
    plt.plot(trajectory_a[:, j], label='Manipulator A')
    plt.plot(reconstructed_trajectory_b[:, j], label='Manipulator B')
    plt.title(f'Joint {j} Trajectory Transfer')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Value')
    plt.legend()
    plt.savefig(path+f'joint_{j}_trajectory_transfer.png')
    plt.close()