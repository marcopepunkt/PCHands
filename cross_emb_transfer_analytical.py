import numpy as np
from adf.manipulator import Manipulator
from adf.mano_hand import ManoHand


# list all available manipulator names
print(Manipulator.names)
# create manipulator instance

manip_a = Manipulator(Manipulator.names[-2], verbose=False, fixed_base=True)
mano_hand = ManoHand(flat_hand=True, calibrated=False, use_pca=True, n_comp=7)
manip_b = Manipulator(Manipulator.names[-1], verbose=False, fixed_base=False)


len_traj = 20
q = np.linspace(0, 1, num=len_traj)
manip_a_dof = manip_a.dof

trajectory_a = np.zeros((len_traj, manip_a_dof))
for i in range(manip_a_dof):
    trajectory_a[:, i] = q  

path = "data/cross_emb_transfer/"

import cv2
import os

video_name_a = os.path.join(path, 'manip_a_video.mp4')
video_name_b = os.path.join(path, 'manip_b_video.mp4')
video_name_mano = os.path.join(path, 'mano_video.mp4')

width = 640
height = 480

video_a = cv2.VideoWriter(video_name_a, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
video_b = cv2.VideoWriter(video_name_b, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
video_mano = cv2.VideoWriter(video_name_mano, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))

for i in range(len_traj):
    
    trajectory_a[i, :] = manip_a.denormalize_joint(trajectory_a[i, :])
    manip_a.forward_kinematic(trajectory_a[i, :])
    anchors_a =manip_a.get_anchor()
    img_manip_a = manip_a.vis_model()#save="return")
    video_a.write(img_manip_a)
     
    print(f"Transferring step {i}...")
    mano_pose = mano_hand.inverse_kinematic(anchors_a, niter=1000, hotstart=True, floating_base=True)
    img_mano = mano_hand.vis_model(save="return")
    video_mano.write(img_mano)

    anchors_mano = mano_hand.get_anchor()
    manip_b_pose = manip_b.inverse_kinematic(anchors_mano)
    img_manip_b = manip_b.vis_model(save="return")
    video_b.write(img_manip_b)



# Create a Video out of the saved images
video_a.release() 
video_b.release()   
video_mano.release()

