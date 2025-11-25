import numpy as np
from adf.manipulator import Manipulator


# list all available manipulator names
print(Manipulator.names)
# create manipulator instance

manip_a = Manipulator(Manipulator.names[-3], verbose=False)
manip_b = Manipulator(Manipulator.names[-4], verbose=False)

len_traj = 20
q = np.linspace(0, 1, num=len_traj)
manip_a_dof = manip_a.dof

trajectory_a = np.zeros((len_traj, manip_a_dof))
for i in range(manip_a_dof):
    trajectory_a[:, i] = q  

path = "./cross_emb_transfer/"
for i in range(len_traj):
    trajectory_a[i, :] = manip_a.denormalize_joint(trajectory_a[i, :])
    manip_a.forward_kinematic(trajectory_a[i, :])
    anchors_a =manip_a.get_anchor()
    pcs = manip_a.anchor_to_pc(anchors_a)

    # just keep the first pcs and zero the rest for simplicity
    pcs[3:] = 0.0
    print(f"PCs at step {i}:\n", pcs)

    manip_a.vis_model(save=path+f"manip_a_{i:02d}.png")

    anchors_b =manip_b.pc_to_anchor(pcs)
    manip_b.inverse_kinematic(anchors_b)
    manip_b.vis_model(save=path+f"manip_b_{i:02d}.png")

# Create a Video out of the saved images
import cv2
import os

image_folder = path
video_name_a = os.path.join(path, 'manip_a_video.mp4')
video_name_b = os.path.join(path, 'manip_b_video.mp4')
images_a = [img for img in os.listdir(image_folder) if img.startswith("manip_a_")]
images_b = [img for img in os.listdir(image_folder) if img.startswith("manip_b_")]
images_a.sort()
images_b.sort()
frame = cv2.imread(os.path.join(image_folder, images_a[0]))
height, width, layers = frame.shape 
video_a = cv2.VideoWriter(video_name_a, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
video_b = cv2.VideoWriter(video_name_b, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
for image in images_a:
    video_a.write(cv2.imread(os.path.join(image_folder, image)))
for image in images_b:
    video_b.write(cv2.imread(os.path.join(image_folder, image)))
# cv2.destroyAllWindows()
video_a.release()
video_b.release()   


