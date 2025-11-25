import numpy as np
from adf.manipulator import Manipulator

# list all available manipulator names
print(Manipulator.names)
# create manipulator instance
i = 10

manip = Manipulator(Manipulator.names[0], verbose=False)
# # random values in joint-space
q = np.zeros(manip.dof)
# print(f"DOF: {manip.dof}")
# # q = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# # # # map into manipulator's joint limit
q = manip.denormalize_joint(q)
# # # # forward kinematic with joint values
#print("Joint values:", q)
manip.forward_kinematic(q)
# # # visualization


manip.vis_model()