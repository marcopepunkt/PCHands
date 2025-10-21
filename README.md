<h1 align="center">
    :hand::love_you_gesture: PCHands :vulcan_salute::v:
</h1>

<h3 align="center">
  PCHands: <br>PCA-based Hand Pose Retargeting <br>on Manipulators with <i>N</i>-DoF
</h3>

<div align="center">
  2025 IEEE-RAS 24th International Conference on Humanoid Robots
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2508.07945"><b>Paper</b></a> |
  <a href="https://www.youtube.com/watch?v=1rJksShwlbU"><b>Video</b></a> | 
  <a href="https://hsp-iit.github.io/PCHands/"><b>Page</b></a> | 
  <a href="https://huggingface.co/datasets/HSP-IIT/PCHands"><b>Dataset</b></a>
  <br>
  <video autoplay loop muted height="400" width="400" src="https://github.com/user-attachments/assets/53ebbece-246d-4359-aed9-92e007885296"></video>

</div>

## :scroll: Table of Contents

- [Updates](#new-Updates)
- [Installation](#gear-Installation)
- [Manipulator](#hammer_and_wrench-Manipulator)
- [Usage](#memo-Usage)
- [Hotfix](#wrench-hotfix)
- [Acknowledgement](#weight_lifting-Acknowledgement)
- [License](#balance_scale-License)
- [Citation](#bookmark_tabs-Citing-this-paper)


## :new: Updates

- [2025-Oct-21]: Official 🎆code🎆 released on [GitHub](https://github.com/hsp-iit/PCHands/).
- [2025-Oct-16]: 🎉Model Checkpoint🎉 and 🎉task demonstrations🎉 are released on [Hugging Face](https://huggingface.co/datasets/HSP-IIT/PCHands) and on [IIT Dataverse](https://dataverse.iit.it/dataset.xhtml?persistentId=doi:10.48557/3GWSE7).


## :gear: Installation

1. [PCHands](https://github.com/hsp-iit/PCHands)
   ```commandline
   git clone https://github.com/hsp-iit/PCHands.git --recursive
   ```
2. Download [MANO-v1_2](https://mano.is.tue.mpg.de) and place `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` in `assets/mano_hand/models`

3. Download [ARC](https://github.com/ZhengdiYu/Arbitrary-Hands-3D-Reconstruction)-[checkpoint](https://drive.google.com/file/d/1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9/view?usp=share_link) and place `wild.pkl` in `arhand/checkpoints`

4. Conda Env:
   
   ```commandline
   conda create -n pch python=3.11
   conda activate pch
   ```
   
5. [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
   
   ```commandline
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
   conda install -c conda-forge cudatoolkit-dev
   FORCE_CUDA=1 pip install https://github.com/facebookresearch/pytorch3d/archive/refs/tags/V0.7.8.tar.gz
   ```
 
6. Others:
 
   ```commandline
   pip install -r requirements.txt
   ```


## :hammer_and_wrench: Manipulator

1. [robotiq_2f85](https://github.com/ros-industrial-attic/robotiq)
2. [franka_gripper](https://github.com/frankaemika/franka_ros)
3. [widowx_gripper](https://github.com/RobotnikAutomation/widowx_arm)
4. [xarm_gripper](https://github.com/xArm-Developer/xarm_ros)
5. [wsg50_gripper](https://github.com/ipa320/ipa325_wsg50)
6. [rethink_egripper](https://github.com/RethinkRobotics/sawyer_robot)
7. [fetch_gripper](https://github.com/ZebraDevs/fetch_ros)
8. [google_gripper](https://github.com/google-deepmind/mujoco_menagerie)
9. [kinova_2f](https://github.com/Kinovarobotics/kinova-ros)
10. [kinova_3f_right](https://github.com/Kinovarobotics/kinova-ros)
11. [ergocub_hand_right](https://github.com/icub-tech-iit/ergocub-software)
12. [schunk_hand_right](https://github.com/fzi-forschungszentrum-informatik/schunk_svh_driver)
13. [allegro_hand_right](https://github.com/simlabrobotics/allegro_hand_ros)
14. [shadow_hand_left](https://github.com/AndrejOrsula/shadow_hand_ign)
15. [armar_hand_right](https://git.h2t.iar.kit.edu/sw/armarx-integration/robots/armar6/models)
16. [Leap_hand_right](https://github.com/leap-hand/LEAP_Hand_Sim)
17. [mano_hand_right](https://mano.is.tue.mpg.de/)

Examine manipulator:
```python
import numpy as np
from adf.manipulator import Manipulator

# list all available manipulator names
print(Manipulator.names)
# create manipulator instance
manip = Manipulator(Manipulator.names[i], verbose=True)
# random values in joint-space
q = np.random.rand(manip.dof)
# map into manipulator's joint limit
q = manip.denormalize_joint(q)
# forward kinematic with joint values
manip.forward_kinematic(q)
# visualization
manip.vis_model()
```


## :memo: Usage

### Dimensionality Reduction

```commandline
cd adf
python calib_eef.py
```

Expected output:

- calib_eef.yaml
- pca.pkl
- pca.pth
- stats.npy

### Collect Demo 

```commandline
cd rl_scripts
# collect_demo: <task_name> <manip_src>
python collect_demo.py table_door ergocub_hand_right
```

Registered key:

- `q`: quit
- `a`: reset env
- `z`: toggle finger lock
- `x`: toggle recording
- `c`: save episode

Typical Procedure:

- `x`record
- move-palm-finger-env
- `x`stop-record
- `c`save
- `a`reset-env
- repeat ...
- `q` quit

Expected output:

- teleop_collection/<task_name>/*.pkl

### Replay Demo
```commandline
# replay_demo: <task_name>
python replay_demo.py table_door
```
Expected output:

- Sapien visualization of the collected demos

### Bake Demo
```commandline
# bake_demo: modify data to bake in bake_demo.py 
python bake_demo.py
```

Expected output:

- rl_scripts/demo/<manip_src>/<task_name>_<n_pc>.pkl
- rl_sim/env/norm/<manip_src>/<task_name>_<n_pc>.pkl

### Train RL
```commandline
# training: modify training params via argv
SAPIEN_RENDER='0' python train_rl.py
```

Expected output:

- < rl-algo > _ < task > _ < manip-tgt > _ < n-pc > _ < datetime >
  - policy_xxxx.pickle: checkpoint
  - cfg.json          : experiment setting
  - log.csv           : values-log


## :wrench: Hotfix

1. If you encounter errors in loading of URDF, try modifying `$CONDA_PREFIX/lib/python3.11/site-packages/xacro/__init__.py`:
   ```diff
   - 103         ...yaml.load...
   + 103         ...yaml.safe_load...
   ```

2. If you encounter errors in loading mano-hand, try modifying `$CONDA_PREFIX/lib/python3.11/site-packages/chumpy/ch.py`:
   ```diff
   - 1203        ...inspect.getargspec...
   + 1203        ...inspect.getfullargspec...
   - 1246        ...inspect.getargspec...
   + 1246        ...inspect.getfullargspec...
   ```

3. If you encounter Sapien errors while training with CPUs, try modifying `$CONDA_PREFIX/lib/python3.11/site-packages/sapien/wrapper/urdf_loader.py`:
   ```diff
   + 68          self.no_visual = False
   + 294         if self.no_visual:
   + 295             break
   + 801         ..., no_visual = False
   + 803         self.no_visual = no_visual
   ```

## :weight_lifting: Acknowledgement

This research was supported by:
- PNRR MUR project PE0000013-FAIR
- National Institute for Insurance against Accidents at Work (INAIL) project ergoCub-2.0
- Brain and Machines Flagship Programme of the Italian Institute of Technology


## :balance_scale: License

The code is released under the _BSD 3-Clause_ License. See [LICENSE](LICENSE) for further details.
This code adapts multiple sources in different sections, see: 
 - [mjrl](https://github.com/aravindr93/mjrl/) - [Apache 2.0 License](rl_scripts/LICENSE_mjrl.txt)
 - [dex-hand-teleop](https://github.com/yzqin/dex-hand-teleop/) - [MIT License](rl_sim/LICENSE_dex_hand_teleop.txt)
 - [LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API/) - [CC BY-NC 4.0 License](inference/leap_hand/LICENSE_leap_hand_api.txt) 


## :bookmark_tabs: Citing this paper

```bibtex
@misc{
title={PCHands: PCA-based Hand Pose Synergy Representation on Manipulators with N-DoF}, 
author={En Yen Puang and Federico Ceola and Giulia Pasquale and Lorenzo Natale},
year={2025},
eprint={2508.07945},
archivePrefix={arXiv},
primaryClass={cs.RO},
url={https://arxiv.org/abs/2508.07945}, 
}
```


## :bearded_person: Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://github.com/hsp-iit/organization/blob/master/team/enyen_puang.jpg" width="40">](https://github.com/enyen) | [@enyen](https://github.com/enyen) |
