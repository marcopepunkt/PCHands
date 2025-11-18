Purpose
-------
These instructions surface the repository-specific knowledge an AI coding assistant needs to be productive in PCHands (PCA-based hand pose retargeting). Keep guidance short, concrete, and tied to files or commands that appear in the repo.

Quick summary
-------------
- Big picture: dataset + models (adf) produce PCA/Anchor representations; inference components (arhand, inference/) consume those to retarget hand poses onto many manipulators; RL scripts in `rl_scripts/` collect, replay, bake and train policies using normed data.DexH2R: Task-oriented Dexterous Manipulation from Human to RobotsV
- Primary entry points: `README.md` (setup & commands), `adf/` (dimensionality reduction & manipulator model), `inference/` (scripts for manipulation and demos), and `rl_scripts/` (teleop, dataset collection, training).

Files & patterns to inspect
--------------------------
- `adf/manipulator.py`: core Manipulator wrapper. Important concepts: `AnchorAE`/PCA usage (`pca.pth`, `stats.npy`), `denormalize_joint`, `pc_to_anchor`/`anchor_to_pc`, and Klampt-based IK/visualization. Example: `Manipulator.names` lists supported robots.
- `adf/calib_eef.py` and `adf/dim_reduction.py`: produce `pca.pth`, `pca.pkl` and `stats.npy` used at runtime. Follow their CLI usage in `README.md`.DexH2R: Task-oriented Dexterous Manipulation from Human to Robots
- `arhand/`: hand reconstruction & MANO integration; expects MANO assets in `assets/mano_hand/models` and ARC checkpoints in `arhand/checkpoints` per README.
- `rl_scripts/`: teleoperation, collection, replay and training flows. Key scripts: `collect_demo.py`, `replay_demo.py`, `bake_demo.py`, `train_rl.py`.

Dev workflows & commands
-----------------------
- Environment: use Conda as in `README.md`. Python 3.11 recommended, named env `pch`.
- Install: pip-install Pytorch3D (see README for exact wheel and CUDA notes) then `pip install -r requirements.txt`.
- Dimensionality reduction (generates PCA & stats):
  - cd adf && python calib_eef.py
  - Expected outputs: `calib_eef.yaml`, `pca.pth`, `pca.pkl`, `stats.npy`
- Collect / replay / bake demos (teleop flow): `rl_scripts/` contains CLI usage; follow the README examples.

Repository conventions and patterns
---------------------------------
- PCA/Anchor workflow: anchors are 22 3D points flattened into 66-D vectors. Normalization uses `stats.npy` (keys: `means`, `stds`). Models expect PC representation length `pca.num_components`.
- Manipulators: `adf/manipulator.py` uses Klampt WorldModel and represents joints both as driver-space (dof) and full joint configuration (`idx_joint`). Use `denormalize_joint` to map [0,1]^dof to driver limits.
- URDF handling: the repo uses `xacro` to preprocess URDFs stored in `assets/<manip>/model_klampt.urdf`. Be aware of the README hotfixes if xacro or chumpy errors appear.
- Hard-coded anchors: manipulator links named `A_00`..`A_21` and color palette defined in `manipulator.py`.

Integration & external dependencies
----------------------------------
- Requires heavy ML and robotics libs: PyTorch, Pytorch3D, SAPIEN (for simulation), Klampt (visualization & IK), and MANO models. See `README.md` for exact installation notes and hotfixes.
- Checkpoints & assets: MANO files must be manually placed in `assets/mano_hand/models`. ARC `wild.pkl` should be in `arhand/checkpoints`.

What to look for when editing
-----------------------------
- Preserve the PCA shape: changing anchor order or normalization breaks precomputed `pca.pth` and `stats.npy` consumers.
- If adding a manipulator, update `Manipulator.names`, provide `model_klampt.urdf` and ensure anchor links `A_00..A_21` exist.
- Maintain driver vs full-joint semantics in `manipulator.py` when modifying kinematics or IK.

Examples (copy-paste patterns)
-----------------------------
- Instantiate manipulator and visualize:
  from adf.manipulator import Manipulator
  m = Manipulator('ergocub_hand_right', verbose=True)
  m.forward_kinematic(m.denormalize_joint(np.random.rand(m.dof)))
  m.vis_model()
- Generate PCA artifacts:
  cd adf && python calib_eef.py

If you change this file
-----------------------
- Keep guidance focused and short. Add any new build steps, required assets, or non-standard fixes you discover.

Questions for the maintainer
---------------------------
- Are there additional non-documented environment tweaks (CUDA, SAPIEN config) that should be added to these instructions?
- Is there an automated test or CI job to validate runtime scripts (dim reduction, demo replay) we should reference?
