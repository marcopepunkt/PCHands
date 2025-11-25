"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import numbers
import sys
import xacro
import tempfile
import numpy as np
from os import path
from klampt.model import ik
from klampt.math import se3
from klampt import WorldModel, vis, GeometricPrimitive
sys.path.append(path.dirname(__file__))
from cvae import AnchorAE
from urdf_parser_py.urdf import URDF
import yaml


class Manipulator:
    names = ['robotiq_2f85', 'franka_gripper', 'widowx_gripper',
             'xarm_gripper', 'wsg50_gripper', 'rethink_egripper', 'fetch_gripper',
             'armar_hand_right', 'google_gripper', 'kinova_2f',  'kinova_3f_right',
             'ergocub_hand_right', 'schunk_hand_right', 'allegro_hand_right', 'shadow_hand_right', 'leap_hand_right', 'mimic_hand_right', 'orca_v1']
    # names = ['armar_hand_right','ergocub_hand_right', 'schunk_hand_right','shadow_hand_right', 'mimic_hand_right', 'orca_v1', 'leap_hand_right', 'kinova_3f_right']
    def __init__(self, model_name, fixed_base=True, verbose=True, headless=False, use_scheme=False):
        """
        init manipulator class
        :param model_name: str name of the available model
        :param fixed_base: bool fixed or floating base
        :param verbose: bool verbose model details
        """
        self.verbose = verbose

        if "mimic" in model_name:
            model_name = "mimic_hand_right"
        if "orca" in model_name:
            model_name = "orca_v1"

        assert model_name in Manipulator.names, "{} not found in PCHands, please fix the name or add the model to PCHands".format(model_name)
        self.world = WorldModel()
        self.name = model_name

        # load robot
        # dir_urdf = path.join(path.dirname(__file__), '../assets', model_name, 'model_klampt.urdf')

        dir_urdf = path.join(path.dirname(__file__), '../assets', model_name)
        urdf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', dir=dir_urdf)
        urdf_ = xacro.process_file(path.join(dir_urdf, 'model_klampt.urdf')).toprettyxml(indent='  ')
        if fixed_base:
            urdf_ = urdf_.replace('freeze_root_link="0"', 'freeze_root_link="1"')
        else:
            urdf_ = urdf_.replace('freeze_root_link="1"', 'freeze_root_link="0"')
        if headless:
            urdf_ = urdf_.replace('visual="1"', 'visual="0"')

        urdf.write(urdf_)
        urdf.flush()

        # SINCE YOU CANNOT GET THE JOINT NAME FROM DRIVER NAME, I NEED TO CREATE A MAP
        urdf_class = URDF.from_xml_file(urdf.name)
        self.link_name_to_joint_name = {j.child: j.name for j in urdf_class.joints}  # map child link -> joint name

        self.world.loadRobot(urdf.name)
        self.robot = self.world.robot(0)
        urdf.close()

        self.base_path = dir_urdf

        self.pca : AnchorAE  = None
        # load pose
        try:
            f_stats = path.join(path.dirname(__file__), 'stats.npy')
            self.stats = np.load(f_stats, allow_pickle=True).item()[self.name]
            self.pca = AnchorAE(load_model='pca.pth')
        except:
            print('PCA data not available!')
            self.stats = None

        # Load scheme if available
        scheme_path = path.join(path.dirname(__file__), '../assets', model_name, 'scheme.yaml')
        if path.exists(scheme_path) and use_scheme:
            with open(scheme_path, 'r') as f:
                self.scheme = yaml.safe_load(f)
            self.joint_name_to_index = {}
            self.dof = len(self.scheme.get('gc_tendons', []))
            
        else:
            self.scheme = None
            self.dof = self.robot.numDrivers()
        
        # for i in range(self.robot.numDrivers()):
        #     d = self.robot.driver(i)
        #     print(f"{i}: {d.getName()} ({d.getType()}) = {d.getValue()}")
            # d = self.robot.driver(i)
            # name = self.robot.link(d.getAffectedLink()).getName()
            # value = d.getValue()         # current angle or displacement
            # type_ = d.getType()          # e.g. "rotation", "translation"
            # print(f"{i}: {name} -> {type_} = {value}")

        # properties
        self.idx_joint = []
        for i in range(self.robot.numLinks()):
            if self.robot.getJointType(i) == 'normal':
                self.idx_joint.append(i)
        self.cfg_init = self.robot.getConfig().copy()
        for i in range(len(self.idx_joint)):
            self.cfg_init[self.idx_joint[i]] = np.clip(self.cfg_init[self.idx_joint[i]],
                                                       self.robot.getJointLimits()[0][self.idx_joint[i]],
                                                       self.robot.getJointLimits()[1][self.idx_joint[i]])

        # joint name for ik
        self.fixed_base = fixed_base
        self.ik_dof = []
        if not fixed_base:
            for i in range(6):
                self.ik_dof.append(self.robot.link(i).getName())
        # THIS IS NOT PRETTY AND DOES NOT COVER THE ROLLING JOINT CASE, BUT IT WILL DO FOR NOW
        # TODO: IMPROVE THIS
        for i in range(self.robot.numDrivers()):
            self.ik_dof.append(self.robot.driver(i).getName())

        # verbose
        if verbose:
            print('-------------------------------------------')
            print('LINK: idx | type | name | l_limit | u_limit')
            for i in range(self.robot.numLinks()):
                print(i, self.robot.getJointType(i), self.robot.link(i).name,
                      self.robot.getJointLimits()[0][i], self.robot.getJointLimits()[1][i])
            print('---------------------------------------------')
            print('DRIVER: idx | type | name | l_limit | u_limit')
            for i in range(self.dof):
                print(i, self.robot.driver(i).getType(), self.robot.driver(i).getName(),
                      self.robot.driver(i).getLimits()[0], self.robot.driver(i).getLimits()[1])
            print('--------------')
            print('IK: idx | name')
            for i in range(len(self.ik_dof)):
                print(i, self.ik_dof[i])

    def reset(self):
        """
        reset all joint configuration to init state
        """
        self.robot.setConfig(self.cfg_init)

    def forward_kinematic(self, q, use_scheme=False):
        """
        set driver joint values
        :param q: list of joint values
        :return: None

        Here we have the joint angles from the dataset and want to apply them to the robot
        The dataset has fewer DOFs than the number of drivers because of the tendon mapping
        """

        if use_scheme and self.scheme is not None:
            gc_tendons = self.scheme.get('gc_tendons', {}) # Dict of Tendons
            assert len(q) == len(gc_tendons), (
                'q is not in the correct length. Expect {}, given {}'.format(len(gc_tendons), len(q)))

            q_mapped = np.zeros(self.robot.numDrivers())
            for tendon_num, tendon_name in enumerate(gc_tendons.keys()):
                value = q[tendon_num]
                if not isinstance(value, numbers.Number):
                    raise ValueError(f"Joint value for tendon '{tendon_name}' is not a number: {value} (type={type(value)})")
                value = float(value)    
                for i in range(self.robot.numDrivers()):
                    driver_name = self.robot.driver(i).getName()
                    joint_name = self.link_name_to_joint_name[driver_name]
                    if joint_name == tendon_name:
                        q_mapped[i] = value

                    # check if this tendon has mappings
                    mapping = gc_tendons[tendon_name]
                    for mapped_joint, ratio in mapping.items():
                        if joint_name == mapped_joint:
                            q_mapped[i] = value 
            self.robot.setConfig(self.robot.configFromDrivers(q_mapped))
        else:
            assert len(q) == self.robot.numDrivers(), (
                'q is not in the correct length. Expect {}, given {}'.format(self.robot.numDrivers(), len(q)))
            q = [float(val) for val in q]
            self.robot.setConfig(self.robot.configFromDrivers(q))

    def get_links_transform(self, link_source, link_target):
        # source
        rt = self.robot.link(link_source).getTransform()
        root2source = np.eye(4)
        root2source[:3, :3] = np.array(rt[0]).reshape(3, 3)
        root2source[:3, 3] = np.array(rt[1])
        # target
        rt = self.robot.link(link_target).getTransform()
        root2target = np.eye(4)
        root2target[:3, :3] = np.array(rt[0]).reshape(3, 3)
        root2target[:3, 3] = np.array(rt[1])
        # source to target
        source2target = np.linalg.inv(root2source) @ root2target
        return source2target

    def get_anchor(self):
        """
        get anchors position
        :return: np.ndarray - anchor x,y,z position [22, 3]
        """
        anchors = np.array([self.robot.link('A_{:02d}'.format(i)).getTransform()[1] for i in range(22)])
        return anchors

    def set_joint(self, q):
        """
        set raw joint values
        :param q: list of joint values
        :return: None
        """
        assert len(q) == len(self.idx_joint), (
            'q is not in the correct length. Expect {}, given {}'.format(len(self.idx_joint), len(q)))
        cfg = self.robot.getConfig()
        for i in range(len(self.idx_joint)):
            cfg[self.idx_joint[i]] = float(q[i])
        self.robot.setConfig(cfg)

    def get_joint(self, all=True, use_scheme=False):
        """
        get joint values
        :param all: False for only driver values, True for all joints values
        :return: list of joint value
        """
        if all:
            cfg = self.robot.getConfig()
            q = [cfg[idx] for idx in self.idx_joint]
            if self.scheme is not None and use_scheme:
                gc_tendons = self.scheme.get('gc_tendons', {})
              
                # Now i want to map the joint_names to the tendon names
                # There are more joints than tendons. gc_tendons looks like:
                # gc_tendons:
                #   thumb_base2cmc : {}
                #   thumb_cmc2mcp : {}
                #   thumb_mcp2pp : {}
                #   thumb_pp2dp_actuated : {}

                #   index_base2mcp : {}
                #   index_mcp2pp : {}
                #   index_pp2mp : {index_mp2dp : 1}

                #   middle_base2mcp : {}
                #   middle_mcp2pp : {}
                #   middle_pp2mp : {middle_mp2dp : 1}

                #   ring_base2mcp : {}
                #   ring_mcp2pp : {}
                #   ring_pp2mp : {ring_mp2dp : 1}
                
                #   pinky_base2mcp : {}
                #   pinky_mcp2pp : {}
                #   pinky_pp2mp : {pinky_mp2dp : 1}

                # for index_pp2mp we take the average of index_pp2mp and index_mp2dp


                q_mapped = np.zeros(len(gc_tendons))
                for tendon_num, tendon_name in enumerate(gc_tendons.keys()):
                    values = []
                    for i, idx in enumerate(range(self.robot.numDrivers())):
                        driver_name = self.robot.driver(idx).getName()
                        joint_name = self.link_name_to_joint_name[driver_name]
                        if joint_name == tendon_name:
                            values.append(q[i])
                        # check if this tendon has mappings
                        mapping = gc_tendons[tendon_name]
                        for mapped_joint, ratio in mapping.items():
                            if joint_name == mapped_joint:
                                values.append(q[i] * ratio)
                        q_mapped[tendon_num] = np.mean(values)
                return q_mapped.tolist()
        else:
            q = [self.robot.driver(i).getValue() for i in range(self.robot.numDrivers())]
        return q

    def get_base(self):
        """
        get base/eef frame
        Returns:
            list: [tx, ty, tz, rz, ry, rz]
        """
        cfg = self.robot.getConfig()
        q = [cfg[0], cfg[1], cfg[2], cfg[5], cfg[4], cfg[3]]
        return q

    def set_base(self, q):
        """
        set base eef frame
        Args:
            q (list): [tx, ty, tz, rz, ry, rz]
        """
        cfg = self.robot.getConfig()
        cfg[0] = q[0]
        cfg[1] = q[1]
        cfg[2] = q[2]
        cfg[3] = q[5]
        cfg[4] = q[4]
        cfg[5] = q[3]
        self.robot.setConfig(cfg)

    def inverse_kinematic(self, pos_anchor, focus_tip=False):
        """
        set driver joint values from anchor position
        :param pos_anchor: list of anchor position [x,y,z]
        :param niter: int number of iteration
        :param tol: tolerance in meter
        :return: None
        """
        if focus_tip:
            # palms and tips
            objs = [ik.objective(self.robot.link('A_{:02d}'.format(i)), local=[0, 0, 0], world=pos_anchor[i])
                    for i in [3, 7, 11, 15, 19]]
            # others
            secondary = [ik.objective(self.robot.link('A_{:02d}'.format(i)), local=[0, 0, 0], world=pos_anchor[i])
                         for i in [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21]]
            # base constraints
            # if not self.fixed_base:
            #     secondary.append(ik.fixed_objective(self.robot.link('eef_link')))
            #     secondary.append(ik.fixed_rotation_objective(self.robot.link('eef_link')))
        else:
            objs = [ik.objective(self.robot.link('A_{:02d}'.format(i)), local=[0, 0, 0], world=pos_anchor[i])
                    for i in range(len(pos_anchor))]
            secondary = None

        ik.solve(objs, secondary=secondary, activeDofs=self.ik_dof, iters=20000, tol=1e-3)

    def denormalize_joint(self, qn):
        """
        convert normalized values to actual driver joint values
        :param qn: list of normalized(0-1) values
        :return: joint values respecting joint limit
        """
        assert len(qn) == self.dof
        q = []
        for i in range(self.dof):
            l_limit, u_limit = self.robot.driver(i).getLimits()

            assert isinstance(l_limit, (int, float)) and isinstance(u_limit, (int, float)), \
                f"Joint limits are not numeric for driver {i}: {l_limit}, {u_limit}"
            
            q.append(l_limit + (u_limit - l_limit) * qn[i])
        return q

    def pc_to_anchor(self, pc):
        """
        convert principal components to anchors position
        :param pc: np.ndarray[66] principal components
        :return: np.ndarray[22x3] anchors
        """
        pc = np.hstack((pc, np.zeros(self.pca.num_components - pc.shape[0])))
        anchor = self.pca.inverse_transform(pc[None], self.name).reshape(-1, 3)
        anchor = anchor * self.stats['stds'] + self.stats['means']
        return anchor

    def anchor_to_pc(self, anchor):
        """
        convert anchors position to principal components
        :param anchor: np.ndarray[22x3]
        :return: np.ndarray[66] principal components
        """
        anchor = (anchor - self.stats['means']) / self.stats['stds']
        pc = self.pca.transform(anchor.flatten()[None], self.name)[0]
        return pc

    def vis_model(self, cam_t=None, cam_r=[0, -1.57, -1.57], cam_dist=0.6, save=None):
        """
        visualize mesh and anchor
        :param cam_t: list of float [x, y, z] visual translation
        :param cam_r: list of float [r, p, y] visual rotation
        :param cam_dist: float visual distance
        :param save: image saving path
        :return: None
        """
        vis.setWindowTitle("Visualization")
        vis.setBackgroundColor(1, 1, 1)
        vis.add('world', se3.identity(), fancy=True, length=0.05, width=0.004, hide_label=True)
        vis.add('robot', self.robot, hide_label=False)
        vis.setColor('robot', 0.7, 0.6, 0.6)
        try:
            for i in range(22):
                name = "A_{:02d}".format(i)
                anc = GeometricPrimitive()
                anc.setSphere(self.robot.link(name).getTransform()[1], 0.005)
                vis.add(name, anc, hide_label=False)
                vis.setColor(name, *self.colors[i])
        except Exception as e:
            print(f"Error visualizing anchors: {e}")

        if self.verbose: 
            #visualize the coordinate frame of each link
            for i in range(self.robot.numLinks()):
                name = self.robot.link(i).getName()
                anc = GeometricPrimitive()
                anc.setSphere(self.robot.link(name).getTransform()[1], 0.005)
                name_vis = name + "_vis"
                vis.add(name_vis, anc, hide_label=False)

        vp = vis.getViewport()
        vp.camera.ori = ['z', 'x', 'y']
        if cam_t is not None:
            vp.camera.tgt = cam_t
        if cam_r is not None:
            vp.camera.rot = cam_r
        vp.camera.dist = cam_dist
        vp.w = 800
        vp.h = 800
        vis.setViewport(vp)
        if save is None:
            vis.dialog()
        elif save == 'return':
            vis.show()
            return vis.screenshot('numpy')
        else:
            vis.show()
            # vis.spin(0.1)
            vis.screenshot('Image').save(save)

    colors = np.array([[5.03830e-02, 2.98030e-02, 5.27975e-01],
                       [1.64070e-01, 2.01710e-02, 5.77478e-01],
                       [2.48032e-01, 1.44390e-02, 6.12868e-01],
                       [3.25150e-01, 6.91500e-03, 6.39512e-01],
                       [3.99411e-01, 8.59000e-04, 6.56133e-01],
                       [4.71457e-01, 5.67800e-03, 6.59897e-01],
                       [5.46157e-01, 3.89540e-02, 6.47010e-01],
                       [6.10667e-01, 9.02040e-02, 6.19951e-01],
                       [6.69845e-01, 1.42992e-01, 5.82154e-01],
                       [7.23444e-01, 1.96158e-01, 5.38981e-01],
                       [7.71958e-01, 2.49237e-01, 4.94813e-01],
                       [8.19651e-01, 3.06812e-01, 4.48306e-01],
                       [8.59750e-01, 3.60588e-01, 4.06917e-01],
                       [8.96131e-01, 4.15712e-01, 3.66407e-01],
                       [9.28329e-01, 4.72975e-01, 3.26067e-01],
                       [9.55470e-01, 5.33093e-01, 2.85490e-01],
                       [9.77856e-01, 6.02051e-01, 2.41387e-01],
                       [9.90681e-01, 6.69558e-01, 2.01642e-01],
                       [9.94495e-01, 7.40880e-01, 1.66335e-01],
                       [9.87621e-01, 8.15978e-01, 1.44363e-01],
                       [9.68443e-01, 8.94564e-01, 1.47014e-01],
                       [9.40015e-01, 9.75158e-01, 1.31326e-01]])
