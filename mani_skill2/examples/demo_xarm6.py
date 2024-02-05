import sapien.core as sapien
import mplib
import numpy as np
from sapien.utils.viewer import Viewer

class PlanningDemo():
    def __init__(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 240.0)
        self.scene.add_ground(-0.8)
        physical_material = self.scene.create_physical_material(1, 1, 0.0)
        self.scene.default_physical_material = physical_material


        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
        self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)

        # Robot
        # Load URDF
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load("mani_skill2/assets/descriptions/xarm_description/xarm6_with_gripper.urdf")
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        # Set initial joint positions
        init_qpos = [-0.139, 0.417, -1.811, -0.035, 1.442, -0.176, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
        self.robot.set_qpos(init_qpos)

        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)

        # table top
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.4, 0.4, 0.025])
        builder.add_box_visual(half_size=[0.4, 0.4, 0.025])
        self.table = builder.build_kinematic(name='table')
        self.table.set_pose(sapien.Pose([0.56, 0, - 0.025]))

        # boxes
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.06])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.06], color=[1, 0, 0])
        self.red_cube = builder.build(name='red_cube')
        self.red_cube.set_pose(sapien.Pose([0.4, 0.3, 0.06]))

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.04])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.04], color=[0, 1, 0])
        self.green_cube = builder.build(name='green_cube')
        self.green_cube.set_pose(sapien.Pose([0.2, -0.3, 0.04]))

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.07])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.07], color=[0, 0, 1])
        self.blue_cube = builder.build(name='blue_cube')
        self.blue_cube.set_pose(sapien.Pose([0.6, 0.1, 0.07]))

        self.setup_planner()
    
    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        print(joint_names)
        self.planner = mplib.Planner(
            urdf="mani_skill2/assets/descriptions/xarm_description/xarm6_with_gripper.urdf",
            srdf="mani_skill2/assets/descriptions/xarm_description/xarm6_with_gripper.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="xarm_gripper_base_link",
            joint_vel_limits=np.ones(6),
            joint_acc_limits=np.ones(6))

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        for i in range(n_step):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(6):
                self.active_joints[j].set_drive_target(result['position'][i][j])
                self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def open_gripper(self):
        for joint in self.active_joints[-6:]:
            # if joint.get_name() in ["right_finger_joint", "left_finger_joint"]:
                # print(joint.get_name())
                # joint.set_drive_target(-0.4)
            joint.set_drive_target(0.8)
        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-6:]:
            # if joint.get_name() in ["right_finger_joint", "left_finger_joint"]:
                # print(joint.get_name())
                # joint.set_drive_target(0)
            joint.set_drive_target(0)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def move_to_pose_with_RRT(self, pose):
        result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1/250)
        if result['status'] != "Success":
            print(result['status'])
            return -1
        self.follow_path(result)
        return 0

    def move_to_pose_with_screw(self, pose):
        qpos = self.robot.get_qpos()
        print(qpos)
        result = self.planner.plan_screw(pose, self.robot.get_qpos(), time_step=1/250)
        print(result)
        if result['status'] != "Success":
            result = self.planner.plan_qpos_to_pose(pose, self.robot.get_qpos(), time_step=1/250)
            if result['status'] != "Success":
                print(result['status'])
                return -1 
        self.follow_path(result)
        print(qpos)
        return 0
    
    def move_to_pose(self, pose, with_screw):
        if with_screw:
            return self.move_to_pose_with_screw(pose)
        else:
            return self.move_to_pose_with_RRT(pose)

    def demo(self, with_screw = True):
        poses = [[0.4, 0.3, 0.12, 0, 1, 0, 0],
                [0.2, -0.3, 0.08, 0, 1, 0, 0],
                [0.6, 0.1, 0.14, 0, 1, 0, 0]
                ]
        for i in range(3):
            pose = poses[i]
            # print(pose[2])
            pose[2] += 0.2
            # print(pose)
            self.move_to_pose(pose, with_screw)
            self.open_gripper()
            pose[2] -= 0.12
            self.move_to_pose(pose, with_screw)
            self.close_gripper()
            pose[2] += 0.12
            self.move_to_pose(pose, with_screw)
            pose[0] += 0.1
            self.move_to_pose(pose, with_screw)
            pose[2] -= 0.12
            self.move_to_pose(pose, with_screw)
            self.open_gripper()
            pose[2] += 0.12
            self.move_to_pose(pose, with_screw)

if __name__ == '__main__':
    demo = PlanningDemo()
    demo.demo()
