"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill2 tasks can minimally be defined by how the environment resets, what agents/objects are 
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self.reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill2. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""
import os
from collections import OrderedDict
from typing import Any, Dict, Type

import numpy as np
import sapien.core as sapien
import mplib

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.configs.panda.variants import PandaPourConfig
from mani_skill2.agents.robots.panda import Panda

from mani_skill2.envs.mpm.base_env import MPMBaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.mpm import perlin
from mani_skill2.utils.sapien_utils import (  # import various useful utilities for working with sapien
    get_entity_by_name, 
    vectorize_pose,
    look_at,
)
from mani_skill2.envs.mpm.utils import actor2meshes


@register_env("myenv_panda", max_episode_steps=250)
class CustomEnv(MPMBaseEnv):
    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot
    def __init__(
        self,
        *args,
        sim_freq=500,
        mpm_freq=2000,
        **kwargs,
    ):
        super().__init__(
            *args,
            sim_freq=sim_freq,
            mpm_freq=mpm_freq,
            **kwargs,
        )

    """
    One time configuration code
    """

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        E = 1e4
        nu = 0.3
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

        # 0 for von-mises, 1 for drucker-prager
        type = 1

        # von-mises
        ys = 1e4

        # drucker-prager
        friction_angle = 0.6
        cohesion = 0.05

        height_map = 0.06 + perlin.added_perlin(
            [0.03, 0.02, 0.02],
            [1, 2, 4],
            phases=[(0, 0), (0, 0), (0, 0)],
            shape=(30, 30),
            random_state=self._episode_rng,
        )

        count = self.model_builder.add_mpm_from_height_map(
            pos=(0.0, 0.0, 0.08),
            vel=(0.0, 0.0, 0.0),
            dx=0.005,
            height_map=height_map,
            density=3.0e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=type,
            jitter=True,
            color=(1, 1, 0.5),
            random_state=self._episode_rng,
        )

        self.model_builder.init_model_state(self.mpm_model, self.mpm_states)
        self.mpm_model.struct.static_ke = 100.0
        self.mpm_model.struct.static_kd = 0.0
        self.mpm_model.struct.static_mu = 1.0
        self.mpm_model.struct.static_ka = 0.0

        self.mpm_model.struct.body_ke = 100.0
        self.mpm_model.struct.body_kd = 0.0
        self.mpm_model.struct.body_mu = 1.0
        self.mpm_model.struct.body_ka = 0.0

        self.mpm_model.adaptive_grid = True

        self.mpm_model.grid_contact = True
        self.mpm_model.particle_contact = True
        self.mpm_model.struct.body_sticky = 1
        self.mpm_model.struct.ground_sticky = 1

        self.mpm_model.struct.particle_radius = 0.0025

    def _register_sensors(self):
        # To customize the sensors that capture images/pointclouds for the environment observations,
        # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
        pose = look_at(
            eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )  # look_at is a utility to get the pose of a camera that looks at a target
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_render_cameras(self):
        # this is just like _register_sensors, but for adding cameras used for rendering when you call env.render()
        pose = look_at(eye=[0.4, 0.4, 0.8], target=[0.0, 0.0, 0.4])

        return [CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)]

    # def _setup_viewer(self):
    #     # you add code after calling super()._setup_viewer() to configure how the SAPIEN viewer (a GUI) looks
    #     super()._setup_viewer()
    #     self._viewer.set_camera_xyz(0.8, 0, 1.0)
    #     self._viewer.set_camera_rpy(0, -0.5, 3.14)

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task.
    """

    def _initialize_agent(self):
        # here you initialize the agent/robot. This usually involves setting the joint position of the robot. We provide
        # some default code below for panda and xmate3 set the robot to a "rest" position with a little bit of noise for randomization

        qpos = np.array([-0.207, 0.203, 0.080, -2.321, -0.111, 2.665, 0.748, 0.04, 0.04])
        # qpos[:-1] += self._episode_rng.normal(0, 0.02, len(qpos) - 1)
        # qpos[-1] += self._episode_rng.normal(0, 0.2, 1)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.56, 0, 0]))
        self.home_ee = np.array([0.0, 0.0, self.target_height])


    def _initialize_actors(self):
        super()._initialize_actors()
        self.target_height = 0.2
        self.target_num = self._episode_rng.choice(range(250, 1150), 1)[0]
        self.mpm_model.struct.n_particles = len(self.model_builder.mpm_particle_q)
        
        # vs = self.spoon.get_visual_bodies()
        # assert len(vs) == 1
        # v = vs[0]
        # vertices = np.concatenate([s.mesh.vertices for s in v.get_render_shapes()], 0)
        # self._target_height = vertices[:, 2].max() * v.scale[2]
        # self._target_radius = v.scale[0]

    def _initialize_articulations(self):
        pass

    def _initialize_task(self):
        # we highly recommend to generate some kind of "goal" information to then later include in observations
        # goal can be parameterized as a state (e.g. target pose of a object)
        pass

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task.
    """

    def _load_agent(self):
        # this code loads the agent into the current scene. You can usually ignore this function by deleting it or calling the inherited
        # BaseEnv._load_agent function
        # super()._load_agent()
        self.agent = Panda(
            self._scene,
            self._control_freq,
            control_mode=self._control_mode,
            config=self._agent_cfg,
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), "panda_hand_tcp"
        )
        self.lfinger = get_entity_by_name(
            self.agent.robot.get_links(), "panda_leftfinger"
        )
        self.rfinger = get_entity_by_name(
            self.agent.robot.get_links(), "panda_rightfinger"
        )

    def _load_actors(self):
        # here you add various objects (called actors). If your task was to push a ball, you may add a dynamic sphere object on the ground
        super()._load_actors()
        bowl_dir = os.path.join(
            PACKAGE_ASSET_DIR, "descriptions/feeding/meshes/bowl.STL"
            )
        bowl_collision_dir = os.path.join(
            PACKAGE_ASSET_DIR, "descriptions/feeding/meshes/bowl.STL.convex.stl"
            )
        pose = sapien.Pose([0, 0, 0.07])
        b = self._scene.create_actor_builder()
        b.add_visual_from_file(bowl_dir, pose, scale=[0.002] * 3)
        b.add_collision_from_file(bowl_collision_dir, pose, scale=[0.002] * 3, density=300)
        self.source_container = b.build("bowl")

        # spoon_dir = os.path.join(
        #     PACKAGE_ASSET_DIR, "descriptions/feeding/meshes/spoon.STL"
        #     )
        # spoon_collision_dir = os.path.join(
        #     PACKAGE_ASSET_DIR, "descriptions/feeding/meshes/spoon.STL.convex.stl"
        #     )
        # pose = sapien.Pose([0.0025, -0.1105, 0.2],[-0.007,-0.699,0.007,0.715])
        # b = self._scene.create_actor_builder()
        # b.add_visual_from_file(spoon_dir, pose, scale=[0.001] * 3)
        # b.add_collision_from_file(spoon_collision_dir, pose, scale=[0.001] * 3, density=300)
        # self.spoon = b.build("spoon")


    def _get_coupling_actors(
        self,
    ):
        return [
            (self.source_container, "visual"),
            # (self.spoon, "visual"),
        ]

    def _configure_agent(self):
        self._agent_cfg = PandaPourConfig()

    # def _setup_sensors(self):
    #     # default code here will setup all sensors. You can add additional code to change the sensors e.g.
    #     # if you want to randomize camera positions
    #     return super()._setup_sensors()

    # def _setup_lighting(self):
    #     # default code here will setup all lighting. You can add additional code to change the lighting e.g.
    #     # if you want to randomize lighting in the scene
    #     return super()._setup_lighting()

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def _get_obs_extra(self) -> OrderedDict:
        # should return an OrderedDict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key
        return OrderedDict(
            tcp_pose=vectorize_pose(self.grasp_site.get_pose()),
            target=np.array([self.target_num]),
        )

    def evaluate(self, **kwargs):
        # should return a dictionary containing "success": bool indicating if the environment is in success state or not. The value here is also what the sparse reward is
        # for the task. You may also include additional keys which will populate the info object returned by self.step
        particles_x = self.get_mpm_state()["x"]
        particles_v = self.get_mpm_state()["v"]
        lift_num = len(np.where(particles_x[:, 2] > self.target_height)[0])
        spill_num = self.mpm_model.struct.n_particles - len(
            np.where(
                (particles_x[:, 0] > -0.12)
                & (particles_x[:, 0] < 0.12)
                & (particles_x[:, 1] > -0.12)
                & (particles_x[:, 1] < 0.12)
            )[0]
        )
        return dict(
            success=(
                lift_num > self.target_num - 100
                and lift_num < self.target_num + 150
                and spill_num < 20
                and len(np.where((particles_v < 0.05) & (particles_v > -0.05))[0])
                / (self.mpm_model.struct.n_particles * 3)
                > 0.99
            )
        )
        

    def compute_dense_reward(self, reward_info=False, **kwargs):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        if self.evaluate()["success"]:
            if reward_info:
                return {"reward": 6.0}
            return 6.0
        particles_x = self.get_mpm_state()["x"]

        stage = 0

        # spill reward
        spill_num = self.n_particles - len(
            np.where(
                (particles_x[:, 0] > -0.12)
                & (particles_x[:, 0] < 0.12)
                & (particles_x[:, 1] > -0.12)
                & (particles_x[:, 1] < 0.12)
            )[0]
        )
        spill_reward = -spill_num / 100

        reward = spill_reward
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        # this should be equal to compute_dense_reward / max possible reward
        # max_reward = 1.0
        # return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

        return self.compute_dense_reward(**kwargs) / 6.0

    def render(self, draw_box=False, draw_target=False):
        if draw_target:
            bbox = self.target_box
            box = self._add_draw_box(bbox)

        img = super().render(draw_box)
        if draw_target:
            self._remove_draw_box(box)
        return img


    
    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf="mani_skill2/assets/descriptions/panda_v2.urdf",
            srdf="mani_skill2/assets/descriptions/panda_v2.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="drive_joint",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        for i in range(n_step):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result['position'][i][j])
                self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
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
        for joint in self.active_joints[-2:]:
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

    def move_to_pose_with_RRTConnect(self, pose):
        result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1/250)
        if result['status'] != "Success":
            print(result['status'])
            return -1
        self.follow_path(result)
        return 0

    def move_to_pose_with_screw(self, pose):
        result = self.planner.plan_screw(pose, self.robot.get_qpos(), time_step=1/250)
        if result['status'] != "Success":
            result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1/250)
            if result['status'] != "Success":
                print(result['status'])
                return -1 
        self.follow_path(result)
        return 0
    
    def move_to_pose(self, pose, with_screw):
        if with_screw:
            return self.move_to_pose_with_screw(pose)
        else:
            return self.move_to_pose_with_RRT(pose)

    def demo(self, with_screw = True):
        poses = [[0.4, 0.3, 0.12, 0, 1, 0, 0],
                [0.2, -0.3, 0.08, 0, 1, 0, 0],
                [0.6, 0.1, 0.14, 0, 1, 0, 0]]
        for i in range(3):
            pose = poses[i]
            pose[2] += 0.2
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

if __name__ == "__main__":
    env = CustomEnv(reward_mode="dense")
    env.reset()
    env.agent.set_control_mode("pd_ee_delta_pose")

    a = env.get_state()
    env.set_state(a)

    for i in range(100):
        env.step(None)
        env.render()
        env.demo()