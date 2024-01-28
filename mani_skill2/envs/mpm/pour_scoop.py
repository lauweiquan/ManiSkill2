import os
from collections import OrderedDict

import numpy as np
import sapien.core as sapien
import warp as wp
from transforms3d.euler import euler2quat

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.configs.xarm6.defaults import xarm6DefaultConfig
from mani_skill2.agents.robots.xarm6 import xarm6
from mani_skill2.envs.mpm import perlin
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.geometry import (
    get_local_aabc_for_actor,
    get_local_axis_aligned_bbox_for_link,
)
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import mani_skill2.envs

import argparse
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode

@wp.kernel
def success_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    center: wp.vec2,
    radius: float,
    height: float,
    h1: float,
    h2: float,
    output: wp.array(dtype=int),
):
    tid = wp.tid()
    x = particle_q[tid]
    a = x[0] - center[0]
    b = x[1] - center[1]
    z = x[2]
    if a * a + b * b < radius * radius and z < height:
        wp.atomic_add(output, 3, 1)
        if z > h1:
            wp.atomic_add(output, 0, 1)
        if z > h2:
            wp.atomic_add(output, 1, 1)
    else:
        # spill
        if z < 0.001:
            wp.atomic_add(output, 2, 1)


def create_ring():
    segs = 16
    angles = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)
    vs = np.zeros((segs, 3))
    vs[:, 0] = xs
    vs[:, 1] = ys

    vs2 = vs.copy()
    vs2[:, 2] = 1

    vertices = np.concatenate([vs, vs2], 0)
    indices = []
    for i in range(segs):
        a = i
        b = (i + 1) % segs
        c = b + segs
        d = a + segs
        indices.append(a)
        indices.append(b)
        indices.append(c)
        indices.append(a)
        indices.append(c)
        indices.append(d)

        indices.append(a)
        indices.append(c)
        indices.append(b)
        indices.append(a)
        indices.append(d)
        indices.append(c)

    return vertices, np.array(indices)


@register_env("scooping", max_episode_steps=350)
class scoopingEnv(MPMBaseEnv):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.robot_uid = "xarm6"
        self._ring = None
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        super()._load_actors()

        # b = self._scene.create_actor_builder()
        # b.add_box_collision(half_size=[0.12, 0.02, 0.03])
        # b.add_box_visual(half_size=[0.12, 0.02, 0.03])
        # w0 = b.build_kinematic("wall")
        # w1 = b.build_kinematic("wall")
        # w2 = b.build_kinematic("wall")
        # w3 = b.build_kinematic("wall")

        # w0.set_pose(sapien.Pose([0, -0.1, 0.03]))
        # w1.set_pose(sapien.Pose([0, 0.1, 0.03]))
        # w2.set_pose(sapien.Pose([-0.1, 0, 0.03], [0.7071068, 0, 0, 0.7071068]))
        # w3.set_pose(sapien.Pose([0.1, 0, 0.03], [0.7071068, 0, 0, 0.7071068]))
        # self.walls = [w0, w1, w2, w3]

        # beaker_file = os.path.join(
        #     PACKAGE_ASSET_DIR, "deformable_manipulation", "beaker.glb"
        # )
        # target_radius = 0.12
        # b = self._scene.create_actor_builder()
        # b.add_visual_from_file(beaker_file, scale=[target_radius] * 3)
        # b.add_collision_from_file(beaker_file, scale=[target_radius] * 3, density=300)
        # self.target_beaker = b.build("target_beaker")
        # self.target_aabb = get_local_axis_aligned_bbox_for_link(self.target_beaker)
        # self.target_aabc = get_local_aabc_for_actor(self.target_beaker)


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
        self.source_aabb = get_local_axis_aligned_bbox_for_link(self.source_container)

        spoon_dir = os.path.join(
            PACKAGE_ASSET_DIR, "descriptions/feeding/meshes/spoon.STL"
            )
        pose = sapien.Pose([0.0025, -0.1105, 0.2],[-0.007,-0.699,0.007,0.715])
        b = self._scene.create_actor_builder()
        b.add_visual_from_file(spoon_dir, pose, scale=[0.001] * 3)
        b.add_collision_from_file(spoon_dir, pose, scale=[0.001] * 3, density=300)
        self.spoon = b.build("spoon")
        self.source_aabb = get_local_axis_aligned_bbox_for_link(self.source_container)

    def _get_coupling_actors(
        self,
    ):
        return [
            (self.source_container, "visual"),
            (self.spoon, "visual"),
        ]

    def _configure_agent(self):
        self._agent_cfg = xarm6DefaultConfig()

    def _load_agent(self):
        self.agent = xarm6(
            self._scene,
            self._control_freq,
            control_mode=self._control_mode,
            config=self._agent_cfg,
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), "drive_joint"
        )
        self.lfinger = get_entity_by_name(
            self.agent.robot.get_links(), "left_inner_knuckle_joint"
        )
        self.rfinger = get_entity_by_name(
            self.agent.robot.get_links(), "right_inner_knuckle_joint"
        )

    def _setup_mpm(self):
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(
            domain_size=[0.8, 0.8, 0.8], grid_length=0.005
        )
        self.model_builder.reserve_mpm_particles(count=self.max_particles)

        self._setup_mpm_bodies()

        self.mpm_simulator = MPMSimulator(device="cuda")
        self.mpm_model = self.model_builder.finalize(device="cuda")
        self.mpm_model.gravity = np.array((0.0, 0.0, -9.81), dtype=np.float32)
        self.mpm_model.struct.ground_normal = wp.vec3(0.0, 0.0, 1.0)
        self.mpm_model.struct.particle_radius = 0.005
        self.mpm_states = [
            self.mpm_model.state() for _ in range(self._mpm_step_per_sapien_step + 1)
        ]
        self._success_helper = wp.zeros(4, dtype=int, device=self.mpm_model.device)

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
        self.mpm_model.struct.static_ke = 1.0
        self.mpm_model.struct.static_kd = 0.0
        self.mpm_model.struct.static_mu = 1.0
        self.mpm_model.struct.static_ka = 0.0

        self.mpm_model.struct.body_ke = 1.0
        self.mpm_model.struct.body_kd = 0.0
        self.mpm_model.struct.body_mu = 1.0
        self.mpm_model.struct.body_ka = 0.0

        self.mpm_model.adaptive_grid = True

        self.mpm_model.grid_contact = True
        self.mpm_model.particle_contact = True
        self.mpm_model.struct.body_sticky = 1
        self.mpm_model.struct.ground_sticky = 1

        self.mpm_model.struct.particle_radius = 0.0025

    def _initialize_actors(self):
        super()._initialize_actors()
        self.target_height = 0.2
        self.target_num = self._episode_rng.choice(range(250, 1150), 1)[0]
        self.mpm_model.struct.n_particles = len(self.model_builder.mpm_particle_q)

        # self.target_beaker.set_pose(self._target_pos)

    def _initialize_agent(self):
        qpos = np.array([-0.139, 0.417, -1.811, -0.035, 1.442, -0.176, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85])
        qpos[:-1] += self._episode_rng.normal(0, 0.02, len(qpos) - 1)
        qpos[-1] += self._episode_rng.normal(0, 0.2, 1)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.56, 0, 0]))
        self.home_ee = np.array([0.0, 0.0, self.target_height])

    def _register_cameras(self):
        p, q = [-0.2, -0, 0.4], euler2quat(0, np.pi / 6, 0)
        return CameraConfig("base_camera", p, q, 128, 128, np.pi / 2, 0.001, 10)

    def _register_render_cameras(self):
        p, q = [-0.35, -0, 0.4], euler2quat(0, np.pi / 6, 0)
        return CameraConfig("render_camera", p, q, 512, 512, 1, 0.001, 10)

    # def initialize_episode(self):
    #     super().initialize_episode()

    #     self.h1 = self._episode_rng.uniform(0.01, 0.02)
    #     self.h2 = self.h1 + 0.004

    #     if self._ring is not None:
    #         self._scene.remove_actor(self._ring)

    #     vertices, indices = create_ring()
    #     b = self._scene.create_actor_builder()
    #     mesh = self._renderer.create_mesh(
    #         vertices.reshape((-1, 3)), indices.reshape((-1, 3))
    #     )
    #     mat = self._renderer.create_material()
    #     mat.set_base_color([1, 0, 0, 1])
    #     b.add_visual_from_mesh(
    #         mesh,
    #         scale=[
    #             self._target_radius * 1.02,
    #             self._target_radius * 1.02,
    #             self.h2 - self.h1,
    #         ],
    #         material=mat,
    #     )
        # ring = b.build_kinematic("ring")
        # ring.set_pose(sapien.Pose([*self.target_beaker.pose.p[:2], self.h1]))
        # self._ring = ring

    def _clear(self):
        if self._ring is not None:
            self._scene.remove_actor(self._ring)
            self._ring = None
        super()._clear()

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(-0.05, 0.3, 0.3)
        self._viewer.set_camera_rpy(0.0, -0.7, 1.57)

    def _determine_target_pos(self):
        pmodel = self.agent.robot.create_pinocchio_model()
        hand = next(
            l for l in self.agent.robot.get_links() if l.name == "drive_joint"
        )
        while True:
            r = self._episode_rng.uniform(0.2, 0.25)
            t = self._episode_rng.uniform(0, np.pi)
            self._target_pos = sapien.Pose([r * np.cos(t), r * np.sin(t), 0.0])

            r = self._episode_rng.uniform(0.05, 0.1)
            t = self._episode_rng.uniform(np.pi, np.pi * 2)
            self._source_pos = sapien.Pose([r * np.cos(t), r * np.sin(t), 0.0])

            from transforms3d.quaternions import axangle2quat, qmult

            q = qmult(
                axangle2quat(
                    [0, 0, 1], self._episode_rng.uniform(-np.pi / 8, np.pi / 8)
                ),
                [0.5, -0.5, -0.5, -0.5],
            )

            result, success, error = pmodel.compute_inverse_kinematics(
                hand.get_index(),
                sapien.Pose(
                    [
                        self._source_pos.p[0] + 0.55,
                        self._source_pos.p[1],
                        self._episode_rng.uniform(0.04, 0.06),
                    ],
                    q,
                ),
                [-0.555, 0.646, 0.181, -1.892, 1.171, 1.423, -1.75, 0.04, 0.04],
                active_qmask=[1] * 7 + [0] * 2,
            )
            if not success:
                continue

            result[-2:] = 0.04
            self._init_qpos = result
            return

    def _get_obs_extra(self) -> OrderedDict:
        return OrderedDict(
            tcp_pose=vectorize_pose(self.grasp_site.get_pose()),
            target=np.array([self.h1]),
        )

    # def in_beaker_num(self):
    #     self._success_helper.zero_()
    #     wp.launch(
    #         success_kernel,
    #         dim=self.mpm_model.struct.n_particles,
    #         inputs=[
    #             self.mpm_states[0].struct.particle_q,
    #             self.target_beaker.pose.p[:2],
    #             self._target_radius,
    #             self._target_height,
    #             self.h1,
    #             self.h2,
    #             self._success_helper,
    #         ],
    #         device=self.mpm_model.device,
    #     )
    #     above_start, above_end, spill, in_beaker = self._success_helper.numpy()
    #     return above_start, above_end

    def evaluate(self, **kwargs):
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

    # def compute_dense_reward(self, reward_info=False, **kwargs):
    #     if self.evaluate()["success"]:
    #         if reward_info:
    #             return {"reward": 6.0}
    #         return 6.0
    #     particles_x = self.get_mpm_state()["x"]

    #     stage = 0

    #     # spill reward
    #     spill_num = self.n_particles - len(
    #         np.where(
    #             (particles_x[:, 0] > -0.12)
    #             & (particles_x[:, 0] < 0.12)
    #             & (particles_x[:, 1] > -0.12)
    #             & (particles_x[:, 1] < 0.12)
    #         )[0]
    #     )
    #     spill_reward = -spill_num / 100

    #     (
    #         bucket_base_pos,
    #         bucket_tlpos,
    #         bucket_trpos,
    #         bucket_blpos,
    #         bucket_brpos,
    #         gripper_mat,
    #     ) = self._bucket_keypoints()

    #     lifted_particles = self.particles_inside_bucket()
    #     lift_num = len(lifted_particles)
    #     lift_reward = (
    #         min(lift_num / self.target_num, 1)
    #         - max(0, lift_num - self.target_num - 500) * 0.001
    #     )

    #     gripper_pos = self.grasp_site.get_pose().p
    #     height_dist = (
    #         max(self.target_height + 0.05 - np.mean(lifted_particles[:, 2]), 0)
    #         if len(lifted_particles) > 0
    #         else 1
    #     )
    #     # reaching reward & height reward & flat reward
    #     if height_dist > 0.1 and lift_num > self.target_num + 300:
    #         reaching_reward = 1
    #         height_reward = 1 - np.tanh(3 * height_dist)
    #         flat_dist = 0.5 * (
    #             max(bucket_base_pos[2] + 0.01 - bucket_blpos[2], 0)
    #             + max(bucket_blpos[2] - bucket_brpos[2], 0)
    #         )
    #         flat_reward = 1 - np.tanh(50 * flat_dist)
    #         stage = 1
    #     elif height_dist <= 0.1:
    #         lift_reward = (
    #             1
    #             + min(lift_num / self.target_num, 1)
    #             - max(0, lift_num - self.target_num - 100) * 0.001
    #         )
    #         reaching_reward = 1
    #         height_reward = 1 - np.tanh(3 * height_dist)
    #         flat_dist = 0.5 * (
    #             max(bucket_base_pos[2] - 0.01 - bucket_blpos[2], 0)
    #             + max(bucket_blpos[2] - bucket_brpos[2], 0)
    #         )
    #         flat_reward = 1 - np.tanh(50 * flat_dist)
    #         stage = 2
    #     else:
    #         if (
    #             gripper_pos[0] > -0.1
    #             and gripper_pos[0] < 0.1
    #             and gripper_pos[1] > -0.1
    #             and gripper_pos[1] < 0.1
    #         ):
    #             dist = gripper_pos[2] + max(0.04 - gripper_pos[0], 0)
    #             reaching_reward = 1 - np.tanh(10 * dist)
    #         else:
    #             reaching_reward = 0
    #         height_reward = 0
    #         flat_reward = 0

    #     reward = (
    #         reaching_reward * 0.5
    #         + lift_reward
    #         + height_reward
    #         + spill_reward
    #         + flat_reward
    #     )
    #     if reward_info:
    #         return {
    #             "reward": reward,
    #             "reaching_reward": reaching_reward,
    #             "lift_reward": lift_reward,
    #             "lift_num": lift_num,
    #             "target_num": self.target_num,
    #             "flat_reward": flat_reward,
    #             "height_reward": height_reward,
    #             "spill_reward": spill_reward,
    #             "stage": stage,
    #             "height_dist": height_dist,
    #         }
    #     return reward

    def compute_normalized_dense_reward(self, **kwargs):
    #     return self.compute_dense_reward(**kwargs) / 6.0
        return -1.0

    def get_mpm_state(self):
        n = self.mpm_model.struct.n_particles

        return OrderedDict(
            x=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_q, n),
            v=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_qd, n),
            F=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_F, n),
            C=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_C, n),
            vol=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_vol, n),
        )

    def set_mpm_state(self, state):
        self.mpm_states[0].struct.particle_q.assign(state["x"])
        self.mpm_states[0].struct.particle_qd.assign(state["v"])
        self.mpm_states[0].struct.particle_F.assign(state["F"])
        self.mpm_states[0].struct.particle_C.assign(state["C"])
        # self.mpm_states[0].struct.particle_vol.assign(state["vol"])

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.target_num])

    def set_state(self, state):
        self.target_num = state[-1]
        super().set_state(state[:-1])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="scooping")
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args

if __name__ == "__main__":

    args = parse_args()
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        **args.env_kwargs
        )
    env.reset()
    after_reset = True
    env.agent.set_control_mode("pd_ee_delta_pose")

    a = env.get_state()
    env.set_state(a)

    for i in range(100):
        env.step(None)
        env.render()

    # Viewer
    if args.enable_sapien_viewer:
        env.render_human()
    opencv_viewer = OpenCVViewer(exit_on_esc=False)

    while True:
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        if args.enable_sapien_viewer:
            env.render_human()

        render_frame = env.render()

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            if args.enable_sapien_viewer:
                opencv_viewer.close()
                opencv_viewer = OpenCVViewer(exit_on_esc=False)