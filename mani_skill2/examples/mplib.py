import argparse

import gymnasium as gym
import numpy as np

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
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

def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        **args.env_kwargs
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, render_mode=args.render_mode)


    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    obs, _ = env.reset()
    after_reset = True

    # Viewer
    if args.enable_sapien_viewer:
        env.render_human()
    opencv_viewer = OpenCVViewer(exit_on_esc=False)

    # def render_wait():
    #     if not args.enable_sapien_viewer:
    #         return
    #     while True:
    #         env.render_human()
    #         sapien_viewer = env.viewer
    #         if sapien_viewer.window.key_down("0"):
    #             break

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
                
        if args.render_mode is not None:
            env.render()

        # if terminated or truncated:
        #     break
    env.close()

if __name__ == "__main__":
    main()