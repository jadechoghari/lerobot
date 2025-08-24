from collections import defaultdict
from typing import Any, Callable
from itertools import chain
import gymnasium as gym
import metaworld
import numpy as np
from gymnasium import spaces
from metaworld.policies import (
    SawyerAssemblyV3Policy,
    SawyerBasketballV3Policy,
    SawyerBinPickingV3Policy,
    SawyerBoxCloseV3Policy,
    SawyerButtonPressTopdownV3Policy,
    SawyerButtonPressTopdownWallV3Policy,
    SawyerButtonPressV3Policy,
    SawyerButtonPressWallV3Policy,
    SawyerCoffeeButtonV3Policy,
    SawyerCoffeePullV3Policy,
    SawyerCoffeePushV3Policy,
    SawyerDialTurnV3Policy,
    SawyerDisassembleV3Policy,
    SawyerDoorCloseV3Policy,
    SawyerDoorLockV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerDoorUnlockV3Policy,
    SawyerDrawerCloseV3Policy,
    SawyerDrawerOpenV3Policy,
    SawyerFaucetCloseV3Policy,
    SawyerFaucetOpenV3Policy,
    SawyerHammerV3Policy,
    SawyerHandInsertV3Policy,
    SawyerHandlePressSideV3Policy,
    SawyerHandlePressV3Policy,
    SawyerHandlePullSideV3Policy,
    SawyerHandlePullV3Policy,
    SawyerLeverPullV3Policy,
    SawyerPegInsertionSideV3Policy,
    SawyerPegUnplugSideV3Policy,
    SawyerPickOutOfHoleV3Policy,
    SawyerPickPlaceV3Policy,
    SawyerPickPlaceWallV3Policy,
    SawyerPlateSlideBackSideV3Policy,
    SawyerPlateSlideBackV3Policy,
    SawyerPlateSlideSideV3Policy,
    SawyerPlateSlideV3Policy,
    SawyerPushBackV3Policy,
    SawyerPushV3Policy,
    SawyerPushWallV3Policy,
    SawyerReachV3Policy,
    SawyerReachWallV3Policy,
    SawyerShelfPlaceV3Policy,
    SawyerSoccerV3Policy,
    SawyerStickPullV3Policy,
    SawyerStickPushV3Policy,
    SawyerSweepIntoV3Policy,
    SawyerSweepV3Policy,
    SawyerWindowCloseV3Policy,
    SawyerWindowOpenV3Policy,
)

TASK_DESCRIPTIONS = {
    "assembly-v3": "Pick up a nut and place it onto a peg",
    "basketball-v3": "Dunk the basketball into the basket",
    "bin-picking-v3": "Grasp the puck from one bin and place it into another bin",
    "box-close-v3": "Grasp the cover and close the box with it",
    "button-press-topdown-v3": "Press a button from the top",
    "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top",
    "button-press-v3": "Press a button",
    "button-press-wall-v3": "Bypass a wall and press a button",
    "coffee-button-v3": "Push a button on the coffee machine",
    "coffee-pull-v3": "Pull a mug from a coffee machine",
    "coffee-push-v3": "Push a mug under a coffee machine",
    "dial-turn-v3": "Rotate a dial 180 degrees",
    "disassemble-v3": "Pick a nut out of a peg",
    "door-close-v3": "Close a door with a revolving joint",
    "door-lock-v3": "Lock the door by rotating the lock clockwise",
    "door-open-v3": "Open a door with a revolving joint",
    "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise",
    "hand-insert-v3": "Insert the gripper into a hole",
    "drawer-close-v3": "Push and close a drawer",
    "drawer-open-v3": "Open a drawer",
    "faucet-open-v3": "Rotate the faucet counter-clockwise",
    "faucet-close-v3": "Rotate the faucet clockwise",
    "hammer-v3": "Hammer a screw on the wall",
    "handle-press-side-v3": "Press a handle down sideways",
    "handle-press-v3": "Press a handle down",
    "handle-pull-side-v3": "Pull a handle up sideways",
    "handle-pull-v3": "Pull a handle up",
    "lever-pull-v3": "Pull a lever down 90 degrees",
    "peg-insert-side-v3": "Insert a peg sideways",
    "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck",
    "pick-out-of-hole-v3": "Pick up a puck from a hole",
    "reach-v3": "Reach a goal position",
    "push-back-v3": "Push the puck to a goal",
    "push-v3": "Push the puck to a goal",
    "pick-place-v3": "Pick and place a puck to a goal",
    "plate-slide-v3": "Slide a plate into a cabinet",
    "plate-slide-side-v3": "Slide a plate into a cabinet sideways",
    "plate-slide-back-v3": "Get a plate from the cabinet",
    "plate-slide-back-side-v3": "Get a plate from the cabinet sideways",
    "peg-unplug-side-v3": "Unplug a peg sideways",
    "soccer-v3": "Kick a soccer into the goal",
    "stick-push-v3": "Grasp a stick and push a box using the stick",
    "stick-pull-v3": "Grasp a stick and pull a box with the stick",
    "push-wall-v3": "Bypass a wall and push a puck to a goal",
    "reach-wall-v3": "Bypass a wall and reach a goal",
    "shelf-place-v3": "Pick and place a puck onto a shelf",
    "sweep-into-v3": "Sweep a puck into a hole",
    "sweep-v3": "Sweep a puck off the table",
    "window-open-v3": "Push and open a window",
    "window-close-v3": "Push and close a window",
}

TASK_POLICY_MAPPING = {
    "assembly-v3": SawyerAssemblyV3Policy,
    "basketball-v3": SawyerBasketballV3Policy,
    "bin-picking-v3": SawyerBinPickingV3Policy,
    "box-close-v3": SawyerBoxCloseV3Policy,
    "button-press-topdown-v3": SawyerButtonPressTopdownV3Policy,
    "button-press-topdown-wall-v3": SawyerButtonPressTopdownWallV3Policy,
    "button-press-v3": SawyerButtonPressV3Policy,
    "button-press-wall-v3": SawyerButtonPressWallV3Policy,
    "coffee-button-v3": SawyerCoffeeButtonV3Policy,
    "coffee-pull-v3": SawyerCoffeePullV3Policy,
    "coffee-push-v3": SawyerCoffeePushV3Policy,
    "dial-turn-v3": SawyerDialTurnV3Policy,
    "disassemble-v3": SawyerDisassembleV3Policy,
    "door-close-v3": SawyerDoorCloseV3Policy,
    "door-lock-v3": SawyerDoorLockV3Policy,
    "door-open-v3": SawyerDoorOpenV3Policy,
    "door-unlock-v3": SawyerDoorUnlockV3Policy,
    "drawer-close-v3": SawyerDrawerCloseV3Policy,
    "drawer-open-v3": SawyerDrawerOpenV3Policy,
    "faucet-close-v3": SawyerFaucetCloseV3Policy,
    "faucet-open-v3": SawyerFaucetOpenV3Policy,
    "hammer-v3": SawyerHammerV3Policy,
    "hand-insert-v3": SawyerHandInsertV3Policy,
    "handle-press-side-v3": SawyerHandlePressSideV3Policy,
    "handle-press-v3": SawyerHandlePressV3Policy,
    "handle-pull-side-v3": SawyerHandlePullSideV3Policy,
    "handle-pull-v3": SawyerHandlePullV3Policy,
    "lever-pull-v3": SawyerLeverPullV3Policy,
    "peg-insert-side-v3": SawyerPegInsertionSideV3Policy,
    "peg-unplug-side-v3": SawyerPegUnplugSideV3Policy,
    "pick-out-of-hole-v3": SawyerPickOutOfHoleV3Policy,
    "pick-place-v3": SawyerPickPlaceV3Policy,
    "pick-place-wall-v3": SawyerPickPlaceWallV3Policy,
    "plate-slide-back-side-v3": SawyerPlateSlideBackSideV3Policy,
    "plate-slide-back-v3": SawyerPlateSlideBackV3Policy,
    "plate-slide-side-v3": SawyerPlateSlideSideV3Policy,
    "plate-slide-v3": SawyerPlateSlideV3Policy,
    "push-back-v3": SawyerPushBackV3Policy,
    "push-v3": SawyerPushV3Policy,
    "push-wall-v3": SawyerPushWallV3Policy,
    "reach-v3": SawyerReachV3Policy,
    "reach-wall-v3": SawyerReachWallV3Policy,
    "shelf-place-v3": SawyerShelfPlaceV3Policy,
    "soccer-v3": SawyerSoccerV3Policy,
    "stick-pull-v3": SawyerStickPullV3Policy,
    "stick-push-v3": SawyerStickPushV3Policy,
    "sweep-into-v3": SawyerSweepIntoV3Policy,
    "sweep-v3": SawyerSweepV3Policy,
    "window-open-v3": SawyerWindowOpenV3Policy,
    "window-close-v3": SawyerWindowCloseV3Policy,
}

TASK_NAME_TO_ID = {
    "assembly-v3": 0,
    "basketball-v3": 1,
    "bin-picking-v3": 2,
    "box-close-v3": 3,
    "button-press-topdown-v3": 4,
    "button-press-topdown-wall-v3": 5,
    "button-press-v3": 6,
    "button-press-wall-v3": 7,
    "coffee-button-v3": 8,
    "coffee-pull-v3": 9,
    "coffee-push-v3": 10,
    "dial-turn-v3": 11,
    "disassemble-v3": 12,
    "door-close-v3": 13,
    "door-lock-v3": 14,
    "door-open-v3": 15,
    "door-unlock-v3": 16,
    "drawer-close-v3": 17,
    "drawer-open-v3": 18,
    "faucet-close-v3": 19,
    "faucet-open-v3": 20,
    "hammer-v3": 21,
    "hand-insert-v3": 22,
    "handle-press-side-v3": 23,
    "handle-press-v3": 24,
    "handle-pull-side-v3": 25,
    "handle-pull-v3": 26,
    "lever-pull-v3": 27,
    "peg-insert-side-v3": 28,
    "peg-unplug-side-v3": 29,
    "pick-out-of-hole-v3": 30,
    "pick-place-v3": 31,
    "pick-place-wall-v3": 32,
    "plate-slide-back-side-v3": 33,
    "plate-slide-back-v3": 34,
    "plate-slide-side-v3": 35,
    "plate-slide-v3": 36,
    "push-back-v3": 37,
    "push-v3": 38,
    "push-wall-v3": 39,
    "reach-v3": 40,
    "reach-wall-v3": 41,
    "shelf-place-v3": 42,
    "soccer-v3": 43,
    "stick-pull-v3": 44,
    "stick-push-v3": 45,
    "sweep-into-v3": 46,
    "sweep-v3": 47,
    "window-open-v3": 48,
    "window-close-v3": 49,
}
DIFFICULTY_TO_TASKS = {
    "easy": [
        "button-press-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-wall-v3",
        "coffee-button-v3",
        "dial-turn-v3",
        "door-close-v3",
        "door-lock-v3",
        "door-open-v3",
        "door-unlock-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-close-v3",
        "faucet-open-v3",
        "handle-press-v3",
        "handle-press-side-v3",
        "handle-pull-v3",
        "handle-pull-side-v3",
        "lever-pull-v3",
        "plate-slide-v3",
        "plate-slide-back-v3",
        "plate-slide-back-side-v3",
        "plate-slide-side-v3",
        "reach-v3",
        "reach-wall-v3",
        "window-close-v3",
        "window-open-v3",
        "peg-unplug-side-v3",
    ],
    "medium": [
        "basketball-v3",
        "bin-picking-v3",
        "box-close-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "hammer-v3",
        "peg-insert-side-v3",
        "push-wall-v3",
        "soccer-v3",
        "sweep-v3",
        "sweep-into-v3",
    ],
    "hard": [
        "assembly-v3",
        "hand-insert-v3",
        "pick-out-of-hole-v3",
        "pick-place-v3",
        "push-v3",
        "push-back-v3",
    ],
    "very_hard": [
        "shelf-place-v3",
        "disassemble-v3",
        "stick-pull-v3",
        "stick-push-v3",
        "pick-place-wall-v3",
    ],
}

def create_metaworld_envs(task: str, n_envs: int, gym_kwargs: dict[str, Any] = None, env_cls: Callable = None, multitask_eval: bool = True)  -> dict[str, dict[str, Any]]:
    if gym_kwargs is None:
        gym_kwargs = {}
    if not multitask_eval:
        tasks = task.split(",")
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(n_envs)]
        elif len(tasks) < n_envs and n_envs % len(tasks) == 0:
            n_repeat = n_envs // len(tasks)
            tasks = list(chain.from_iterable([[item] * n_repeat for item in tasks]))
        elif n_envs < len(tasks):
            tasks = tasks[:n_envs]
        assert n_envs == len(tasks), "n_envs and len(tasks) must be the same!"
        print(f"Creating Meta-World envs with tasks {tasks}")
        return env_cls([lambda i=i: MetaworldEnv(task=tasks[i], **gym_kwargs) for i in range(n_envs)])
    else:
        envs = defaultdict(dict)
        tasks = task.split(",")
        if tasks[0] not in DIFFICULTY_TO_TASKS: # evaluation on individual tasks
            task_groups = ["all"]
        else:
            task_groups = tasks
        for task_group in task_groups:
            _tasks = DIFFICULTY_TO_TASKS.get(task_group, task_groups)
            breakpoint()
            for _task in _tasks:
                print(f"Creating Meta-World envs with task {_task} from task group {task_group}")
                envs_list = [lambda i=i: MetaworldEnv(task=_task, **gym_kwargs) for i in range(n_envs)]
                envs[task_group][_task] = env_cls(envs_list)
        return envs


class MetaworldEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task,
        camera_name="corner2",
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task.replace("metaworld-", "")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = 480
        self.observation_height = 480
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.camera_name = camera_name

        self._env = self._make_envs_task(self.task)
        self._max_episode_steps = self._env.max_path_length
        self.task_description = TASK_DESCRIPTIONS[self.task]

        self.expert_policy = TASK_POLICY_MAPPING[self.task]()

        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(4,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def render(self):
        image = self._env.render()
        if self.camera_name == "corner2":
            image = np.flip(image, (0, 1))  # images for some reason are flipped
        return image

    def _make_envs_task(self, env_name: str):
        mt1 = metaworld.MT1(env_name, seed=42)
        env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name=self.camera_name)
        env.set_task(mt1.train_tasks[0])
        if self.camera_name == "corner2":
            env.model.cam_pos[2] = [
                0.75,
                0.075,
                0.7,
            ]  # corner2 position, similar to https://arxiv.org/pdf/2206.14244
        env.reset()
        env._freeze_rand_vec = False  # otherwise no randomization
        return env

    def _format_raw_obs(self, raw_obs, env=None):
        image = None
        if env is not None:
            image = env.render()
            if self.camera_name == "corner2":
                image = np.flip(image, (0, 1))  # images for some reason are flipped
        agent_pos = raw_obs[:4]
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"pixels": image.copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": image.copy(),
                "agent_pos": agent_pos,
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        raw_obs, info = self._env.reset(seed=seed)

        observation = self._format_raw_obs(raw_obs, env=self._env)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        raw_obs, reward, done, truncated, info = self._env.step(action)

        terminated = is_success = int(info["success"]) == 1
        info["is_success"] = is_success

        observation = self._format_raw_obs(raw_obs, env=self._env)

        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()