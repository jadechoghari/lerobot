from collections import defaultdict
from typing import Any, Callable
from itertools import chain
import gymnasium as gym
import metaworld
import numpy as np
from gymnasium import spaces
from metaworld.policies import (
    SawyerAssemblyV2Policy,
    SawyerBasketballV2Policy,
    SawyerBinPickingV2Policy,
    SawyerBoxCloseV2Policy,
    SawyerButtonPressTopdownV2Policy,
    SawyerButtonPressTopdownWallV2Policy,
    SawyerButtonPressV2Policy,
    SawyerButtonPressWallV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerCoffeePullV2Policy,
    SawyerCoffeePushV2Policy,
    SawyerDialTurnV2Policy,
    SawyerDisassembleV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorOpenV2Policy,
    SawyerDoorUnlockV2Policy,
    SawyerDrawerCloseV2Policy,
    SawyerDrawerOpenV2Policy,
    SawyerFaucetCloseV2Policy,
    SawyerFaucetOpenV2Policy,
    SawyerHammerV2Policy,
    SawyerHandInsertV2Policy,
    SawyerHandlePressSideV2Policy,
    SawyerHandlePressV2Policy,
    SawyerHandlePullSideV2Policy,
    SawyerHandlePullV2Policy,
    SawyerLeverPullV2Policy,
    SawyerPegInsertionSideV2Policy,
    SawyerPegUnplugSideV2Policy,
    SawyerPickOutOfHoleV2Policy,
    SawyerPickPlaceV2Policy,
    SawyerPickPlaceWallV2Policy,
    SawyerPlateSlideBackSideV2Policy,
    SawyerPlateSlideBackV2Policy,
    SawyerPlateSlideSideV2Policy,
    SawyerPlateSlideV2Policy,
    SawyerPushBackV2Policy,
    SawyerPushV2Policy,
    SawyerPushWallV2Policy,
    SawyerReachV2Policy,
    SawyerReachWallV2Policy,
    SawyerShelfPlaceV2Policy,
    SawyerSoccerV2Policy,
    SawyerStickPullV2Policy,
    SawyerStickPushV2Policy,
    SawyerSweepIntoV2Policy,
    SawyerSweepV2Policy,
    SawyerWindowCloseV2Policy,
    SawyerWindowOpenV2Policy,
)

TASK_DESCRIPTIONS = {
    "assembly-v2": "Pick up a nut and place it onto a peg",
    "basketball-v2": "Dunk the basketball into the basket",
    "bin-picking-v2": "Grasp the puck from one bin and place it into another bin",
    "box-close-v2": "Grasp the cover and close the box with it",
    "button-press-topdown-v2": "Press a button from the top",
    "button-press-topdown-wall-v2": "Bypass a wall and press a button from the top",
    "button-press-v2": "Press a button",
    "button-press-wall-v2": "Bypass a wall and press a button",
    "coffee-button-v2": "Push a button on the coffee machine",
    "coffee-pull-v2": "Pull a mug from a coffee machine",
    "coffee-push-v2": "Push a mug under a coffee machine",
    "dial-turn-v2": "Rotate a dial 180 degrees",
    "disassemble-v2": "Pick a nut out of a peg",
    "door-close-v2": "Close a door with a revolving joint",
    "door-lock-v2": "Lock the door by rotating the lock clockwise",
    "door-open-v2": "Open a door with a revolving joint",
    "door-unlock-v2": "Unlock the door by rotating the lock counter-clockwise",
    "hand-insert-v2": "Insert the gripper into a hole",
    "drawer-close-v2": "Push and close a drawer",
    "drawer-open-v2": "Open a drawer",
    "faucet-open-v2": "Rotate the faucet counter-clockwise",
    "faucet-close-v2": "Rotate the faucet clockwise",
    "hammer-v2": "Hammer a screw on the wall",
    "handle-press-side-v2": "Press a handle down sideways",
    "handle-press-v2": "Press a handle down",
    "handle-pull-side-v2": "Pull a handle up sideways",
    "handle-pull-v2": "Pull a handle up",
    "lever-pull-v2": "Pull a lever down 90 degrees",
    "peg-insert-side-v2": "Insert a peg sideways",
    "pick-place-wall-v2": "Pick a puck, bypass a wall and place the puck",
    "pick-out-of-hole-v2": "Pick up a puck from a hole",
    "reach-v2": "Reach a goal position",
    "push-back-v2": "Push the puck to a goal",
    "push-v2": "Push the puck to a goal",
    "pick-place-v2": "Pick and place a puck to a goal",
    "plate-slide-v2": "Slide a plate into a cabinet",
    "plate-slide-side-v2": "Slide a plate into a cabinet sideways",
    "plate-slide-back-v2": "Get a plate from the cabinet",
    "plate-slide-back-side-v2": "Get a plate from the cabinet sideways",
    "peg-unplug-side-v2": "Unplug a peg sideways",
    "soccer-v2": "Kick a soccer into the goal",
    "stick-push-v2": "Grasp a stick and push a box using the stick",
    "stick-pull-v2": "Grasp a stick and pull a box with the stick",
    "push-wall-v2": "Bypass a wall and push a puck to a goal",
    "reach-wall-v2": "Bypass a wall and reach a goal",
    "shelf-place-v2": "Pick and place a puck onto a shelf",
    "sweep-into-v2": "Sweep a puck into a hole",
    "sweep-v2": "Sweep a puck off the table",
    "window-open-v2": "Push and open a window",
    "window-close-v2": "Push and close a window",
}


TASK_POLICY_MAPPING = {
    "assembly-v2": SawyerAssemblyV2Policy,
    "basketball-v2": SawyerBasketballV2Policy,
    "bin-picking-v2": SawyerBinPickingV2Policy,
    "box-close-v2": SawyerBoxCloseV2Policy,
    "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy,
    "button-press-topdown-wall-v2": SawyerButtonPressTopdownWallV2Policy,
    "button-press-v2": SawyerButtonPressV2Policy,
    "button-press-wall-v2": SawyerButtonPressWallV2Policy,
    "coffee-button-v2": SawyerCoffeeButtonV2Policy,
    "coffee-pull-v2": SawyerCoffeePullV2Policy,
    "coffee-push-v2": SawyerCoffeePushV2Policy,
    "dial-turn-v2": SawyerDialTurnV2Policy,
    "disassemble-v2": SawyerDisassembleV2Policy,
    "door-close-v2": SawyerDoorCloseV2Policy,
    "door-lock-v2": SawyerDoorLockV2Policy,
    "door-open-v2": SawyerDoorOpenV2Policy,
    "door-unlock-v2": SawyerDoorUnlockV2Policy,
    "drawer-close-v2": SawyerDrawerCloseV2Policy,
    "drawer-open-v2": SawyerDrawerOpenV2Policy,
    "faucet-close-v2": SawyerFaucetCloseV2Policy,
    "faucet-open-v2": SawyerFaucetOpenV2Policy,
    "hammer-v2": SawyerHammerV2Policy,
    "hand-insert-v2": SawyerHandInsertV2Policy,
    "handle-press-side-v2": SawyerHandlePressSideV2Policy,
    "handle-press-v2": SawyerHandlePressV2Policy,
    "handle-pull-side-v2": SawyerHandlePullSideV2Policy,
    "handle-pull-v2": SawyerHandlePullV2Policy,
    "lever-pull-v2": SawyerLeverPullV2Policy,
    "peg-insert-side-v2": SawyerPegInsertionSideV2Policy,
    "peg-unplug-side-v2": SawyerPegUnplugSideV2Policy,
    "pick-out-of-hole-v2": SawyerPickOutOfHoleV2Policy,
    "pick-place-v2": SawyerPickPlaceV2Policy,
    "pick-place-wall-v2": SawyerPickPlaceWallV2Policy,
    "plate-slide-back-side-v2": SawyerPlateSlideBackSideV2Policy,
    "plate-slide-back-v2": SawyerPlateSlideBackV2Policy,
    "plate-slide-side-v2": SawyerPlateSlideSideV2Policy,
    "plate-slide-v2": SawyerPlateSlideV2Policy,
    "push-back-v2": SawyerPushBackV2Policy,
    "push-v2": SawyerPushV2Policy,
    "push-wall-v2": SawyerPushWallV2Policy,
    "reach-v2": SawyerReachV2Policy,
    "reach-wall-v2": SawyerReachWallV2Policy,
    "shelf-place-v2": SawyerShelfPlaceV2Policy,
    "soccer-v2": SawyerSoccerV2Policy,
    "stick-pull-v2": SawyerStickPullV2Policy,
    "stick-push-v2": SawyerStickPushV2Policy,
    "sweep-into-v2": SawyerSweepIntoV2Policy,
    "sweep-v2": SawyerSweepV2Policy,
    "window-open-v2": SawyerWindowOpenV2Policy,
    "window-close-v2": SawyerWindowCloseV2Policy,
}
TASK_NAME_TO_ID = {
    "assembly-v2": 0,
    "basketball-v2": 1,
    "bin-picking-v2": 2,
    "box-close-v2": 3,
    "button-press-topdown-v2": 4,
    "button-press-topdown-wall-v2": 5,
    "button-press-v2": 6,
    "button-press-wall-v2": 7,
    "coffee-button-v2": 8,
    "coffee-pull-v2": 9,
    "coffee-push-v2": 10,
    "dial-turn-v2": 11,
    "disassemble-v2": 12,
    "door-close-v2": 13,
    "door-lock-v2": 14,
    "door-open-v2": 15,
    "door-unlock-v2": 16,
    "drawer-close-v2": 17,
    "drawer-open-v2": 18,
    "faucet-close-v2": 19,
    "faucet-open-v2": 20,
    "hammer-v2": 21,
    "hand-insert-v2": 22,
    "handle-press-side-v2": 23,
    "handle-press-v2": 24,
    "handle-pull-side-v2": 25,
    "handle-pull-v2": 26,
    "lever-pull-v2": 27,
    "peg-insert-side-v2": 28,
    "peg-unplug-side-v2": 29,
    "pick-out-of-hole-v2": 30,
    "pick-place-v2": 31,
    "pick-place-wall-v2": 32,
    "plate-slide-back-side-v2": 33,
    "plate-slide-back-v2": 34,
    "plate-slide-side-v2": 35,
    "plate-slide-v2": 36,
    "push-back-v2": 37,
    "push-v2": 38,
    "push-wall-v2": 39,
    "reach-v2": 40,
    "reach-wall-v2": 41,
    "shelf-place-v2": 42,
    "soccer-v2": 43,
    "stick-pull-v2": 44,
    "stick-push-v2": 45,
    "sweep-into-v2": 46,
    "sweep-v2": 47,
    "window-open-v2": 48,
    "window-close-v2": 49,
}
DIFFICULTY_TO_TASKS = {
    "easy": [
        "button-press-v2",
        "button-press-topdown-v2",
        "button-press-topdown-wall-v2",
        "button-press-wall-v2",
        "coffee-button-v2",
        "dial-turn-v2",
        "door-close-v2",
        "door-lock-v2",
        "door-open-v2",
        "door-unlock-v2",
        "drawer-close-v2",
        "drawer-open-v2",
        "faucet-close-v2",
        "faucet-open-v2",
        "handle-press-v2",
        "handle-press-side-v2",
        "handle-pull-v2",
        "handle-pull-side-v2",
        "lever-pull-v2",
        "plate-slide-v2",
        "plate-slide-back-v2",
        "plate-slide-back-side-v2",
        "plate-slide-side-v2",
        "reach-v2",
        "reach-wall-v2",
        "window-close-v2",
        "window-open-v2",
        "peg-unplug-side-v2",
    ],
    "medium": [
        "basketball-v2",
        "bin-picking-v2",
        "box-close-v2",
        "coffee-pull-v2",
        "coffee-push-v2",
        "hammer-v2",
        "peg-insert-side-v2",
        "push-wall-v2",
        "soccer-v2",
        "sweep-v2",
        "sweep-into-v2",
    ],
    "hard": [
        "assembly-v2",
        "hand-insert-v2",
        "pick-out-of-hole-v2",
        "pick-place-v2",
        "push-v2",
        "push-back-v2",
    ],
    "very_hard": [
        "shelf-place-v2",
        "disassemble-v2",
        "stick-pull-v2",
        "stick-push-v2",
        "pick-place-wall-v2",
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