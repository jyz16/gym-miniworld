import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

class ObstacleRoom(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room, while avoiding yellow box.
    """

    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=max_episode_steps,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)
        # self.reward = 0

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        self.box = self.place_entity(Box(color='red'))
        self.obs_box1 = self.place_entity(Box(color='yellow'))
        self.obs_box2 = self.place_entity(Box(color='yellow'))
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        if self.near(self.obs_box1):
            reward = -2

        if self.near(self.obs_box2):
            reward = -2

        return obs, reward, done, info
