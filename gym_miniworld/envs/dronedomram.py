import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

class DroneDomRam(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, max_episode_steps=500, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=max_episode_steps,
            domain_rand=True,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+3)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            # wall_height=4
        )

        self.box = self.place_entity(Box(color='red'), grounded=False, min_y=1.5)
        self.place_agent(min_y=1.6, max_y=1, grounded=False)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box, near_const=2):
            reward += self._reward()
            done = True

        return obs, reward, done, info
