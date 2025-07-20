import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.whisker_racer import binding

class WhiskerRacer(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 llw_ang=-3.14/4, flw_ang=-3.14/6,
                 frw_ang=3.14/6, rrw_ang=3.14/4,
                 max_whisker_length=100,
                 turn_pi_frac=20,
                 maxv=5,
                 continuous=False, log_interval=128,
                 buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
                                            shape=(1,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.continuous = continuous
        self.log_interval = log_interval
        self.tick = 0

        if continuous:
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.single_action_space = gymnasium.spaces.Discrete(3)

        super().__init__(buf)
        if continuous:
            self.actions = self.actions.flatten()
        else:
            self.actions = self.actions.astype(np.float32)

        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, frameskip=frameskip, width=width, height=height,
            paddle_width=paddle_width, paddle_height=paddle_height, ball_width=ball_width, ball_height=ball_height,
            brick_width=brick_width, brick_height=brick_height, brick_rows=brick_rows,
            brick_cols=brick_cols, continuous=continuous
        )

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []
    
    def step(self, action):
        pass

def test_performance(timeout=10, atn_cache=1024):
    env = WhiskerRacer(num_envs=100)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_agents * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
