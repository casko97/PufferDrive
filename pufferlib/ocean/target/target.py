'''A simple sample environment. Use this as a template for your own envs.'''

'''
Pain points for docs:
    - Build in C first
    - Make sure obs types match in C and python
    - Getting obs and action spaces and types correct
    - Double check obs are not zero
    - Correct reset behavior
    - Make sure rewards look correct
    - don't forget params/init in binding
    - Use debug mode to catch segaults
    - TODO: Add check on num agents vs obs shape!!
'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.target import binding

class Target(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=1080, height=720, num_agents=8,
            num_goals=8, render_mode=None, log_interval=128, size=11, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(2*(num_agents+num_goals) + 4,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
            low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        #self.single_action_space = gymnasium.spaces.Discrete(9)

        self.render_mode = render_mode
        self.num_agents = num_envs*num_agents
        self.log_interval = log_interval

        super().__init__(buf)
        #self.actions = self.actions.astype(np.float32)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_agents:(i+1)*num_agents],
                self.actions[i*num_agents:(i+1)*num_agents],
                self.rewards[i*num_agents:(i+1)*num_agents],
                self.terminals[i*num_agents:(i+1)*num_agents],
                self.truncations[i*num_agents:(i+1)*num_agents],
                seed, width=width, height=height,
                num_agents=num_agents, num_goals=num_goals)
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        #actions = (actions.astype(np.float32) - 4)/8
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 2048
    TIME = 10
    env = Target(num_envs=2048)
    actions = np.random.randint(0, 5, N)
    env.reset()

    import time
    steps = 0
    start = time.time()
    while time.time() - start < TIME:
        env.step(actions)
        steps += N

    print('Cython SPS:', steps / (time.time() - start))


