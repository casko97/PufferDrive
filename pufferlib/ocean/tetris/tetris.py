import gymnasium
import numpy as np
import pufferlib
from pufferlib.ocean.tetris import binding

class Tetris(pufferlib.PufferEnv):
    def __init__(
        self, 
        num_envs=1, 
        n_cols=10, 
        n_rows=10,
        deck_size=3,
        render_mode=None, 
        log_interval=32,
        buf=None, 
        seed=0
    ):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(1 + 7 * deck_size + 4 * n_cols + n_cols*n_rows,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(4 * n_cols)
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.num_agents = num_envs

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)
        self.deck_size = deck_size
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            n_cols=n_cols,
            n_rows=n_rows,
            deck_size=deck_size,
        )
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    TIME = 5
    num_envs = 512
    env = Tetris(num_envs=num_envs)
    actions = [
        [env.single_action_space.sample() for _ in range(num_envs) ]for _ in range(1000)
    ]
    obs, _ = env.reset(seed = np.random.randint(0,1000))

    import time
    start = time.time()
    tick = 0
    
    while time.time() - start < TIME:
        action = actions[tick%1000]
        obs, _, _, _, _ = env.step(action)
        tick += 1
        # env.render()
        # time.sleep(10)

    print('SPS:', (tick*num_envs) / (time.time() - start))

