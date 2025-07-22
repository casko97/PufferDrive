import functools
import numpy as np
import gymnasium

import pufferlib

from omegaconf import OmegaConf
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.replay_writer import ReplayWriter

#from mettagrid.mettagrid_env import MettaGridEnv
#from mettagrid.curriculum import SingleTaskCurriculum

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='pufferlib/environments/metta/metta.yaml', render_mode='auto', buf=None, seed=0, **kwargs):
    '''Metta creation function'''
    # Debug: print all kwargs to see what's being passed
    print(f"DEBUG: make() called with kwargs: {kwargs}")
    
    # Extract expected parameters with defaults
    ore_reward = kwargs.pop('ore_reward', 0.25)
    heart_reward = kwargs.pop('heart_reward', 0.5)
    battery_reward = kwargs.pop('battery_reward', 0.25)
    
    # Check if there are any unexpected kwargs
    if kwargs:
        print(f"WARNING: Unexpected kwargs passed to make(): {kwargs}")
    
    OmegaConf.register_new_resolver("div", oc_divide, replace=True)
    cfg = OmegaConf.load(config)
    
    # Update rewards under the new structure: agent.rewards.inventory
    inventory_rewards = cfg['game']['agent']['rewards']['inventory']
    inventory_rewards['ore_red'] = float(ore_reward)
    inventory_rewards['heart'] = float(heart_reward)
    inventory_rewards['battery_red'] = float(battery_reward)
    
    cfg = SingleTaskCurriculum('puffer', cfg)
    return MettaPuff(cfg, render_mode=render_mode, buf=buf)

def oc_divide(a, b):
    """
    Divide a by b, returning an int if both inputs are ints and result is a whole number,
    otherwise return a float.
    """
    result = a / b
    # If both inputs are integers and the result is a whole number, return as int
    if isinstance(a, int) and isinstance(b, int) and result.is_integer():
        return int(result)
    return result

class MettaPuff(MettaGridEnv):
    def __init__(self, config, render_mode='human', buf=None, seed=0):
        self.replay_writer = None
        #if render_mode == 'auto':
        #    self.replay_writer = ReplayWriter("metta/")

        super().__init__(config, render_mode=render_mode, buf=buf, replay_writer=self.replay_writer)
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.actions = self.actions.astype(np.int32)

    @property
    def single_action_space(self):
        return gymnasium.spaces.MultiDiscrete(super().single_action_space.nvec, dtype=np.int32)

    def step(self, actions):
        obs, rew, term, trunc, info = super().step(actions)

        if all(term) or all(trunc):
            self.reset()
            if 'agent_raw' in info:
                del info['agent_raw']
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']

        else:
            info = []

        return obs, rew, term, trunc, [info]
