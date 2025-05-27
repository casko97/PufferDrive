import functools

import pufferlib

from mettagrid.mettagrid_env import MettaGridEnv

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='pufferlib/environments/metta/metta.yaml', render_mode='auto', buf=None, seed=0):
    '''Crafter creation function'''
    #return MettaPuff(config, render_mode, buf)
    import mettagrid.mettagrid_env
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("div", oc_divide, replace=True)
    cfg = OmegaConf.load(config)
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
        super().__init__(config, render_mode=render_mode, buf=buf)

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
