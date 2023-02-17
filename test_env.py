import retro
import os
import numpy as np
import gym

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class SonicDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[['A'], ['B'], ['A', 'UP'], ['A', 'DOWN'], ['A', 'RIGHT'], ['B','UP'], ['B','DOWN'], ['B','RIGHT']])

def main():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integration")
    )
    print("Excitebike-NES-Track-1" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("Excitebike-NES-Track-1", inttype=retro.data.Integrations.ALL, record=".")
    env = SonicDiscretizer(env)
    print('SonicDiscretizer action_space', env.action_space)
    #print(env)

    obs = env.reset()
    for i in range(200):
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        print(act)
        #env.render()
        #print(rew)

        if done:
            obs = env.reset()
    env.close()



if __name__ == "__main__":
    main()
