import retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integration")
    )
    print("Excitebike-NES-Track-1" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("Excitebike-NES-Track-1", inttype=retro.data.Integrations.ALL, record=".")
    print('SonicDiscretizer action_space', env.action_space)
    #print(env)

    obs = env.reset()
    for i in range(200):
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        #print(act)
        #env.render()
        #print(rew)

        if done:
            obs = env.reset()
    env.close()



if __name__ == "__main__":
    main()
