import retro
import os
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class ProcessExcitebikeFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessExcitebikeFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(90, 90, 3), dtype=np.uint8)

    def observation(self, obs):
        return ProcessExcitebikeFrame.process(obs)

    @staticmethod
    def process(frame):
        if frame.size != 224 * 240 * 3:
            assert False, "Unknown resolution."
        #crop out the top and left portions as they are unnecessary
        #crop out bit of the right as it shouldn't make too much of an impact on training
        crop_obs = frame[31:211,45:225]

        #colored out the time on bottom to black
        crop_obs[160:180,0:56] = np.array([0,0,0])
        crop_obs[160:180,95:] = np.array([0,0,0])

        #resize from 180x180 to 90x90
        resized = cv2.resize(crop_obs, (90,90),interpolation = cv2.INTER_LINEAR)
        return resized

def main():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integration")
    )
    print("Excitebike-NES-Track-1" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("Excitebike-NES-Track-1", inttype=retro.data.Integrations.ALL)
    env = ProcessExcitebikeFrame(env)
    #print(env)

    obs = env.reset()
    print(obs.shape)

    '''
    crop_obs = obs[31:211,45:225]
    crop_obs[160:180,0:56] = np.array([0,0,0])
    crop_obs[160:180,95:] = np.array([0,0,0])

    resized = cv2.resize(crop_obs, (90,90),interpolation = cv2.INTER_LINEAR)
    print(resized.shape)
    '''

    plt.imshow(obs)
    plt.show()

    #plt.imshow(obs)
    #plt.show()

    env.close()


if __name__ == "__main__":
    main()
