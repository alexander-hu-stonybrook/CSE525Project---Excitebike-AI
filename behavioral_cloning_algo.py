import gym

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

import math
import glob
import io
import base64
import time
import pdb
import cv2
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import retro

from collections import namedtuple, deque
import collections
from itertools import count

# Colab comes with PyTorch
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

###################
# Neural Network #
##################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_Action_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        if input_dim[1] != 90:
            raise ValueError(f"Expecting input height: 90, got: {input_dim[1]}")
        if input_dim[2] != 90:
            raise ValueError(f"Expecting input width: 90, got: {input_dim[2]}")

        #print("CNN_Action_Model inputs/outputs")
        #print(input_dim)
        #print(output_dim)

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.net2 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input):
        #print(input)
        feature_v = self.net1(input)
        #print("Feature vector shape:")
        #print(feature_v.shape)
        return self.net2(feature_v)

#################
# Replay Buffer #
#################

SingleExperience = namedtuple('SingleExperience',
                              field_names = ['state','action','reward','done','nextstate'])

# Now let us define the Replay Buffer
# More info about deque here : https://pythontic.com/containers/deque/introduction
class ReplayBuffer:
    def __init__(self,size):
        self.buffer = deque(maxlen = size)
    def sampleBuf(self,size):
        # First let us get a list of elements to sample
        # Make sure to choose replace = False so that we cannot choose the same
        # sample tuples multiple times
        el_inds = np.random.choice(len(self.buffer), size, replace=False)

        # A nifty piece of code implemented by @Jordi Torres, this makes use of the
        # zip function to combine each respective data field, and returns the np arrays
        # of each separate entity.
        arr_chosen_samples = [self.buffer[i] for i in el_inds]
        # Take the samples and break them into their respective iterables
        state_arr, actions_arr, reward_arr, done_arr, next_state_arr = unzip(arr_chosen_samples)
        # Return these iteratables as np arrays of the correct types
        # Action will be returned as a straight array due to its input being an array of booleans
        return np.array(state_arr),actions_arr,np.array(reward_arr,dtype=np.float32),done_arr,np.array(next_state_arr)

    def append(self, sample):
        self.buffer.append(sample)

    def size(self):
        return len(self.buffer)

##########
# Agents #
##########

#Need to make adjustments to account for actual environment parameters

class Agent:
    def __init__(self, buffer, environment):
        # Set object variables
        self.env = environment
        self.replay_buffer = buffer
        self.restart_episode()

    # Restarts environment, and resets all necessary variables
    def restart_episode(self):
        self.state = self.env.reset()
        self.total_reward = float(0)

    # Define epsilon greedy policy
    def choose_epsilon_greedy_action(self,model,epsilon,act_space,device="cpu"):
        if random.uniform(0,1) > epsilon:
            state_numpy_array = np.array([self.state], copy=False)
            state_tensor = torch.tensor(state_numpy_array).float().to(device)
            model_estimated_action_values = model(state_tensor)
            #print("model estimate action value shape")
            #print(model_estimated_action_values.shape)
            act_v = torch.argmax(model_estimated_action_values, dim=1) # This is the same as torch.argmax
            action = int(act_v.item())
        else:
            action = random.randrange(len(act_space))
        return action

    # Function that interacts one step with the environment, adds that sample
    # to the replay buffer, and returns the final cumulative reward or None if
    # episode hasn't terminated

    # We pass in the epsilon for choosing random actions
    # Let's also define the device for standard practice, and so that we can use
    # GPU training effectively.
    def advance_state_and_log(self, model, epsilon, act_space, buf_append=False, device="cpu"):
        # First, let us choose an action to take using epsilon-greedy
        act_ind = self.choose_epsilon_greedy_action(model,epsilon,act_space,device)
        action = act_space[act_ind]

        # Now that we have chosen an action, take a step and increment total reward count
        observation, reward, done, info = self.env.step(action)
        if reward != 5000:
            reward = -1
        self.total_reward += reward

        # Now that we have the output, we must append a value to our replay buffer
        # First we must actually create a tuple according to the object we specified
        # Remember: field_names = ['state','action','reward','done','nextstate'])
        sample_tuple = SingleExperience(self.state, act_ind, reward, done, observation)
        # Add to our buffer
        if buf_append:
            self.replay_buffer.append(sample_tuple)

        # Update our current state
        self.state = observation

        if done:
            t_reward = self.total_reward
            self.restart_episode()
            return (True, t_reward)
        return (False, None)

################################
# Wrappers and Other Functions #
################################

class ProcessExcitebikeFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessExcitebikeFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 90, 90), dtype=np.uint8)

    def observation(self, obs):
        return ProcessExcitebikeFrame.process(obs)

    @staticmethod
    def process(frame):
        #check if right dimensions
        if frame.size == 224 * 240 * 3:
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        #crop out the top and left portions as they are unnecessary
        #crop out bit of the right as it shouldn't make too much of an impact on training
        crop_obs = img[31:211,45:225]

        #colored out the time on bottom to black
        crop_obs[160:180,0:56] = np.array([0,0,0],dtype=np.float32)
        crop_obs[160:180,95:] = np.array([0,0,0],dtype=np.float32)

        #convert to greyscale
        crop_obs = crop_obs[:, :, 0] * 0.299 + crop_obs[:, :, 1] * 0.587 + crop_obs[:, :, 2] * 0.114

        #resize from 180x180 to 90x90, also eliminating the 1 channel at the end
        resized = cv2.resize(crop_obs, (90,90),interpolation = cv2.INTER_AREA)
        resized = np.reshape(resized, [90, 90*1])

        #add a dimension in the front for the sake of nn processing
        resized = np.expand_dims(resized, axis=0)
        return resized.astype(np.uint8)

# Unzip iterable function definition
def unzip(iterable):
    return zip(*iterable)

##############
# Algorithm #
#############

# Hyperparameter init
#environment_name = 'PongNoFrameskip-v4'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integration")
)

# Set up replay buffer parameters
buffer_size = 100000
batch_size = 500

# Define learning rate and other learning parameters
lr = 0.0001
gamma = 0.9
sync_target_frequency = 1000
max_steps = 400000

# Epsilon parameters
'''
epsilon = 1
decay = 0.999935
min_epsilon = 0.01
'''
epsilon = 0.02
decay = 1
min_epsilon = 0.02

# Initialize the buffer
player_data = ReplayBuffer(buffer_size)

dir = os.fsencode('./Test_Runs')

#player_data = []
unique_keys = set()

for file in os.listdir(dir):
    fname = os.fsdecode(file)

    print("Now running: " + fname)

    movie = retro.Movie('./Test_Runs/'+fname)
    movie.step()

    env = retro.make("Excitebike-NES-Track-1",
        inttype=retro.data.Integrations.ALL,
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )

    env.initial_state = movie.get_state()
    env = ProcessExcitebikeFrame(env)
    cur_state = env.reset()

    #print("Made it past env")

    total_rew = 0
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        #print(keys)
        keys = tuple(keys)
        unique_keys.add(keys)
        new_state, rew, done, info = env.step(keys)

        #print(new_state.shape)
        '''
        for x in new_state:
            for y in x:
                for z in y:
                    if z < 0 or z > 255:
                        print(z)
        '''

        '''
        if done == 0:
            bool_done = False
        else:
            bool_done = True
        '''

        if rew != 5000: #manually assigning reward values because gym-retro didn't play nice
            rew = -1 #assign a penalty for every time step agent takes to finish the race

        #if done:
            #plt.imshow(np.squeeze(new_state,axis=0))
            #plt.show()
            #new_state = env.reset()

        #print(new_state)

        sample_data = SingleExperience(cur_state,keys,rew,done,new_state)
        player_data.append(sample_data)
        total_rew += rew
        #if rew == 10000:
            #print(rew)
            #print(done)
        cur_state = new_state
        if done:
            print("Finished race")
            print(done)
            # Print data
            #print("observation: ", obs, ", observation space: ", env.observation_space) # Note that the observation is four floats that likely represent the "cart" position and the angle/velocity of pendulum
            # This means that our observations can be between -inf and inf, there are 4 of them, all of type float32
            #print("reward: ", rew, ", reward range: ", env.reward_range)
            #print("info: ", info)
            #print("action space: ", env.action_space) # Can interpret this as take one action between -1 and 1, of type float32
            #env.reset()
            break #remove this later rn just for testing

    print(total_rew)
    env.close()

    #emulator, movie, duration = pbm.load_movie('Test_Run_1.bk2')
    #print(emulator)

    #pbm.playback_movie(env,movie,video_file="test.mp4")

#print(player_data.size())
#print(len(unique_keys))

#for k in unique_keys:
    #print(k)

unique_keys = list(unique_keys)
#print(unique_keys)
#print(len(unique_keys))


env = retro.make("Excitebike-NES-Track-1", inttype=retro.data.Integrations.ALL)
env = ProcessExcitebikeFrame(env)
#print(env)
obs = env.reset()

#plt.imshow(obs)
#plt.show()

#print(obs.shape)
#print(obs)


agent = Agent(player_data, env)


#test_obs = torch.unsqueeze(state_numpy_array, 0)

observation_shapes = env.observation_space.shape

# Initialize networks
# Change num_actions to the action array formed by behavioral data
behavior_model = CNN_Action_Model(observation_shapes, len(unique_keys)).to(device)
target_model = CNN_Action_Model(observation_shapes, len(unique_keys)).to(device)

# Define optimizer as Adam, with our learning rate
optimizer = optim.Adam(behavior_model.parameters(), lr=lr)

# Let's log the returns, and step in episode
return_save = []
loss_save = []
episode_num = 0

for step in range(max_steps):

    # Take a step, and record whether or not we finished an episode, and if so,
    # the total reward
    if step % 1000 == 0:
        print("Reached step " + str(step))

    ep_done, tot_reward = agent.advance_state_and_log(behavior_model,epsilon,unique_keys,device=device)

    if ep_done:
        return_save.append(tot_reward)
        episode_num += 1
        print(f"Episode {episode_num}, Step {step}, Total Reward: {tot_reward}")

    # Implement early stopping
    if np.average(return_save[-5:]) >= 1000:
        break


    # Decay epsilon
    epsilon *= decay
    if epsilon < min_epsilon:
        epsilon = min_epsilon

    # Sync target network
    if step % sync_target_frequency == 0:
        # Copy weights to target network
        target_model.load_state_dict(behavior_model.state_dict())

    # Train DQN

    # if we have enough data in buffer
    if player_data.size() > 2*batch_size:

        # First, sample from the replay buffer
        cur_state_arr, action_arr, reward_arr, done_arr, next_state_arr = player_data.sampleBuf(batch_size)
        aind_arr = []
        for a in action_arr:
            aind_arr.append(unique_keys.index(a))

        #print(next_state_arr)

        #print(done_arr)

        #print(len(action_arr))
        #print(action_arr)

        # Copy the arrays to the GPU as tensors.
        # This allows the GPU to be used for computation speed improvements.
        # Follow guide https://discuss.pytorch.org/t/converting-numpy-array-to-tensor-on-gpu/19423/3
        # and
        # https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
        # and originally
        # https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/tree/master/Chapter04
        cur_state_tensor = torch.tensor(cur_state_arr).float().to(device)
        action_tensor = torch.tensor(aind_arr).to(device)
        reward_tensor = torch.tensor(reward_arr).to(device)
        done_tensor_mask = torch.ByteTensor(done_arr).to(device)
        next_state_tensor = torch.tensor(next_state_arr).float().to(device)

        #print(action_tensor)

        #print(next_state_tensor)
        #print(done_tensor_mask)

        #print(cur_state_tensor.shape)
        #print(next_state_tensor.shape)

        # Now that we have the tensorized versions of our separated batch data, we
        # must pass the cur_states into the behavior model to get the values of the
        # taken actions.

        # First pass the batch into the model
        beh_model_output_cur_state = behavior_model(cur_state_tensor)

        #print(beh_model_output_cur_state)

        # Now we must process this tensor and extract the Q-values for taken actions.
        # This is done with a pretty magical command that was constructed by
        # Maxim Lapan, source in the resources section of hw document

        #print(beh_model_output_cur_state.shape)
        #print(action_tensor.shape)
        estimated_taken_action_vals = beh_model_output_cur_state.gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)
        # Note that this should return a 1d tensor of action values taken

        # Now we must calculate the target value

        #print(next_state_tensor)

        # First we must calculate the predicted value of taking the max action at the
        # next state. This is because we are following the equation of format:
        # Value of (state,action) = reward + discount * max(Q(s',a'))
        # with the last term calculated from the target network
        max_next_action_value_tensor = target_model(next_state_tensor).max(1)[0]
        # Note that max(1)[0] gets the maximum value in each batch sample, hence
        # getting the max from dimension 1. [0] just extracts the value.

        #print(max_next_action_value_tensor.shape)
        #print(max_next_action_value_tensor)

        # Now we must mask the done values such that reward is 0.
        max_next_action_value_tensor[done_tensor_mask] = float(0)

        target_values = max_next_action_value_tensor.detach() * gamma + reward_tensor

        # Calculate loss
        loss = nn.MSELoss()(estimated_taken_action_vals, target_values)

        if step % 1000 == 0:
            loss_save.append(loss.item())

        # Perform back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
