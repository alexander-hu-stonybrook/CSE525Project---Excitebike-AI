import retro
import os
import retro.scripts.playback_movie as pbm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integration")
    )
    #print("Excitebike-Track-1" in retro.data.list_games(inttype=retro.data.Integrations.ALL))

    '''
    print("Before loading movie")
    movie = retro.Movie('Test_Runs/Test_Run_1.bk2')
    movie.step()
    print("After loading movie")
    '''

    dir = os.fsencode('./Test_Runs')

    player_data = []
    unique_keys = set()

    for file in os.listdir(dir):
        fname = os.fsdecode(file)

        print("Now running: " + fname)

        movie = retro.Movie('./Test_Runs/'+fname)
        movie.step()

        env = retro.make("Excitebike-Track-1",
            inttype=retro.data.Integrations.ALL,
            state=None,
            # bk2s can contain any button presses, so allow everything
            use_restricted_actions=retro.Actions.ALL,
            players=movie.players,
        )

        env.initial_state = movie.get_state()
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

            if rew != 5000: #manually assigning reward values because gym-retro didn't play nice
                rew = -1 #assign a penalty for every time step agent takes to finish the race

            player_data.append((cur_state,keys,rew,new_state,done))
            total_rew += rew
            #if rew == 10000:
                #print(rew)
                #print(done)
            if done:
                print("Finished race")
                # Print data
                #print("observation: ", obs, ", observation space: ", env.observation_space) # Note that the observation is four floats that likely represent the "cart" position and the angle/velocity of pendulum
                # This means that our observations can be between -inf and inf, there are 4 of them, all of type float32
                #print("reward: ", rew, ", reward range: ", env.reward_range)
                #print("info: ", info)
                #print("action space: ", env.action_space) # Can interpret this as take one action between -1 and 1, of type float32
                env.reset()
                break #remove this later rn just for testing

        print(total_rew)
        env.close()

        #emulator, movie, duration = pbm.load_movie('Test_Run_1.bk2')
        #print(emulator)

        #pbm.playback_movie(env,movie,video_file="test.mp4")

    print(len(player_data))
    print(len(unique_keys))

    for k in unique_keys:
        print(k)


if __name__ == "__main__":
    main()
