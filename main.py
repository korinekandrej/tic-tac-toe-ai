import tic_tac_toe_env as ttte
import tic_tac_toe_agent as ttta


if __name__ == "__main__":
    ########################################################################################################################

    #Change the number of episodes:
    episodes = 1000

    #Comment/uncomment different environments: (render_mode = "human" works only for dimension = 2)
    env = ttte.TicTacToeEnv(size=3, dimension=2, connect_n=3)           #for speed (with more episodes)
    #env = ttte.TicTacToeEnv(size=3, render_mode="human")               #for visual testing with a window (with less episodes)

    #env = ttte.TicTacToeEnv(size=2, dimension=3) #The env is setup to work with any dimension, but there are bugs that crash the game :(

    #Change parameters of the agent
    agent = ttta.TicTacToeAgent(alpha=0.1, epsilon=0.1, gamma=0.9) #Learning, Exploration, Discount

    ########################################################################################################################
    

    total_reward = 0
    for episode in range(episodes):
        print(f"Episode started")
              
        observation, _ = env.reset()
        done = False
        episodic_reward = 0

        while not done:
            action = agent.choose_action(observation)

            next_observation, reward, done, _, _ = env.step(action)
            agent.learn(observation, action, reward, next_observation, done)

            observation = next_observation

            episodic_reward += reward

        print(f"Episode: {episode + 1} Episodic reward: {episodic_reward}")
        total_reward += episodic_reward

    print(f"Average reward: {total_reward / episodes}")
    print(f"Total reward: {total_reward}")