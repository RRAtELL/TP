def sarsa(num_episodes=500, gamma_discount=0.9, alpha=0.5, epsilon=0.1):
    """
    Implementation of SARSA algorithm. (Sutton's book)

    Args:
        num_episodes -- type(int) number of games to train agent
        gamma_discount -- type(float) discount factor determines importance of future rewards
        alpha -- type(float) determines convergence rate of the algorithm (can think as updating states fast or slow)
        epsilon -- type(float) explore/ exploit ratio (exe: default value 0.1 indicates %10 exploration)

    Returns:
        q_table -- type(np.array) Determines state value
        reward_cache -- type(list) contains cumulative_reward
    """
    # initialize all states to 0
    # Terminal state cliff_walking ends
    q_table = createQ_table()
    step_cache = list()
    reward_cache = list()

    # start iterating through the episodes

    for episode in range(0, num_episodes):

        agent = (3, 0)  # starting from left down corner

        game_end = False

        reward_cum = 0  # cumulative reward of the episode
        step_cum = 0  # keeps number of iterations untill the end of the game

        env = np.zeros((4, 12))
        env = visited_env(agent, env)

        # choose action using policy
        # episode가 시작하면서 state, action initilaize
        state, _ = get_state(agent, q_table)
        # policy에 의거한) q_table을 참고해서 # action을 출력
        action = epsilon_greedy_policy(state, q_table)

        while(game_end == False):

            # move agent to the next state
            agent = move_agent(agent, action)

            env = visited_env(agent, env)
            step_cum += 1

            # observe next state value
            next_state, _ = get_state(agent, q_table)

            # observe reward and determine whether game ends
            reward, game_end = get_reward(next_state)
            reward_cum += reward

            # choose next_action using policy and next state
            next_action = epsilon_greedy_policy(next_state, q_table)

            # update q_table
            # differs from q-learning uses the next action determined by policy
            next_state_value = q_table[next_action][next_state]
            q_table = update_qTable(
                q_table, state, action, reward, next_state_value, gamma_discount, alpha)

            # update the state and action
            state = next_state
            action = next_action  # differs q_learning both state and action must updated

        reward_cache.append(reward_cum)
        step_cache.append(step_cum)

        if(episode > 498):
            print("Agent trained with SARSA after 500 iterations")
            print(env)  # display the last 2 path agent takes

    return q_table, reward_cache, step_cache
