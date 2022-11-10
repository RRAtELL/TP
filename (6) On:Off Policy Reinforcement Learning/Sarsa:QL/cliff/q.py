def qlearning(num_episodes=500, gamma_discount=0.9, alpha=0.5, epsilon=0.1):
    """
    Implementation of q-learning algorithm. (Sutton's book)

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
    reward_cache = list()
    step_cache = list()

    q_table = createQ_table()

    agent = (3, 0)  # starting from left down corner

    # start iterating through the episodes

    for episode in range(0, num_episodes):

        env = np.zeros((4, 12))
        env = visited_env(agent, env)

        agent = (3, 0)  # starting from left down corner
        game_end = False

        reward_cum = 0  # cumulative reward of the episode
        step_cum = 0  # keeps number of iterations untill the end of the game

        while(game_end == False):

            # get the state from agent's position
            state, _ = get_state(agent, q_table)  # state값만 받아옴

            # choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(state, q_table)  # action select

            # move agent to the next state
            agent = move_agent(agent, action)  # agent 이동
            step_cum += 1
            env = visited_env(agent, env)  # mark the visited path

            # observe next state value
            # action에 관계 없이, 가장 큰 value만 찾아냄 --> action에는 관심이 X
            next_state, max_next_state_value = get_state(agent, q_table)

            # observe reward and determine whether game ends
            reward, game_end = get_reward(next_state)
            reward_cum += reward

            # update q_table
            q_table = update_qTable(
                q_table, state, action, reward, max_next_state_value, gamma_discount, alpha)

            # update the state
            state = next_state

        reward_cache.append(reward_cum)

        if(episode > 498):
            print("Agent trained with Q-learning after 500 iterations")
            print(env)  # display the last 2 path agent takes
        step_cache.append(step_cum)

    return q_table, reward_cache, step_cache

# state 를 인식하고, action을 취했을 때의 value를 비교해서 action 하나를 선택
# 선택된 action에 따라 변화된 state가 있을텐데, 그 state에서 value들을 비교해서 가장 큰 값을 선택함
# next state의 value로 업데이트 하는 과정의 의미는, 현재 state에서 그 state로 이동하면,
# 이만큼의 value를 얻을 수 있으니 그 결과값을 반영해서 현재 state의 value를 update하자 라는 의미
# action에 대한 update는 따로 없고 새로운 state에서의 value들을 한번 보고 결정
