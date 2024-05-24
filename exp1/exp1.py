import matplotlib.pyplot as plt
import numpy as np

import gym


class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])  # Q-learning 用的是在下一个状态中Q值最大的动作的Q值
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q

    # 把 Q表格 的数据保存到文件中
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到 Q表格
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


class SarsaAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索

    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 下一个动作a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa 中用的是下一个状态和动作的Q值
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q

    # 把 Q表格 的数据保存到文件中
    def save(self):
        npy_file = './sarsa_q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到 Q表格
    def restore(self, npy_file='./sarsa_q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


# train.py

def run_episode_QLearning(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = int(obs)
    while True:
        action = agent.sample(obs)  # 根据算法选择一个动作
        print('obs=', obs, 'action=', action)
        next_obs, reward, done, _, _ = env.step(action)  # 与环境进行一个交互
        print('obs=', obs, 'action=', action, 'next_obs=', next_obs, 'reward=', reward)
        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  # 渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def run_episode_sarsa(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = int(obs)

    action = agent.sample(obs)  # 初始动作

    while True:
        next_obs, reward, done, _, _ = env.step(action)  # 与环境进行一个交互
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        next_obs = int(next_obs)

        next_action = agent.sample(next_obs)  # 下一个动作

        # 训练 Sarsa算法
        print('obs=', obs, 'action=', action, 'next_obs=', next_obs, 'reward=', reward)
        agent.learn(obs, action, reward, next_obs, next_action, done)

        obs = next_obs  # 存储上一个观察值
        action = next_action  # 存储上一个动作
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  # 渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = int(obs)
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        # time.sleep(0.5)
        # env.render()
        if done:
            break
    return total_reward

# 计算收敛速度和稳定性
def calculate_convergence_speed(rewards, threshold, window_size):
    for i in range(window_size, len(rewards)):
        if np.std(rewards[i-window_size:i]) < threshold:
            return i
    return len(rewards)

def calculate_stability(rewards, window_size):
    recent_rewards = rewards[-window_size:]
    mean_reward = np.mean(recent_rewards)
    std_reward = np.std(recent_rewards)
    mae_reward = np.mean(np.abs(recent_rewards - mean_reward))
    return mean_reward, std_reward, mae_reward

# 使用gym创建悬崖环境
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

# 创建一个agent实例，输入超参数
agent = QLearningAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    learning_rate=0.1,
    gamma=0.1,
    e_greed=0.1)

agent_sarsa = SarsaAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    learning_rate=0.1,
    gamma=0.1,
    e_greed=0.1)


method = input('请输入算法：1 (QLearning), 2 (Sarsa)\n')
# 训练500个episode，打印每个episode的分数
# 设定参数
# 最后100个回合的回报值标准差小于60
num_episodes = 500
convergence_threshold = 60
window_size = 100

# 初始化
rewards_QLearning = []
rewards_Sarsa = []

reword_list = []
for episode in range(500):
    if method == '1':
        ep_reward, ep_steps = run_episode_QLearning(env, agent, False)
        rewards_QLearning.append(ep_reward)
    else:
        ep_reward, ep_steps = run_episode_sarsa(env, agent_sarsa, False)
        rewards_Sarsa.append(ep_reward)
    # ep_reward, ep_steps = run_episode_QLearning(env, agent, False)
    # ep_reward, ep_steps = run_episode_sarsa(env, agent, False)
    reword_list.append(ep_reward)
    print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
if method == '1':
    covergence_episode_QLearning = calculate_convergence_speed(rewards_QLearning, convergence_threshold, window_size) # 收敛速度, 100个episode的标准差小于60的最后一个episode
    mean_reward_QLearning, std_reward_QLearning, mae_reward_QLearning = calculate_stability(rewards_QLearning, window_size) # 平均值和标准差以及平均绝对误差
    print(f'Q-learning: convergence episode = {covergence_episode_QLearning}, mean reward = {mean_reward_QLearning}, std reward = {std_reward_QLearning}, mae reward = {mae_reward_QLearning}')
else:
    covergence_episode_Sarsa = calculate_convergence_speed(rewards_Sarsa, convergence_threshold, window_size)
    mean_reward_Sarsa, std_reward_Sarsa, mae_reward_Sarsa = calculate_stability(rewards_Sarsa, window_size)
    print(f'Sarsa: convergence episode = {covergence_episode_Sarsa}, mean reward = {mean_reward_Sarsa}, std reward = {std_reward_Sarsa}, mae reward = {mae_reward_Sarsa}')

# 全部训练结束，查看算法效果
# if method == '1':
#     test_reward = test_episode(env, agent) #
# else:
#     test_reward = test_episode(env, agent_sarsa) #
# # test_reward = test_episode(env, agent)
#
# print('test reward = %.1f' % (test_reward))
plt.plot(reword_list)
if method == '1':
    plt.title(f'Q-learning, lr={agent.lr}, gamma={agent.gamma}, e_greed={agent.epsilon}')
else:
    plt.title(f'Sarsa, lr={agent_sarsa.lr}, gamma={agent_sarsa.gamma}, e_greed={agent_sarsa.epsilon}')
# plt.title(f', lr={agent.lr}, gamma={agent.gamma}, e_greed={agent.epsilon}')
plt.xlabel('Episode')
#调整y轴的刻度范围
# plt.ylim(-30, 0)
plt.ylabel('Reward')
test_reward = -13
plt.axhline(test_reward, color='r', linestyle='--', label=f'end_y={test_reward}')
if method == '1':
    plt.axhline(mean_reward_QLearning, color='r', linestyle='--', label=f'last100mean_y={mean_reward_QLearning}')
    plt.axvline(covergence_episode_QLearning, color='g', linestyle='--', label=f'covergence_x={covergence_episode_QLearning}')
else:
    plt.axhline(mean_reward_Sarsa, color='r', linestyle='--', label=f'last100mean_y={mean_reward_Sarsa}')
    plt.axvline(covergence_episode_Sarsa, color='g', linestyle='--', label=f'covergence_x={covergence_episode_Sarsa}')
# plt.axhline(mean_reward_Sarsa, color='r', linestyle='--', label=f'last100mean_y={mean_reward_Sarsa}')
# plt.axvline(covergence_episode_QLearning, color='g', linestyle='--', label=f'covergence_x={covergence_episode_QLearning}')
plt.legend()
plt.show()
# file_path = './lr0.05,ga0.9,gr0.1/1_1.png'
# plt.savefig(file_path, dpi=300)

