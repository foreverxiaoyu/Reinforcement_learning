import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random
import gym
import matplotlib.pyplot as plt


# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #

class Net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #

class DQN:
    def __init__(self, n_states, n_hidden, n_actions, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min,
                 target_update, device):
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.device = device
        self.count = 0

        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def take_action(self, state):
        state = torch.Tensor(state[np.newaxis, :])
        if np.random.random() < 1 - self.epsilon:
            actions_value = self.q_net(state)
            action = actions_value.argmax().item()
        else:
            action = np.random.randint(self.n_actions)
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1


# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #

capacity = 5000  # 经验池容量
lr = 5e-4  # 学习率
gamma = 0.98 # 折扣因子
epsilon = 0.99  # 贪心系数
epsilon_decay = 0.99  # 贪心系数衰减
epsilon_min = 0.001  # 最小贪心系数
target_update = 200  # 目标网络的参数的更新频率
batch_size = 64
n_hidden = 64  # 隐含层神经元个数
min_size = 200  # 经验池超过200后再训练
return_list = []  # 记录每个回合的回报

# 加载环境
env = gym.make("CartPole-v0")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# 实例化经验池
replay_buffer = ReplayBuffer(capacity)
# 实例化DQN
agent = DQN(n_states=n_states, n_hidden=n_hidden, n_actions=n_actions, learning_rate=lr, gamma=gamma, epsilon=epsilon,
            epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, target_update=target_update, device=device)

# 训练模型
for i in range(1000):
    state = env.reset()[0]
    episode_return = 0
    done = False
    count = 0
    while not done and count < 200:
        count += 1
        action = agent.take_action(state)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward

        if replay_buffer.size() > min_size:
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': s,
                'actions': a,
                'next_states': ns,
                'rewards': r,
                'dones': d,
            }
            agent.update(transition_dict)
        if done: break

    return_list.append(episode_return)

    # 更新epsilon
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
    print(f'Episode {i}, Return {episode_return}, Epsilon {agent.epsilon})')

# 绘图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'DQN,lr={agent.learning_rate},gamma={agent.gamma},epsilon={agent.epsilon},capacity={capacity}')


# 评价指标
returns = np.array(return_list)

# 平均奖励
average_reward = np.mean(returns)
print(f'Average Reward: {average_reward}')

# 完成率（假设200为完成任务的阈值）
success_rate = np.sum(returns >= 200) / len(returns)
print(f'Success Rate: {success_rate * 100}%')

# 训练回合数
episodes_to_convergence = np.argmax(returns >= 200) + 1 if np.any(returns >= 200) else len(returns)
print(f'Episodes to Convergence: {episodes_to_convergence}')

# 打印一些基本统计数据
print(f'Max Reward: {np.max(returns)}')
print(f'Min Reward: {np.min(returns)}')
print(f'Standard Deviation of Reward: {np.std(returns)}')

# 绘制收敛速度，放在右下角
# plt.axhline(average_reward, color='r', linestyle='--', label='Average Reward')
plt.axvline(episodes_to_convergence, color='g', linestyle='--', label=f'Episodes to Convergence: {episodes_to_convergence}')
plt.axhline(int(np.std(returns)), color='r', linestyle='--', label=f'Std:{int(np.std(returns))}')
plt.legend(loc='lower right')

plt.show()
