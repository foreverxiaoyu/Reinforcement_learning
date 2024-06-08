import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
import gym


# ------------------------------------- #
# 经验回放池
# ------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen=capacity)

    # 在队列中添加数据
    def add(self, state, action, reward, next_state, done):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state, done))

    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)


# ------------------------------------- #
# 策略网络
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        # 环境可以接受的动作最大值
        # 只包含一个隐含层
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        x = 2 * torch.tanh(x)  # 将数值调整到 [-1,1]
        # x = x * self.action_bound  # 缩放到 [-action_bound, action_bound]
        return x


# ------------------------------------- #
# 价值网络
# ------------------------------------- #

class QValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(QValueNet, self).__init__()
        #
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

    # 前向传播
    def forward(self, x, a):
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]
        x = self.fc1(cat)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # -->[b, 1]
        return x


# ------------------------------------- #
# 算法主体
# ------------------------------------- #

class DDPG:
    def __init__(self, n_states, n_hiddens, n_actions,
                 sigma, actor_lr, critic_lr, tau, gamma, device):
        # 策略网络--训练
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 价值网络--训练
        self.critic = QValueNet(n_states, n_hiddens, n_actions).to(device)
        # 策略网络--目标
        self.target_actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 价值网络--目标
        self.target_critic = QValueNet(n_states, n_hiddens, n_actions).to(device
                                                                          )
        # 初始化价值网络的参数，两个价值网络的参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化策略网络的参数，两个策略网络的参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 属性分配
        self.gamma = gamma  # 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差，均值设为0
        self.tau = tau  # 目标网络的软更新参数
        self.n_actions = n_actions
        self.device = device

    # 动作选择
    def take_action(self, state):
        # 维度变换 list[n_states]-->tensor[1,n_states]-->gpu
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        # 策略网络计算出当前状态下的动作价值 [1,n_states]-->[1,1]-->int
        action = self.actor(state).item()
        # 给动作添加噪声，增加搜索
        action = action + self.sigma * np.random.randn(self.n_actions)
        return action

    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    # 训练
    def update(self, transition_dict):
        # 从训练集中取出数据
        # print(f"type:{type(transition_dict)}")
        # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        transition_dict = {k: np.array(v) for k, v in transition_dict.items()}
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,next_states]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]

        # 价值目标网络获取下一时刻的每个动作价值[b,n_states]-->[b,n_actors]
        next_q_values = self.target_actor(next_states)
        # 策略目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_q_values = self.target_critic(next_states, next_q_values)
        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
        q_values = self.critic(states, actions)

        # 预测值和目标值之间的均方差损失
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 当前状态的每个动作的价值 [b, n_actions]
        actor_q_values = self.actor(states)
        # 当前状态选出的动作价值 [b,1]
        score = self.critic(states, actor_q_values)
        # 计算损失
        actor_loss = -torch.mean(score)
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络的参数
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)


# 参数定义
import argparse  # 参数设置

# 创建解释器
parser = argparse.ArgumentParser()

# 参数定义
parser.add_argument('--actor_lr', type=float, default=3e-4, help='策略网络的学习率')
parser.add_argument('--critic_lr', type=float, default=3e-3, help='价值网络的学习率')
parser.add_argument('--n_hiddens', type=int, default=64, help='隐含层神经元个数')
parser.add_argument('--gamma', type=float, default=0.98, help='折扣因子')
parser.add_argument('--tau', type=float, default=0.005, help='软更新系数')
parser.add_argument('--buffer_size', type=int, default=10000, help='经验池容量')
parser.add_argument('--min_size', type=int, default=1000, help='经验池超过buffer_size/10再训练')
parser.add_argument('--batch_size', type=int, default=64, help='每次训练64组样本')
parser.add_argument('--sigma', type=int, default=0.01, help='高斯噪声标准差')

# 参数解析
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #

# env_name = "MountainCarContinuous-v0"  # 连续型动作
env_name = 'Pendulum-v1'
env = gym.make(env_name)

n_states = env.observation_space.shape[0]  # 状态数 2
n_actions = env.action_space.shape[0]  # 动作数 1
# action_bound = env.action_space.high[0]  # 动作的最大值 1.0

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

# 经验回放池实例化
replay_buffer = ReplayBuffer(capacity=args.buffer_size)
# 模型实例化
agent = DDPG(n_states=n_states,  # 状态数
             n_hiddens=args.n_hiddens,  # 隐含层数
             n_actions=n_actions,  # 动作数
             # action_bound=action_bound,  # 动作最大值
             sigma=args.sigma,  # 高斯噪声
             actor_lr=args.actor_lr,  # 策略网络学习率
             critic_lr=args.critic_lr,  # 价值网络学习率
             tau=args.tau,  # 软更新系数
             gamma=args.gamma,  # 折扣因子
             device=device
             )

# -------------------------------------- #
# 模型训练
# -------------------------------------- #

return_list = []  # 记录每个回合的return均值
sum_return_list = []  # 记录每个回合的return总和

for i in range(50):  # 迭代10回合
    episode_return = 0  # 累计每条链上的reward
    state = env.reset()[0]  # 初始时的状态
    done = False  # 回合结束标记
    count = 0  # 计数器
    while not done and count < 200:
        # 获取当前状态对应的动作
        count += 1
        action = agent.take_action(state)
        # 环境更新
        next_state, reward, done, _, _ = env.step(action)
        # 更新经验回放池
        replay_buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计每一步的reward
        episode_return += reward
        # print(f"reward={reward}")

        # 如果经验池超过容量，开始训练
        if replay_buffer.size() > args.min_size:
            # 经验池随机采样batch_size组
            s, a, r, ns, d = replay_buffer.sample(args.batch_size)
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # 模型训练
            agent.update(transition_dict)

    # 保存每一个回合的回报
    return_list.append(episode_return)
    sum_return_list.append(episode_return)
    # 打印回合信息
    print(f'iter:{i}, return:{int(episode_return/count)}, sum_return:{episode_return}')

# 关闭动画窗格
env.close()

# -------------------------------------- #
# 绘图
# -------------------------------------- #

x_range = list(range(len(return_list)))
# 成功率
success_count = 0
# print(len(return_list))
# print([x / 200 for x in return_list])
for i in [x / 200 for x in return_list]:
    if int(i) == 0:
        success_count += 1
success_rate = success_count / len(return_list)
# print(f'Success Rate: {success_rate * 100}%')


plt.plot(x_range, [x / 200 for x in return_list])
plt.title(f"DDPG,actor_lr={args.actor_lr},critic_lr={args.critic_lr},buffer_size={args.buffer_size}")
plt.xlabel('Episode')
plt.ylabel('Return')
plt.text(0, 0.1, f"success={success_rate * 100}%")
#设置平均值
plt.axhline(y=np.mean([x / 200 for x in return_list]), color='r', linestyle='--', label=f'mean={np.mean([x / 200 for x in return_list]):.2f}')
plt.legend(loc='lower right')
plt.show()
