"""
`RL 工具 <rl_utils.html>`_ ||
**离散动作空间环境下的 PPO 算法简洁实现** ||
`RotatE-WN18RR <train_rotate_WN18RR_adv.html>`_

离散环境动作空间下的 PPO 算法简洁实现
=======================================================

这一部分介绍如何用在 离散动作空间环境（Cart Pole）中 上训练 PPO。

.. image:: /_static/images/simple-implementation/cart_pole.gif
    :align: center
    :height: 300

源教程地址：https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/

PPO 原论文：`Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`__ 。

我的论文笔记：https://cdn.jsdelivr.net/gh/LuYF-Lemon-love/susu-rl-papers/papers/01-PPO.pdf

Cart Pole：https://gymnasium.farama.org/environments/classic_control/cart_pole/

PPO 是 TRPO 算法的改进版，实现更加简洁，而且更快。PPO 的优化目标与 TRPO 相同，但 PPO 用了一些相对简单的方法来求解。

PPO 使用截断的方法在目标函数中进行限制，以保证新的参数和旧的参数的差距不会太大，即：

.. image:: /_static/images/simple-implementation/ppo-loss.svg
    :align: center
    :height: 300


:math:`clip(x,l,r)=max(min(x,r),l)`，即把 :math:`x` 限制在 :math:`[l,r]` 内。

导入第三方库
-----------------

"""

import os
import torch
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils

###############################################################################
# ------------------------------
#

###############################################################################
# 定义策略网络和价值网络
# ------------------------------

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

###############################################################################
# ------------------------------
#

###############################################################################
# 定义 PPO 算法
# ------------------------------
# PPO算法,采用截断方式

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1,-1).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

###############################################################################
# ------------------------------
#

###############################################################################
# 模型训练
# ------------------------------
# 在车杆环境中训练 PPO 算法

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='rgb_array')
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

print(return_list[-1])

episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, return_list, color='pink', label='raw')
plt.plot(episodes_list, mv_return, color='green', label='moving_average')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.legend()
plt.savefig('./docs/_static/images/simple-implementation/ppo-discrete-returns.jpg')

###############################################################################
# .. figure:: /_static/images/simple-implementation/ppo-discrete-returns.jpg
#      :align: center
#      :height: 300
#
#      PPO 算法在训练过程中获得奖励

###############################################################################
# --------------
#

###############################################################################
# PPO playground
# ------------------

temp_path = "./temp"
if not os.path.exists(temp_path):
    os.makedirs(temp_path, exist_ok=True)

with open(os.path.join(temp_path, 'ppo.pickle'), 'wb') as f:
    pickle.dump(agent, f)

with open(os.path.join(temp_path, 'ppo.pickle'), 'rb') as f:
    best_agent = pickle.load(f)

state, info = env.reset()

frames = []

for step in range(500):
    action = best_agent.take_action(state)
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    img = env.render()
    frames.append(img)
    
anim = rl_utils.plot_animation(frames)
anim.save('./docs/_static/images/simple-implementation/play-ppo-discrete.gif', writer='pillow')

###############################################################################
# .. figure:: /_static/images/simple-implementation/play-ppo-discrete.gif
#      :align: center
#      :height: 300
#
#      PPO 智能体的表现