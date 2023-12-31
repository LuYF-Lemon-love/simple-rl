"""
**PPO-Discrete** ||
`SimplE-WN18RR <train_simple_WN18RR.html>`_ ||
`RotatE-WN18RR <train_rotate_WN18RR_adv.html>`_

PPO-Discrete
===================
这一部分介绍如何用在 离散环境（Cart Pole）中 上训练 PPO。

PPO 原论文：`Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`__ 。

Cart Pole：https://gymnasium.farama.org/environments/classic_control/cart_pole/

导入第三方库
-----------------

"""

import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from gym.spaces import Box
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic

######################################################################
# --------------
#

################################
# 定义超参数
# ------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v1')
    # 停止条件，用于 stop_fn
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--training-num', type=int, default=20)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args

args=get_args()

######################################################################
# --------------
#

################################
# 设置矢量化环境
# ------------------
# 首先获得环境观测空间和动作空间的形状，作为神经网络的超参数，然后利用 :py:class:`tianshou.env.SubprocVectorEnv` 构建矢量化环境。:py:class:`tianshou.env.SubprocVectorEnv` 使用 Python 多进程进行并发执行。

env = gym.make(args.task)
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
print(f"agrs.state_shape: {args.state_shape}")
print(f"args.action_shape: {args.action_shape}")

if args.reward_threshold is None:
    default_reward_threshold = {"CartPole-v1": 495}
    args.reward_threshold = default_reward_threshold.get(
        args.task, env.spec.reward_threshold
    )
train_envs = SubprocVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.training_num)]
)
test_envs = SubprocVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.test_num)]
)
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)

######################################################################
# --------------
#

################################
# 构建神经网络
# ------------------
# 利用 :py:class:`tianshou.utils.net.common.Net`, :py:class:`tianshou.utils.net.discrete.Actor`, :py:class:`tianshou.utils.net.discrete.Critic`, :py:class:`tianshou.utils.net.common.DataParallelNet`, :py:class:`tianshou.utils.net.common.ActorCritic` 构建神经网络。

net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
if torch.cuda.is_available():
    actor = DataParallelNet(
        Actor(net, args.action_shape, device=None).to(args.device)
    )
    critic = DataParallelNet(Critic(net, device=None).to(args.device))
else:
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
actor_critic = ActorCritic(actor, critic)
# orthogonal initialization
for m in actor_critic.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)
optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

######################################################################
# --------------
#

################################
# 初始化策略
# ------------------
# 我们使用上述代码中定义的网络和优化器，以及其他超参数，来定义一个 :py:class:`tianshou.policy.PPOPolicy` 策略。

dist = torch.distributions.Categorical
policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist,
    action_scaling=isinstance(env.action_space, Box),
    discount_factor=args.gamma,
    max_grad_norm=args.max_grad_norm,
    eps_clip=args.eps_clip,
    vf_coef=args.vf_coef,
    ent_coef=args.ent_coef,
    gae_lambda=args.gae_lambda,
    reward_normalization=args.rew_norm,
    dual_clip=args.dual_clip,
    value_clip=args.value_clip,
    action_space=env.action_space,
    deterministic_eval=True,
    advantage_normalization=args.norm_adv,
    recompute_advantage=args.recompute_adv
)

######################################################################
# --------------
#

################################
# 定义采集器
# ------------------
# 我们使用 :py:class:`tianshou.data.Collector`, :py:class:`tianshou.data.VectorReplayBuffer` 来定义一个采集器。

train_collector = Collector(
    policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
)
test_collector = Collector(policy, test_envs)

######################################################################
# --------------
#

################################
# 使用训练器训练策略
# ------------------
# 使用 :py:class:`torch.utils.tensorboard.writer.SummaryWriter` 和 :py:class:`tianshou.utils.TensorboardLogger` 来生成日志。可以使用 ``tensorboard --logdir=log/CartPole-v1/ppo`` 查看。
# 我们使用 :py:class:`tianshou.trainer.OnpolicyTrainer` 来训练 ``PPO`` 策略。

# log
log_path = os.path.join(args.logdir, args.task, 'ppo')
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)
def save_best_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
def stop_fn(mean_rewards):
    return mean_rewards >= args.reward_threshold
# trainer
result = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=args.epoch,
    step_per_epoch=args.step_per_epoch,
    repeat_per_collect=args.repeat_per_collect,
    episode_per_test=args.test_num,
    batch_size=args.batch_size,
    step_per_collect=args.step_per_collect,
    stop_fn=stop_fn,
    save_best_fn=save_best_fn,
    logger=logger
).run()
assert stop_fn(result['best_reward'])
pprint.pprint(result)

######################################################################
# --------------
#

################################
# 查看智能体的表现
# ------------------
# 让我们看看它的性能！

env = gym.make(args.task)
policy.eval()
collector = Collector(policy, env)
result = collector.collect(n_episode=1, render=args.render)
rews, lens = result["rews"], result["lens"]
print(f"Final reward: {rews.mean()}, length: {lens.mean()}")