Deep Q Network
==============

深度强化学习在各种应用中取得了重大成功。
**Deep Q Network** (DQN) :cite:`DQN` 是其中的先驱。
在本教程中，我们会逐步展示如何在 Cartpole 任务上使用天授训练一个 DQN 智能体。 
完整的代码位于 `test/discrete/test_dqn.py <https://github.com/LuYF-Lemon-love/simple-rl/blob/main/test/discrete/test_dqn.py>`_.

概述
--------

在强化学习中，智能体与环境交互以改善自身。

.. image:: /_static/images/rl-loop.jpg
    :align: center
    :height: 200

RL 训练管道中有三种类型的数据流：

1. 智能体 -> 环境： ``action`` 将由智能体生成并发送到环境;
2. 环境 -> 智能体： ``env.step`` 采取 ``action``，然后返回一个 ``(observation, reward, done, info)`` 元组；
3. 智能体-环境交互 -> 智能体训练：交互生成的数据将被存储并发送给智能体让其学习。

在以下部分中，我们将设置（vectorized）环境、策略（neural network）、收集器（buffer）和训练器（trainer），以成功运行 RL 训练和评估管道。这是整个系统：

.. image:: /_static/images/pipeline.png
    :align: center
    :height: 300


创建环境
-------------------

首先，您必须为智能体创建一个与之交互的环境。您可以使用 ``gym.make(environment_name)`` 为您的智能体创建环境。
对于环境接口，我们遵循 `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ 的惯例。在你的 Python 代码中，只需导入天授并创建环境：
::

    import gymnasium as gym
    import tianshou as ts

    env = gym.make('CartPole-v0')

CartPole-v0 是一辆携带杆子在轨道上移动的推车。这是一个具有离散动作空间（discrete action space）的简单环境，DQN 适用于该环境。在使用不同种类的强化学习算法前，您需要了解每个算法是否能够应用在离散动作空间场景 / 连续动作空间场景中，比如像 DDPG :cite:`DDPG` 就只能用在连续动作空间任务中，其他基于策略梯度的算法可以用在任意这两个场景中。

以下是 CartPole-v0 有用字段的详细信息：

- ``state``: 推车的位置、推车的速度、杆子的角度和杆尖的速度；
- ``action``: 只能是 ``[0, 1, 2]`` 其中之一，用于将推车向左移动、不移动和向右移动；
- ``reward``: 你每坚持一步，就会获得 +1 ``reward``；
- ``done``: 如果 CartPole 超出范围或超时（杆与垂直方向的夹角超过 15 度，或者推车与中心的距离超过 2.4 个单位，或者你持续了 200 多个时间步）；
- ``info``: 来自环境模拟的额外信息。

我们的目标是制定一个好的策略，在这种环境下可以获得最高的回报。

设置矢量化环境
----------------------------

此处定义训练环境和测试环境。使用原来的 ``gym.Env`` 当然是可以的：
::

    train_envs = gym.make('CartPole-v0')
    test_envs = gym.make('CartPole-v0')

天授支持所有算法的矢量化环境。它提供了四种类型的矢量化环境包装器：

- :class:`~tianshou.env.DummyVectorEnv`: 顺序版本，使用单线程 for 循环;
- :class:`~tianshou.env.SubprocVectorEnv`: 使用 Python 多进程进行并发执行;
- :class:`~tianshou.env.ShmemVectorEnv`: 使用共享内存而不是基于 SubprocVectorEnv 的管道;
- :class:`~tianshou.env.RayVectorEnv`: 将 Ray 用于并发活动，是目前在具有多台机器的集群中进行并行模拟的唯一选择。它可以按如下方式使用：（更多解释可以在 :ref:`parallel_sampling` 中找到）

::

    train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])

在这里，我们在 ``train_envs`` 中设置了 10 个环境，在 ``test_envs`` 中设置了 100 个环境。

您也可以通过以下方式尝试超快速矢量化环境 `EnvPool <https://github.com/sail-sg/envpool/>`_ 。

::

    import envpool
    train_envs = envpool.make_gymnasium("CartPole-v0", num_envs=10)
    test_envs = envpool.make_gymnasium("CartPole-v0", num_envs=100)

为了演示，这里我们使用第二个代码块。

.. warning::

    如果您使用自己的环境，请确保正确设置 ``seed`` 方法，例如：

    ::

        def seed(self, seed):
            np.random.seed(seed)

    否则，这些 env 的输出可能彼此相同。


.. _build_the_network:

构建神经网络
-----------------

天授支持任意的用户定义的 PyTorch 网络和优化器，但是输入输出需要遵循既定 API，比如像下面这样：
::

    import torch, numpy as np
    from torch import nn

    class Net(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape)),
            )

        def forward(self, obs, state=None, info={}):
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            return logits, state

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

您还可以使用 :mod:`~tianshou.utils.net.common`, :mod:`~tianshou.utils.net.discrete`, 和 :mod:`~tianshou.utils.net.continuous` 的预定义 MLP 网络。自定义网络的规则是：

1. 输入: observation ``obs`` (可能是 ``numpy.ndarray``, ``torch.Tensor``, dict, 或 self-defined class), hidden state ``state`` (用于 RNN), and 环境提供的其他信息 ``info``.
2. 输出: some ``logits``, the next hidden state ``state``. logits 可以是 tuple 而不是 ``torch.Tensor``, 或者在策略 forward 过程中是一些其他有用的变量或结果。这取决于策略类如何处理网络输出。例如, 在 PPO :cite:`PPO`, 网络的返回可能是 ``(mu, sigma), state`` ，即高斯策略的状态.

.. note::

    这里的 logits 表示网络的原始输出。在监督学习中，预测/分类模型的原始输出称为 logits，这里我们将此定义扩展到神经网络的任何原始输出。


初始化策略
------------

我们使用上述代码中定义的 ``net`` 和 ``optim``，以及其他超参数，来定义一个策略。此处定义了一个有目标网络（Target Network）的 DQN 策略：
::

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)


定义采集器
---------------

采集器（Collector）是天授中的一个关键概念。它定义了策略与不同环境交互的逻辑。
在每一回合（step）中，采集器会让策略与环境交互指定数目（至少）的步数或者轮数，并且会将产生的数据存储在重放缓冲区中。

以下代码演示如何在实践中设置采集器。值得注意的是，VectorReplayBuffer 将用于矢量化环境方案，并且缓冲区数（在以下情况下为 10）优先设置为环境数。
::

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

采集器的主要功能是采集功能，可以概括为以下几行：
::

    result = self.policy(self.data, last_state)                         # the agent predicts the batch action from batch observation
    act = to_numpy(result.act)
    self.data.update(act=act)                                           # update the data with new action/policy
    result = self.env.step(act, ready_env_ids)                          # apply action to environment
    obs_next, rew, done, info = result
    self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)  # update the data with new state/reward/done/info


使用训练器训练策略
---------------------------

天授提供了 :func:`~tianshou.trainer.onpolicy_trainer`, :func:`~tianshou.trainer.offpolicy_trainer`, 和 :func:`~tianshou.trainer.offline_trainer`. 训练器会在 test collector 的 ``stop_fn`` 达到条件的时候停止训练。由于 DQN 是一种异策略算法 (off-policy algorithm) , 因此使用 :func:`~tianshou.trainer.offpolicy_trainer` 进行训练：
::

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
    print(f'Finished training! Use {result["duration"]}')

每个参数的具体含义如下：(完整的描述可以在 :func:`~tianshou.trainer.offpolicy_trainer` 中找到):

* ``max_epoch``: 最大允许的训练轮数，有可能没训练完这么多轮就会停止（因为满足了 ``stop_fn`` 的条件）；
* ``step_per_epoch``: 每个 epoch 要更新多少次策略网络；
* ``step_per_collect``: 每次更新前要收集多少帧与环境的交互数据。上面的代码参数意思是，每收集 10 帧进行一次网络更新；
* ``episode_per_test``: 一次策略评估的回合数；
* ``batch_size``: 每次策略计算的时候批量处理多少数据；
* ``train_fn``: 在每个 epoch 训练之前被调用的函数，输入的是当前第几轮 epoch 和当前用于训练的 env 一共 step 了多少次。上面的代码意味着，在每次训练前将 epsilon 设置成 0.1；
* ``test_fn``: 在每个 epoch 测试之前被调用的函数，输入的是当前第几轮 epoch 和当前用于训练的 env 一共 step 了多少次。上面的代码意味着，在每次测试前将 epsilon 设置成 0.05；
* ``stop_fn``: 停止条件，输入是当前平均总奖励回报（the average undiscounted returns），返回是否要停止训练；
* ``logger``: 天授支持 `TensorBoard <https://www.tensorflow.org/tensorboard>`_，可以像下面这样初始化：

::

    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    writer = SummaryWriter('log/dqn')
    logger = TensorboardLogger(writer)

把 logger 送进去，训练器会自动把训练日志记录在里面。

训练器返回的结果是个字典，如下所示：
::

    {
        'train_step': 9246,
        'train_episode': 504.0,
        'train_time/collector': '0.65s',
        'train_time/model': '1.97s',
        'train_speed': '3518.79 step/s',
        'test_step': 49112,
        'test_episode': 400.0,
        'test_time': '1.38s',
        'test_speed': '35600.52 step/s',
        'best_reward': 199.03,
        'duration': '4.01s'
    }

可以看出大概 4 秒就在 CartPole 任务上训练出来一个 DQN 智能体，在 100 轮测试中平均总奖励回报为 199.03。

存储、导入策略
----------------

因为策略继承了 ``torch.nn.Module``，所以存储和导入策略和 PyTorch 中的网络并无差别，如下所示：
::

    torch.save(policy.state_dict(), 'dqn.pth')
    policy.load_state_dict(torch.load('dqn.pth'))


可视化智能体的表现
-----------------------------

采集器 :class:`~tianshou.data.Collector` 支持渲染智能体的表现。下面的代码展示了以 35 FPS 的帧率查看智能体表现：
::

    policy.eval()
    policy.set_eps(0.05)
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=1, render=1 / 35)

如果您想手动查看由训练有素的智能体生成的 ``action``：
::

    # assume obs is a single environment observation
    action = policy(Batch(obs=np.array([obs]))).act[0]


.. _customized_trainer:

定制化训练器
------------------------------------

"I don't want to use your provided trainer. I want to customize it!"

天授为了能够支持用户的定制化训练器，在 Trainer 做了尽可能少的封装。使用者可以自由地编写自己所需要的训练策略，比如：
::

    # pre-collect at least 5000 transitions with random action before training
    train_collector.collect(n_step=5000, random=True)

    policy.set_eps(0.1)
    for i in range(int(1e6)):  # total step
        collect_result = train_collector.collect(n_step=10)

        # once if the collected episodes' mean returns reach the threshold,
        # or every 1000 steps, we test it on test_collector
        if collect_result['rews'].mean() >= env.spec.reward_threshold or i % 1000 == 0:
            policy.set_eps(0.05)
            result = test_collector.collect(n_episode=100)
            if result['rews'].mean() >= env.spec.reward_threshold:
                print(f'Finished training! Test mean returns: {result["rews"].mean()}')
                break
            else:
                # back to training eps
                policy.set_eps(0.1)

        # train policy with a sampled batch data from buffer
        losses = policy.update(64, train_collector.buffer)

如需进一步使用，您可以参考 :doc:`/tutorials/cheatsheet`。

.. rubric:: References

.. bibliography:: /refs.bib
    :style: unsrtalpha
