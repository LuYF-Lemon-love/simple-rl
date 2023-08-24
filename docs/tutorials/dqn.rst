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

Tianshou supports any user-defined PyTorch networks and optimizers. Yet, of course, the inputs and outputs must comply with Tianshou's API. Here is an example:
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

You can also use pre-defined MLP networks in :mod:`~tianshou.utils.net.common`, :mod:`~tianshou.utils.net.discrete`, and :mod:`~tianshou.utils.net.continuous`. The rules of self-defined networks are:

1. Input: observation ``obs`` (may be a ``numpy.ndarray``, ``torch.Tensor``, dict, or self-defined class), hidden state ``state`` (for RNN usage), and other information ``info`` provided by the environment.
2. Output: some ``logits``, the next hidden state ``state``. The logits could be a tuple instead of a ``torch.Tensor``, or some other useful variables or results during the policy forwarding procedure. It depends on how the policy class process the network output. For example, in PPO :cite:`PPO`, the return of the network might be ``(mu, sigma), state`` for Gaussian policy.

.. note::

    The logits here indicates the raw output of the network. In supervised learning, the raw output of prediction/classification model is called logits, and here we extend this definition to any raw output of the neural network.


Setup Policy
------------

We use the defined ``net`` and ``optim`` above, with extra policy hyper-parameters, to define a policy. Here we define a DQN policy with a target network:
::

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)


Setup Collector
---------------

The collector is a key concept in Tianshou. It allows the policy to interact with different types of environments conveniently.
In each step, the collector will let the policy perform (at least) a specified number of steps or episodes and store the data in a replay buffer.

The following code shows how to set up a collector in practice. It is worth noticing that VectorReplayBuffer is to be used in vectorized environment scenarios, and the number of buffers, in the following case 10, is preferred to be set as the number of environments.

::

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

The main function of collector is the collect function, which can be summarized in the following lines:

::

    result = self.policy(self.data, last_state)                         # the agent predicts the batch action from batch observation
    act = to_numpy(result.act)
    self.data.update(act=act)                                           # update the data with new action/policy
    result = self.env.step(act, ready_env_ids)                          # apply action to environment
    obs_next, rew, done, info = result
    self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)  # update the data with new state/reward/done/info


Train Policy with a Trainer
---------------------------

Tianshou provides :func:`~tianshou.trainer.onpolicy_trainer`, :func:`~tianshou.trainer.offpolicy_trainer`, and :func:`~tianshou.trainer.offline_trainer`. The trainer will automatically stop training when the policy reach the stop condition ``stop_fn`` on test collector. Since DQN is an off-policy algorithm, we use the :func:`~tianshou.trainer.offpolicy_trainer` as follows:
::

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
    print(f'Finished training! Use {result["duration"]}')

The meaning of each parameter is as follows (full description can be found at :func:`~tianshou.trainer.offpolicy_trainer`):

* ``max_epoch``: The maximum of epochs for training. The training process might be finished before reaching the ``max_epoch``;
* ``step_per_epoch``: The number of environment step (a.k.a. transition) collected per epoch;
* ``step_per_collect``: The number of transition the collector would collect before the network update. For example, the code above means "collect 10 transitions and do one policy network update";
* ``episode_per_test``: The number of episodes for one policy evaluation.
* ``batch_size``: The batch size of sample data, which is going to feed in the policy network.
* ``train_fn``: A function receives the current number of epoch and step index, and performs some operations at the beginning of training in this epoch. For example, the code above means "reset the epsilon to 0.1 in DQN before training".
* ``test_fn``: A function receives the current number of epoch and step index, and performs some operations at the beginning of testing in this epoch. For example, the code above means "reset the epsilon to 0.05 in DQN before testing".
* ``stop_fn``: A function receives the average undiscounted returns of the testing result, return a boolean which indicates whether reaching the goal.
* ``logger``: See below.

The trainer supports `TensorBoard <https://www.tensorflow.org/tensorboard>`_ for logging. It can be used as:
::

    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    writer = SummaryWriter('log/dqn')
    logger = TensorboardLogger(writer)

Pass the logger into the trainer, and the training result will be recorded into the TensorBoard.

The returned result is a dictionary as follows:
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

It shows that within approximately 4 seconds, we finished training a DQN agent on CartPole. The mean returns over 100 consecutive episodes is 199.03.


Save/Load Policy
----------------

Since the policy inherits the class ``torch.nn.Module``, saving and loading the policy are exactly the same as a torch module:
::

    torch.save(policy.state_dict(), 'dqn.pth')
    policy.load_state_dict(torch.load('dqn.pth'))


Watch the Agent's Performance
-----------------------------

:class:`~tianshou.data.Collector` supports rendering. Here is the example of watching the agent's performance in 35 FPS:
::

    policy.eval()
    policy.set_eps(0.05)
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=1, render=1 / 35)

If you'd like to manually see the action generated by a well-trained agent:
::

    # assume obs is a single environment observation
    action = policy(Batch(obs=np.array([obs]))).act[0]


.. _customized_trainer:

Train a Policy with Customized Codes
------------------------------------

"I don't want to use your provided trainer. I want to customize it!"

Tianshou supports user-defined training code. Here is the code snippet:
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

For further usage, you can refer to the :doc:`/tutorials/cheatsheet`.

.. rubric:: References

.. bibliography:: /refs.bib
    :style: unsrtalpha
