## Simple-RL

本项目是基于 [thu-ml/tianshou](https://github.com/thu-ml/tianshou) 开发的，主要用于个人学习，如果想要直接使用强化学习算法建议直接使用源项目 [thu-ml/tianshou](https://github.com/thu-ml/tianshou)。

>在开头，我想先表达一下对原作者的致敬：[我与清华学生的差距](https://www.zhihu.com/question/377263715)

>**Tianshou** is a reinforcement learning platform based on pure PyTorch. The supported interface algorithms currently include:

- [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
- [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)
- [Branching DQN](https://arxiv.org/pdf/1711.08946.pdf)
- [Categorical DQN (C51)](https://arxiv.org/pdf/1707.06887.pdf)
- [Rainbow DQN (Rainbow)](https://arxiv.org/pdf/1710.02298.pdf)
- [Quantile Regression DQN (QRDQN)](https://arxiv.org/pdf/1710.10044.pdf)
- [Implicit Quantile Network (IQN)](https://arxiv.org/pdf/1806.06923.pdf)
- [Fully-parameterized Quantile Function (FQF)](https://arxiv.org/pdf/1911.02140.pdf)
- [Policy Gradient (PG)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [Natural Policy Gradient (NPG)](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- [Advantage Actor-Critic (A2C)](https://openai.com/blog/baselines-acktr-a2c/)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Randomized Ensembled Double Q-Learning (REDQ)](https://arxiv.org/pdf/2101.05982.pdf)
- [Discrete Soft Actor-Critic (SAC-Discrete)](https://arxiv.org/pdf/1910.07207.pdf)
- Vanilla Imitation Learning
- [Batch-Constrained deep Q-Learning (BCQ)](https://arxiv.org/pdf/1812.02900.pdf)
- [Conservative Q-Learning (CQL)](https://arxiv.org/pdf/2006.04779.pdf)
- [Twin Delayed DDPG with Behavior Cloning (TD3+BC)](https://arxiv.org/pdf/2106.06860.pdf)
- [Discrete Batch-Constrained deep Q-Learning (BCQ-Discrete)](https://arxiv.org/pdf/1910.01708.pdf)
- [Discrete Conservative Q-Learning (CQL-Discrete)](https://arxiv.org/pdf/2006.04779.pdf)
- [Discrete Critic Regularized Regression (CRR-Discrete)](https://arxiv.org/pdf/2006.15134.pdf)
- [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/pdf/1606.03476.pdf)
- [Prioritized Experience Replay (PER)](https://arxiv.org/pdf/1511.05952.pdf)
- [Generalized Advantage Estimator (GAE)](https://arxiv.org/pdf/1506.02438.pdf)
- [Posterior Sampling Reinforcement Learning (PSRL)](https://www.ece.uvic.ca/~bctill/papers/learning/Strens_2000.pdf)
- [Intrinsic Curiosity Module (ICM)](https://arxiv.org/pdf/1705.05363.pdf)
- [Hindsight Experience Replay (HER)](https://arxiv.org/pdf/1707.01495.pdf)

Here are Tianshou's other features:

- Elegant framework, using only ~4000 lines of code
- State-of-the-art [MuJoCo benchmark](https://github.com/thu-ml/tianshou/tree/master/examples/mujoco) for REINFORCE/A2C/TRPO/PPO/DDPG/TD3/SAC algorithms
- Support vectorized environment (synchronous or asynchronous) for all algorithms [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#parallel-sampling)
- Support super-fast vectorized environment [EnvPool](https://github.com/sail-sg/envpool/) for all algorithms [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#envpool-integration)
- Support recurrent state representation in actor network and critic network (RNN-style training for POMDP) [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#rnn-style-training)
- Support any type of environment state/action (e.g. a dict, a self-defined class, ...) [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#user-defined-environment-and-different-state-representation)
- Support customized training process [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#customize-training-process)
- Support n-step returns estimation and prioritized experience replay for all Q-learning based algorithms; GAE, nstep and PER are very fast thanks to numba jit function and vectorized numpy operation
- Support multi-agent RL [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#multi-agent-reinforcement-learning)
- Support both [TensorBoard](https://www.tensorflow.org/tensorboard) and [W&B](https://wandb.ai/) log tools
- Support multi-GPU training [Usage](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#multi-gpu)
- Comprehensive documentation, PEP8 code-style checking, type checking and thorough [tests](https://github.com/thu-ml/tianshou/actions)

## 安装

你能使用下面命令来安装：

```bash
$ git clone git@github.com:LuYF-Lemon-love/simple-rl.git
$ cd simple-rl
$ python -m venv env
$ source env/bin/activate
$ which python
$ pip install --upgrade pip
$ pip install .
```

你也能直接从 GitHub 安装最新的版本：

```bash
$ pip install git+https://github.com/LuYF-Lemon-love/simple-rl.git@main --upgrade
```

安装完成后，打开 Python 运行下面的代码：

```python
import tianshou
print(tianshou.__version__)
```

如果没有错误发生，你已经安装成功了。

### 安装文档（可选）

1. 安装 nginx：

```shell
$ sudo apt-get install nginx
```

2. 创建存放网站的文件夹：

```shell
$ sudo mkdir -p /var/www/simple_rl
```

3. 修改配置文件，修改 `root` 后面的内容，使其指向存放网页文件的文件夹（`root /var/www/simple_rl;`）：

```shell
$ sudo vim /etc/nginx/sites-enabled/default
```

4. 构建文档并复制到上面创建的文件中：

```shell
$ pip install -r docs/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$ cd docs/
$ make html
$ sudo cp -r _build/html/* /var/www/simple_rl/
```

5. 重启服务：

```shell
$ sudo systemctl restart nginx
```

6. 在浏览器中输入服务器 IP 地址即可访问文档了。

## 例子

保存在 [test/](./test/) 文件夹和 [examples/](./examples/) 文件夹。

## 模块化策略

天授把所有的算法大致解耦为以下部分：

- `__init__`: 策略初始化；
- `forward`: 根据给定的观测值 obs，计算出动作值 action；
- `process_fn`: 对来自重放缓冲区的数据进行预处理（因为我们已经将所有算法重新制定为基于重放缓冲区算法）；
- `learn`: 使用一个 batch 的数据进行策略的更新；
- `post_process_fn`: 从学习过程中更新重放缓冲区（例如，优先重放缓冲区需要更新权重）;
- `update`: 最主要的训练接口。这个 ``update`` 函数先是从 ``buffer`` 采样数据，然后调用 ``process_fn`` 预处理数据 (such as computing n-step return)，然后学习并更新策略，然后调用 ``post_process_fn`` (such as updating prioritized replay buffer) 完成一次迭代： ``process_fn -> learn -> post_process_fn``。

在此 API 中，我们可以方便地与不同的策略进行交互。

## Quick Start

This is an example of Deep Q Network. You can also run the full script at [test/discrete/test_dqn.py](./test/discrete/test_dqn.py).

First, import some relevant packages:

```python
import gymnasium as gym
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
```

Define some hyper-parameters:

```python
task = 'CartPole-v0'
lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10
logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))  # TensorBoard is supported!
# For other loggers: https://tianshou.readthedocs.io/en/master/tutorials/logger.html
```

Make environments:

```python
# you can also try with SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
```

Define the network:

```python
from tianshou.utils.net.common import Net
# you can define other net by following the API:
# https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network
env = gym.make(task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
optim = torch.optim.Adam(net.parameters(), lr=lr)
```

Setup policy and collectors:

```python
policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method
```

Let's train it:

```python
result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
    test_num, batch_size, update_per_step=1 / step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    logger=logger)
print(f'Finished training! Use {result["duration"]}')
```

Save / load the trained policy (it's exactly the same as PyTorch `nn.module`):

```python
torch.save(policy.state_dict(), 'dqn.pth')
policy.load_state_dict(torch.load('dqn.pth'))
```

Watch the performance with 35 FPS:

```python
policy.eval()
policy.set_eps(eps_test)
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)
```

Look at the result saved in tensorboard: (with bash script in your terminal)

```bash
$ tensorboard --logdir log/dqn
```

You can check out the [documentation](https://tianshou.readthedocs.io) for advanced usage.

It's worth a try: here is a test on a laptop (i7-8750H + GTX1060). It only uses **3** seconds for training an agent based on vanilla policy gradient on the CartPole-v0 task: (seed may be different across different platform and device)

```bash
$ python3 test/discrete/test_pg.py --seed 0 --render 0.03
```

<div align="center">
  <img src="https://github.com/thu-ml/tianshou/raw/master/docs/_static/images/testpg.gif"></a>
</div>

## 参考

<details><summary> 展开 </summary><p>

[1] [Gymnasium](http://github.com/Farama-Foundation/Gymnasium)

[2] [Welcome to Tianshou!](https://tianshou.readthedocs.io/en/master/)

[3] [欢迎查看天授平台中文文档](https://tianshou.readthedocs.io/zh/master/)

[4] [Tianshou: A Highly Modularized Deep Reinforcement Learning Library](https://jmlr.org/papers/v23/21-1127.html)

[5] [MANIFEST.in](https://packaging.python.org/en/latest/guides/using-manifest-in/)

[6] [Including Data Files](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#including-data-files)

[7] [thu-ml/tianshou](https://github.com/thu-ml/tianshou)

[8] [Getting started with Sphinx](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html)

[9] [Getting started](https://www.sphinx-doc.org/en/master/tutorial/getting-started.html)

[10] [使用Sphinx搭建本地(window)文档并把它部署到网上(github)-本地搭建(1)](https://blog.csdn.net/u013716535/article/details/104902308/)

[11] [Nginx+Ubuntu实现静态网页Web服务器](https://zhuanlan.zhihu.com/p/400447282)

[12] [Nginx快速入门](https://www.kuangstudy.com/bbs/1353634800149213186)

[13] [使用Sphinx搭建本地(window)文档并把它部署到网上(github)-布置主题并发布(2)](https://blog.csdn.net/u013716535/article/details/104907240)

[14] [原作者的毕业论文](https://tianshou.readthedocs.io/zh/master/_static/thesis.pdf)

[15] [ArgumentParser.parse_known_args](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser.parse_known_args)

[16] [github_tag_no_valid 【最新】解决方法](https://blog.csdn.net/sandwich_iot/article/details/120452369)

[17] [typing.cast(typ, val)](https://docs.python.org/zh-cn/3/library/typing.html#typing.cast)

[18] [collections.deque](https://docs.python.org/zh-cn/3/library/collections.html#collections.deque)

[19] [UndefinedError: 'style' is undefined](https://github.com/readthedocs/readthedocs.org/issues/10279#issuecomment-1544411815)

[20] [torch.distributions.categorical.Categorical](https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical)

[21] [torch.distributions.normal.Normal](https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal)

[22] [torch.nn.Softplus](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#torch.nn.Softplus)

[23] [正态分布](https://baike.baidu.com/item/%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83/829892)

[24] [数据可视化——plt:用python画图（一）](https://zhuanlan.zhihu.com/p/457419869)

[25] [Numpy怎样给数组增加一个维度 _](https://blog.csdn.net/hello_program_world/article/details/119417863)

[26] [【Pytorch】Tensor张量增加维度的四种方法](https://blog.csdn.net/weixin_43941438/article/details/130906989)

[27] [vscode ssh 远程ubuntu，plt.show不显示图片问题](https://blog.csdn.net/qq_42817826/article/details/129759368)

[28] [python在VScode中使用matplotlib库的plt.show()无法显示图标窗口](https://blog.csdn.net/weixin_64709717/article/details/130814421)

[29] [AttributeError:‘CartPoleEnv‘ object has no attribute ‘seed‘解决方案](https://blog.csdn.net/dream6985/article/details/126847399)

[30] [Interacting with the Environment](https://gymnasium.farama.org/content/basic_usage/#interacting-with-the-environment)

[31] [gymnasium.Env.reset](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)

</p></details>