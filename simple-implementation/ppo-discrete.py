"""
`RL 工具 <rl_utils.html>`_ ||
**离散环境下的 PPO 算法简洁实现** ||
`RotatE-WN18RR <train_rotate_WN18RR_adv.html>`_

离散环境下的 PPO 算法简洁实现
===============================

这一部分介绍如何用在 离散环境（Cart Pole）中 上训练 PPO。

源教程地址：https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/

PPO 原论文：`Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`__ 。

我的论文笔记：https://cdn.jsdelivr.net/gh/LuYF-Lemon-love/susu-rl-papers/papers/01-PPO.pdf

Cart Pole：https://gymnasium.farama.org/environments/classic_control/cart_pole/

PPO 是 TRPO 算法的改进版，实现更加简洁，而且更快。PPO 的优化目标与 TRPO 相同，但 PPO 用了一些相对简单的方法来求解。

PPO 使用截断的方法在目标函数中进行限制，以保证新的参数和旧的参数的差距不会太大，即：

.. image:: /_static/images/ppo-loss.svg
    :align: center
    :height: 300


:math:`clip(x,l,r)=max(min(x,r),l)`，即把 :math:`x` 限制在 :math:`[l,r]` 内。

导入第三方库
-----------------

"""

import argparse
import os
import pprint

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
    