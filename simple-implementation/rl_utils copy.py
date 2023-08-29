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
    