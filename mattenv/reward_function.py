"""回报函数定义
"""
import numpy as np
import numpy.linalg as LA


def sharing_reward_fun(belief_targets, num_agents, num_targets, is_col_n, c_mean=0.1, c_std=0.0, c_penalty=1.0):
    """回报函数

    Parameters
    ----------
    belief_targets : dict
        (i, j) : belief
        第 i 个智能体关于第 j 个目标的置信
    num_agents : int
        智能体个数
    num_targets : int
        目标个数
    is_col_n : bool
        是否碰撞
    c_mean : float, optional
        均值系数, by default 0.1
    c_std : float, optional
        方差项系数, by default 0.0
    c_penalty : float, optional
        惩罚项系数, by default 1.0

    Returns
    -------
    reward: float list, shape (num_agents, )
        每个智能体的回报
    r_detcov_mean: float
        平均 det cov 
    r_detcov_std: float
        det cov 的协方差
    """

    detcov = [np.min([LA.det(belief_targets[(i, j)].cov) for i in range(num_agents)]) for j in range(num_targets)]
    r_detcov_mean = - np.mean(np.log(detcov))
    r_detcov_std = - np.std(np.log(detcov))

    reward = c_mean * r_detcov_mean + c_std * r_detcov_std

    # 为每个智能体生成回报
    reward_n = np.ones(num_agents) * reward
    reward_n -= np.array(is_col_n, dtype=np.int) * c_penalty

    return reward_n
