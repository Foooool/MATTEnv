"""
Target Tracking Environment Base Model.
"""
import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np
from numpy import linalg as LA
import os, copy

from numpy.lib.index_tricks import _fill_diagonal_dispatcher

from ttenv.maps import map_utils
from ttenv.agent_models import *
from ttenv.policies import *
from ttenv.belief_tracker import KFbelief, UKFbelief
from ttenv.metadata import METADATA
import ttenv.util as util

from ttenv.maps.dynamic_map import DynamicMap


class MultiAgentTargetTrackingBase(gym.Env):
    def __init__(self, num_agents=1, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        """多智能体目标追踪环境基类

        Parameters
        ----------
        num_agents : int, optional
            智能体数量, by default 1
        num_targets : int, optional
            目标数量, by default 1
        map_name : str, optional
            地图名称, by default 'empty'
        is_training : bool, optional
            是否训练, by default True
        known_noise : bool, optional
            是否已知噪声协方差, by default True
        """
        gym.Env.__init__(self)
        self.seed()

        self.state = None
        self.action_space = spaces.Discrete(len(METADATA['action_v']) * len(METADATA['action_w']))
        self.action_map = {}
        for (i, v) in enumerate(METADATA['action_v']):
            for (j, w) in enumerate(METADATA['action_w']):
                self.action_map[len(METADATA['action_w'])*i+j] = (v, w)
        assert(len(self.action_map.keys())==self.action_space.n)

        self.num_agents = num_agents
        self.num_targets = num_targets
        self.viewer = None
        self.is_training = is_training

        self.sampling_period = 0.5  # 时间间隔，秒
        self.sensor_r_sd = METADATA['sensor_r_sd']  # 距离传感器噪声
        self.sensor_b_sd = METADATA['sensor_b_sd']  # 方位传感器噪声
        self.sensor_r = METADATA['sensor_r']  # 观测距离
        self.fov = METADATA['fov']  # 观测角度

        # 加载地图
        ossep = os.path.sep
        map_dir_path = ossep.join(map_utils.__file__.split(ossep)[:-1])
        if 'dynamic_map' in map_name :
            self.MAP = DynamicMap(
                map_dir_path = map_dir_path,
                map_name = map_name,
                margin2wall = METADATA['margin2wall'])
        else:
            self.MAP = map_utils.GridMap(
                map_path = os.path.join(map_dir_path, map_name),
                margin2wall = METADATA['margin2wall'])

        self.agent_init_pos =  np.array([self.MAP.origin[0], self.MAP.origin[1], 0.0])
        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = METADATA['target_init_cov']

        self.reset_num = 0

    def reset(self, **kwargs):
        self.MAP.generate_map(**kwargs)
        self.has_discovered = np.ones((self.num_agents, self.num_targets), dtype=bool)
        self.state = [[] for _ in range(self.num_agents)]
        self.num_collisions = 0
        return self.get_init_pose(**kwargs)

    def step(self, actions):
        """单步模拟

        Parameters
        ----------
        actions : int array of shape (n, )
            智能体动作列表

        Returns
        -------
        state: list
            每个智能体的观测
        reward: list of float
            每个智能体的回报
        done: bool
            该 episode 是否结束
        info: dict
            mean_nlogdetcov: float
                平均 log det cov
            std_nlogdetcov: float
                log det cov 的协方差
        """
        # 智能体移动
        is_col_n = []  # 记录是否碰撞
        action_vw_n = []
        for i, action in enumerate(actions):  # 执行每个智能体的动作
            action_vw = self.action_map[action]
            is_col = self.agents[i].update(action_vw, [t.state[:2] for t in self.targets])
            self.num_collisions += int(is_col)
            is_col_n.append(is_col)
            action_vw_n.append(action_vw)

        # 目标移动，如果还没有发现，目标保持原位
        for i in range(self.num_targets):
            if self.has_discovered[:, i].any():
                self.targets[i].update([a.state[:2] for a in self.agents])

        # 观测以及更新置信
        observed = self.observe_and_update_belief()

        # 计算回报函数
        reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(self.is_training,
                                                                is_col=is_col_n)

        # 预测 b_t+2|t+1，滤波
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                self.belief_targets[(i, j)].predict()

        # 状态更新
        self.state_func(action_vw_n, observed)

        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov, 'std_nlogdetcov': std_nlogdetcov}

    def get_init_pose(self, init_pose_list=[], target_path=[], **kwargs):
        """为智能体、目标、置信生成初始值

        Parameters
        ----------
        init_pose_list : a list of dictionaries with pre-defined initial positions.
        lin_dist_range : a tuple of the minimum and maximum distance of a target
                        and a belief target from the agent.
        ang_dist_range_target : a tuple of the minimum and maximum angular
                            distance (counter clockwise) of a target from the
                            agent. -pi <= x <= pi
        ang_dist_range_belief : a tuple of the minimum and maximum angular
                            distance (counter clockwise) of a belief from the
                            agent. -pi <= x <= pi
        blocked : True if there is an obstacle between a target and the agent.
        """
        if init_pose_list != []:
            if target_path != []:
                self.set_target_path(target_path[self.reset_num])
            self.reset_num += 1
            return init_pose_list[self.reset_num-1]
        else:
            return self.get_init_pose_random(**kwargs)

    def get_init_pose_random(self, **kwargs):
        """生成随机位置

        不再检测智能体到置信的范围以及置信到目标的范围。

        Returns
        -------
        init_pose : dict
            agents : list of [x, y, theta]
                智能体初始状态列表
            targets : list of [x, y, theta]
                目标初始状态列表
            belief : dict
                (i, j) : [x, y, theta]
                    智能体 i 关于 目标 j 的置信
        """
        init_pose = {}

        # 生成智能体初始位置和角度
        init_pose['agents'] = []
        for _ in range(self.num_agents):  # 为每个智能体生成初始位置和角度
            is_current_agent_valid = False
            while(not is_current_agent_valid):  # 生成位置并检测是否在障碍物中
                a_init = np.random.random((2,)) * (self.MAP.mapmax - self.MAP.mapmin) + self.MAP.mapmin
                is_current_agent_valid = not(self.MAP.is_collision(a_init))
            init_pose['agents'].append([a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi])

        # 生成目标初始位置和角度
        init_pose['targets'] = []
        for _ in range(self.num_targets):
            is_current_target_valid = False
            while(not is_current_target_valid):  # 生成位置并检测是否在障碍物中
                t_init = np.random.random((2,)) * (self.MAP.mapmax - self.MAP.mapmin) + self.MAP.mapmin
                is_current_target_valid = not(self.MAP.is_collision(t_init))
            init_pose['targets'].append([t_init[0], t_init[1], np.random.random() * 2 * np.pi - np.pi])

        # 生成初始置信
        init_pose['belief'] = {}
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                b_init = np.random.random((2,)) * (self.MAP.mapmax - self.MAP.mapmin) + self.MAP.mapmin
                init_pose['belief'][(i, j)] = [b_init[0], b_init[1], np.random.random() * 2 * np.pi - np.pi]

        return init_pose

    def add_history_to_state(self, state, num_target_dep_vars, num_target_indep_vars, logdetcov_idx):
        """
        Replacing the current logetcov value to a sequence of the recent few
        logdetcov values for each target.
        It uses fixed values for :
            1) the number of target dependent variables
            2) current logdetcov index at each target dependent vector
            3) the number of target independent variables
        """
        new_state = []
        for i in range(self.num_targets):
            self.logdetcov_history[i].add(state[num_target_dep_vars*i+logdetcov_idx])
            new_state = np.concatenate((new_state, state[num_target_dep_vars*i: num_target_dep_vars*i+logdetcov_idx]))
            new_state = np.concatenate((new_state, self.logdetcov_history[i].get_values()))
            new_state = np.concatenate((new_state, state[num_target_dep_vars*i+logdetcov_idx+1:num_target_dep_vars*(i+1)]))
        new_state = np.concatenate((new_state, state[-num_target_indep_vars:]))
        return new_state

    def set_target_path(self, target_path):
        targets = [Agent2DFixedPath(dim=self.target_dim, sampling_period=self.sampling_period,
                                limit=self.limit['target'],
                                collision_func=lambda x: self.MAP.is_collision(x),
                                path=target_path[i]) for i in range(self.num_targets)]
        self.targets = targets

    def observation(self, agent, target):
        """观测函数，返回智能体对于目标的观测

        Parameters
        ----------
        agent: agent_models.Agent
            智能体
        target: agent_models.Agent
            目标

        Returns
        -------
        observed: bool
            是否观测到目标
        z: (float, float)
            观测，距离和方位值
        """
        r, alpha = util.relative_distance_polar(target.state[:2],
                                                xy_base=agent.state[:2],
                                                theta_base=agent.state[2])
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(self.MAP.is_blocked(agent.state, target.state)))
        z = None
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        return observed, z

    def observation_noise(self, z):
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0],
                                [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def observe_and_update_belief(self):
        """观测及置信更新

        Returns
        -------
        observed: array of shape (num_agents, num_targets)
            布尔矩阵，表示智能体 i 是否观测到目标 j
        """
        observed = np.zeros((self.num_agents, self.num_targets), dtype=bool)
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                observation = self.observation(self.agents[i], self.targets[j])
                observed[i, j] = bool(observation[0])

                # 如果观测到目标，更新置信
                if observation[0]:  # if observed, update the target belief.
                    self.belief_targets[(i, j)].update(observation[1], self.agents[i].state)
                    if not(self.has_discovered[i, j]):
                        self.has_discovered[i, j] = 1
        return observed

    def get_reward(self, is_training=True, **kwargs):
        return sharing_reward_fun(self.belief_targets, self.num_agents, self.num_targets, is_training=is_training, **kwargs)


def sharing_reward_fun(belief_targets, num_agents, num_targets, is_col, is_training=True, c_mean=0.1, c_std=0.0, c_penalty=1.0):
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
    is_col : bool
        是否碰撞
    is_training : bool, optional
        是否训练, by default True
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
    done: bool
        是否结束
    r_detcov_mean: float
        平均 det cov 
    r_detcov_std: float
        det cov 的协方差
    """

    detcov = [np.min([LA.det(belief_targets[(i, j)].cov) for i in range(num_agents)]) for j in range(num_targets)]
    r_detcov_mean = - np.mean(np.log(detcov))
    r_detcov_std = - np.std(np.log(detcov))

    reward = c_mean * r_detcov_mean + c_std * r_detcov_std
    if is_col:
        reward = min(0.0, reward) - c_penalty * 1.0
    return [reward] * num_agents, False, r_detcov_mean, r_detcov_std
