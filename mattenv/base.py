"""
多智能体目标追踪环境基类
"""
import os

import numpy as np
from numpy import linalg as LA
import gym
from gym import spaces

from mattenv.metadata import METADATA
from mattenv.agent_models import *
from mattenv.maps import map_utils
from mattenv.maps.dynamic_map import DynamicMap
import mattenv.util as util
from mattenv.reward_function import sharing_reward_fun


class MultiAgentTargetTrackingBase(gym.Env):
    def __init__(self,
                 num_agents=1,
                 num_targets=1,
                 map_name='empty',
                 seed=None,
                 reward_function=None,
                 im_size=28,
                 **kwargs):
        """多智能体目标追踪环境基类

        环境需要继承该类，新环境需要完成：
        + 设置智能体与目标的具体模型
        + 创建观测函数
        + 滤波设置

        Parameters
        ----------
        num_agents : int, optional
            智能体数量, by default 1
        num_targets : int, optional
            目标数量, by default 1
        map_name : str, optional
            地图名称, by default 'empty'
        seed : int | None, optimal
            随机数种子, by default None
        """
        super(MultiAgentTargetTrackingBase, self).__init__()

        # 随机数种子
        if seed is not None:
            self.seed(seed)

        # 动作空间
        # action_space 为单个智能体的工作空间，大小为线速度离散值与角速度离散值的积
        # action_map 将离散动作编号映射为 <v, w> pair 上
        self.action_space = spaces.Discrete(len(METADATA['action_v']) * len(METADATA['action_w']))
        self.action_map = {}
        for (i, v) in enumerate(METADATA['action_v']):
            for (j, w) in enumerate(METADATA['action_w']):
                self.action_map[len(METADATA['action_w'])*i+j] = (v, w)
        assert(len(self.action_map.keys())==self.action_space.n)

        # 模拟设置
        self.sampling_period = METADATA.get('sampling_period', 0.5)                  # 时间间隔，秒
        self.sensor_r_sd = METADATA['sensor_r_sd']  # 距离传感器噪声
        self.sensor_b_sd = METADATA['sensor_b_sd']  # 方位传感器噪声
        self.sensor_r = METADATA['sensor_r']        # 观测距离
        self.fov = METADATA['fov']                  # 观测角度

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

        # 智能体和目标数量
        self.num_agents = num_agents
        self.num_targets = num_targets

        # 初始化模型
        self._build_models()

        # 回报函数
        if reward_function is None:
            self.reward_function = sharing_reward_fun
        else:
            self.reward_function = reward_function

        self.last_actions = [[0, 0] for _ in range(self.num_agents)]
        self.im_size = im_size

    def reset(self, **kwargs):
        """重置环境
        """
        # 初始化地图
        self.MAP.generate_map(**kwargs)

        # 重置智能体
        self.initialize_models()

        # 返回观测
        return self.get_observation()

    def _build_models(self):
        """初始化智能体、目标、belief 模型"""
        raise NotImplementedError('未实现构造模型方法')

    def initialize_models(self):
        """初始化模型
        """
        self._build_models()

    def get_observation(self):
        """返回每个智能体的观测
        """
        raise NotImplementedError('未实现观测函数')

    def step(self, actions):
        """单步模拟

        Parameters
        ----------
        actions : int array of shape (n, )
            智能体动作列表

        Returns
        -------
        observation_n : list of observation
            每个智能体的观测
        reward_n : list of float
            每个智能体的回报
        done: bool
            该 episode 是否结束
        info: None
        """
        self.last_actions = []
        # 智能体移动
        is_col_n = []  # 记录是否碰撞
        for agent, action in zip(self.agents, actions):
            # 执行每个智能体的动作
            action_vw = self.action_map[action]
            is_col = agent.update(action_vw)
            # 记录
            is_col_n.append(is_col)
            self.last_actions.append(action_vw)

        # 目标移动
        for target in self.targets:
            target.update([agent.state[:2] for agent in self.agents])

        # 观测以及更新置信
        observation = self.get_observation()

        # 计算回报函数
        reward = self.reward_function(self.belief_targets,
            self.num_agents,
            self.num_targets,
            is_col_n=is_col_n)

        # 预测 b_t+2|t+1，滤波
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                self.belief_targets[(i, j)].predict()

        # info
        done = False
        info = None

        return observation, reward, done, info

    def _get_init_pose_random(self):
        """生成随机位置以及初始

        智能体  随机分布在地图上，如果有障碍物，保证智能体不在障碍物中；角度在 [-pi, pi] 中随机选取

        目标    随机分布在地图上，如果有障碍物，保证目标不在障碍物中；角度在 [-pi, pi] 中随机选取

        置信    智能体 i 对目标 j 的置信随机分布在地图上

        Returns
        -------
        init_pose : dict
            agents : [[x, y, theta]]
                智能体初始状态列表
            targets : [[x, y, x\dot, y\dot]]
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
            init_pose['targets'].append([t_init[0], t_init[1]] + METADATA.get('target_init_vel', [0, 0]))

        # 生成初始置信
        init_pose['belief'] = {}
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                b_init = np.random.random((2,)) * (self.MAP.mapmax - self.MAP.mapmin) + self.MAP.mapmin
                init_pose['belief'][(i, j)] = [b_init[0], b_init[1], np.random.random() * 2 * np.pi - np.pi]

        return init_pose

    def _observation(self, agent, target):
        """观测函数，返回一个智能体对于一个目标的观测

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
        # 计算以智能体为中心，目标的极坐标
        r, alpha = util.relative_distance_polar(target.state[:2],
                                                xy_base=agent.state[:2],
                                                theta_base=agent.state[2])
        
        # 计算能否观测到目标：距离范围、角度范围、遮挡
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(self.MAP.is_blocked(agent.state, target.state)))
        z = None

        # 如果可以观测到，添加观测噪声
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2,), self._observation_noise_cov(z))
        return observed, z

    def _observation_noise_cov(self, z):
        """返回观测的噪声

        这里的实现是噪声与观测范围以及角度无关，距离的观测与角度观测独立

        Parameters
        ----------
        z : numpy.ndarray of shape (2, )
            原始观测，距离与方位，(r, theta)

        Returns
        -------
        obs_noise_cov : numpy.ndarray of shape (2, 2)
            协方差矩阵
        """
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0],
                                [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def _observe_and_update_belief(self):
        """观测及置信更新

        Returns
        -------
        observed: array of shape (num_agents, num_targets)
            布尔矩阵，表示智能体 i 是否观测到目标 j
        """
        observed = np.zeros((self.num_agents, self.num_targets), dtype=bool)
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                observation = self._observation(self.agents[i], self.targets[j])
                observed[i, j] = bool(observation[0])

                # 如果观测到目标，更新置信
                if observation[0]:  # if observed, update the target belief.
                    self.belief_targets[(i, j)].update(observation[1], self.agents[i].state)
                    
        return observed
