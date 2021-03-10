"""多智能体目标追踪环境
"""
import numpy as np
import numpy.linalg as LA
from gym import spaces
from mattenv.base import MultiAgentTargetTrackingBase
from mattenv.agent_models import AgentSE2, AgentDoubleInt2D_Nonlinear
from mattenv.belief_tracker import KFbelief
from mattenv.metadata import METADATA
import mattenv.util as util


class MultiAgentTargetTrackingEnv1(MultiAgentTargetTrackingBase):
    def __init__(self,
                 num_agents=2,
                 num_targets=2,
                 map_name='empty',
                 seed=None,
                 reward_function=None,
                 known_noise=True,
                 **kwargs):
        """多智能体目标追踪环境 V1
        + 智能体采用 SE2 模型
        + 目标采用二重积分模型，状态为 $ x, y, x\dot, y\dot $
        + 观测为 $ (r, \theta) $
        + RL 状态为 $ (r, \theta) $

        Parameters
        ----------
        num_agents : int, optional
            智能体数量, by default 2
        num_targets : int, optional
            目标数量, by default 2
        map_name : str, optional
            地图名称, by default 'empty'
        seed : int, optional
            随机数种子, by default None
        reward_function : callable, optional
            回报函数, by default None
        """
        self.id = 'MultiAgentTargetTracking-v1'
        self.known_noise = known_noise  # 是否知道目标动力学协方差矩阵
        self.target_dim = 4

        MultiAgentTargetTrackingBase.__init__(self,
                                              num_agents=num_agents,
                                              num_targets=num_targets,
                                              map_name=map_name,
                                              seed=seed,
                                              reward_function=reward_function,
                                              **kwargs)

    def _build_models(self):
        """初始化智能体、目标、belief 模型"""
        # 生成初始状态
        init_pose = self._get_init_pose_random()

        # 生成 limit
        self._set_limits()

        # 创建智能体
        self.agents = [AgentSE2(init_state,
                                self.sampling_period,
                                self.limit['agents'],
                                collision_func=self.MAP.is_collision)
                       for init_state in init_pose['agents']]
        
        # 创建目标
        # A 矩阵
        # 1  0 \tau  0
        # 0  1  0   \tau
        # 0  0  1    0
        # 0  0  0    1
        self.target_A = np.eye(4) + self.sampling_period * np.eye(4, k=2)
        # 协方差矩阵
        self.const_q = METADATA.get('const_q', 0.2)  # scalar
        self.target_noise_cov = self.const_q * np.concatenate((
            np.concatenate((self.sampling_period**3/3*np.eye(2),
                            self.sampling_period**2/2*np.eye(2)), axis=1),
            np.concatenate((self.sampling_period**2/2*np.eye(2),
                            self.sampling_period*np.eye(2)), axis=1)))
        self.targets = [AgentDoubleInt2D_Nonlinear(init_state,
                                                   self.sampling_period,
                                                   self.limit['targets'],
                                                   collision_func=self.MAP.is_collision,
                                                   A=self.target_A,
                                                   W=self.target_noise_cov,
                                                   obs_check_func=lambda x: self.MAP.get_closest_obstacle(
                                                       x, fov=2*np.pi, r_max=10e2))
                        for init_state in init_pose['targets']]

        # belief
        self.belief_targets = {}
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                self.belief_targets[(i, j)] = KFbelief(dim=self.target_dim,
                                                       limit=self.limit['targets'],
                                                       A=self.target_A,
                                                       W=self.target_noise_cov,
                                                       obs_noise_func=self._observation_noise_cov,
                                                       collision_func=self.MAP.is_collision)

    def _set_limits(self, target_speed_limit=None):
        """生成智能体、目标、观测的上下界

        Results
        -------
        self.limit : dict
            agents : [low, high], dim = 3
            targets : [low, high], dim = 4
            observation : [low, high], dim = 6 * num_targets

        Parameters
        ----------
        target_speed_limit : float, optional
            目标最大速度, by default None
        """
        # 目标最大速度
        if target_speed_limit is None:
            self.target_speed_limit = METADATA.get('target_speed_limit', 1.0)
        else:
            self.target_speed_limit = target_speed_limit
        # 最大相对速度
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0]  

        # 设置 limit
        self.limit = {}  # 0: low, 1:highs
        # 智能体状态为 (x, y, theta)
        self.limit['agents'] = [np.concatenate((self.MAP.mapmin, [-np.pi])),
                               np.concatenate((self.MAP.mapmax, [np.pi]))]
        # 目标状态为 (x, y, x\dot, y\dot)
        self.limit['targets'] = [np.concatenate((self.MAP.mapmin, [-self.target_speed_limit, -self.target_speed_limit])),
                                np.concatenate((self.MAP.mapmax, [self.target_speed_limit, self.target_speed_limit]))]
        # RL state, (r_belief, theta_belief, r_dot_belief, theta_dot_belief, log_det_cov, observed) for each target
        self.num_target_dep_vars = 6  # 观测中和目标有关的量
        self.num_target_indep_vars = 0  # 观测中和目标无关的量
        self.limit['observation'] = [np.array([0.0, -np.pi, -rel_speed_limit, -10*np.pi, -50.0, 0.0]*self.num_targets),
                                     np.array([600.0, np.pi, rel_speed_limit, 10*np.pi,  50.0, 2.0]*self.num_targets)]
        self.observation_space = spaces.Box(
            self.limit['observation'][0], self.limit['observation'][1], dtype=np.float32)
        assert(len(self.limit['observation'][0]) == (self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def initialize_models(self):
        """初始化模型
        """
        # 生成初始状态
        init_pose = self._get_init_pose_random()

        # 重置智能体、目标、beliefs 的初始状态
        for i in range(self.num_agents):
            self.agents[i].set_state(init_pose['agents'][i])

        for i in range(self.num_targets):
            self.targets[i].set_state(np.concatenate((init_pose['targets'][i][:2],
                    METADATA.get('target_init_vel', [0.0, 0.0]))))
        
        for i in range(self.num_agents):
            for j in range(self.num_targets):
                self.belief_targets[i, j].reset(
                    init_state=np.concatenate((init_pose['belief'][i, j][:2], np.zeros(2))),
                    init_cov=METADATA.get('target_init_cov', 30.0))

    def get_observation(self):
        """计算每个智能体的观测

        Return
        ------
        obs_n : [observation]
            观测列表
        """
        # 观测并更新置信
        observed = self._observe_and_update_belief()

        obs_n = []
        for i, agent in enumerate(self.agents):
            obs = []
            for j, target in enumerate(self.targets):
                # 距离与方位的置信
                r_b, alpha_b = util.relative_distance_polar(
                    self.belief_targets[i, j].state[:2],
                    xy_base=agent.state[:2],
                    theta_base=agent.state[2])

                # 相对线速度和角速度置信
                r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                    self.belief_targets[i, j].state[:2],
                    self.belief_targets[i, j].state[2:],
                    agent.state[:2], agent.state[2],
                    self.last_actions[i][0], self.last_actions[i][1])
                
                # log det 和是否观测到目标
                obs.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                      np.log(LA.det(self.belief_targets[i, j].cov)),
                                      float(observed[i, j])])
            obs = np.array(obs)
            obs_n.append(obs)
        
        return obs_n
