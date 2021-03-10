"""智能体动力学模型

包括以下模型：

    AgentDoubleInt2D : Double Integrator Model in 2D
                        state: x,y,xdot,ydot
    AgentDoubleInt2D_Nonlinear : Double Integrator Model with non-linear term for obstalce avoidance in 2D
                        state: x,y,xdot,ydot
    AgentSE2 : SE2 Model
           state x,y,theta

    Agent2DFixedPath : Model with a pre-defined path
"""

import numpy as np
from mattenv.metadata import METADATA
import mattenv.util as util


class Agent(object):
    def __init__(self,
                 init_state,
                 dim,
                 sampling_period,
                 limit,
                 collision_func,
                 margin=METADATA['margin']):
        """智能体基类

        Parameters
        ----------
        init_state : ndarray of shape (dim, )
            智能体初始状态
        dim : int
            智能体状态维度
        sampling_period : float
            采样间隔
        limit : [[low], [high]]
            状态的上下界
        collision_func : callable
            碰撞检测函数
        margin : float, optional
            智能体与目标的间隔距离, by default METADATA['margin']
        """
        self.state = np.array(init_state)
        self.dim = dim
        self.sampling_period = sampling_period
        self.limit = limit
        self.collision_func = collision_func
        self.margin = margin

        # 状态合法性检查
        self.range_check()

    def range_check(self):
        """将智能体状态裁剪到范围内
        """
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def collision_check(self, pos):
        """碰撞检测

        Parameters
        ----------
        pos : (x, y)
            待检查的位置

        Return
        ------
        is_col : bool
            是否碰撞
        """
        return self.collision_func(pos)

    def margin_check(self, pos, target_pos):
        return any(np.sqrt(np.sum((pos - target_pos)**2, axis=1)) < self.margin)

    def set_state(self, new_state):
        """设置智能体状态，如果新状态不在范围内，会自动进行规整

        Parameters
        ----------
        new_state : ndarray of shape (dim, )
            新状态
        """
        assert len(new_state) == self.dim, '位置定义与维度不符'
        self.state = np.array(new_state)
        self.range_check()
        return self.state


class AgentDoubleInt2D(Agent):
    def __init__(self, init_state, sampling_period, limit, collision_func,
                    margin=METADATA['margin'], A=None, W=None):
        """二次积分模型 $ (x, y, x\dot, y\dot) $

        状态更新方程 $ s_new = As + w, w \sim N(0, W) $

        Parameters
        ----------
        init_state : ndarray of shape (4, )
            初始状态
        sampling_period : float
            采样间隔
        limit : [[low], [high]]
            状态的上下界
        collision_func : callable
            碰撞检测函数
        margin : float, optional
            智能体与目标的间隔距离, by default METADATA['margin']
        A : ndarray of shape (4, 4) | None, optional
            线性项系数，默认单位矩阵
        W : ndarray of shape (4, 4) | None, optional
            噪声协方差矩阵, by default None
        """
        Agent.__init__(self, init_state, 4, sampling_period, limit, collision_func, margin=margin)
        self.A = np.eye(self.dim) if A is None else A
        self.W = W

    def update(self, agent_pos=None):
        """状态更新

        Returns
        -------
        is_col : bool
            是否碰撞
        """
        # 状态更新
        new_state = np.matmul(self.A, self.state)
        if self.W is not None:
            noise_sample = np.random.multivariate_normal(np.zeros(self.dim,), self.W)
            new_state += noise_sample

        # 碰撞检测
        is_col = False
        if self.collision_check(new_state[:2]):
            # 如果碰撞，位置不变
            is_col = True
            new_state[:2] = self.state[:2]

        self.state = new_state

        # 范围检查
        self.range_check()
        return is_col

    def range_check(self):
        """状态范围检查，将位置裁剪到大小限制中，将速度的大小缩放到最大速度 limit[1][2] 上
        """
        # 位置
        self.state[:2] = np.clip(self.state[:2], self.limit[0][:2], self.limit[1][:2])

        # 速度
        v_square = np.sum(self.state[2:]**2)
        if v_square > self.limit[1][2]**2:
            factor = np.sqrt(self.limit[1][2] / v_square)
            self.state[2] *= factor
            self.state[3] *= factor


class AgentDoubleInt2D_Nonlinear(AgentDoubleInt2D):
    def __init__(self, init_state, sampling_period, limit, collision_func,
                    margin=METADATA['margin'], A=None, W=None, obs_check_func=None):
        """非线性二次积分模型 $ (x, y, x\dot, y\dot) $

        状态更新方程 $ s_new = As + w, w \sim N(0, W) $

        Parameters
        ----------
        init_state : ndarray of shape (4, )
            初始状态
        sampling_period : float
            采样间隔
        limit : [[low], [high]]
            状态的上下界, dim=3
        collision_func : callable
            碰撞检测函数
        margin : float, optional
            智能体与目标的间隔距离, by default METADATA['margin']
        A : ndarray of shape (4, 4) | None, optional
            线性项系数，默认单位矩阵
        W : ndarray of shape (4, 4) | None, optional
            噪声协方差矩阵, by default None
        obs_check_func : callable, optional
            障碍物检测函数，获取最近的障碍物, by default None
        """
        AgentDoubleInt2D.__init__(self, init_state, sampling_period, limit,
            collision_func, margin=margin, A=A, W=W)
        self.obs_check_func = obs_check_func

    def update(self, agent_pos=None):
        # 状态更新
        new_state = np.matmul(self.A, self.state)
        if self.W is not None:
            noise_sample = np.random.multivariate_normal(np.zeros(self.dim,), self.W)
            new_state += noise_sample

        # 碰撞检测
        is_col = False
        if self.collision_check(new_state[:2]):
            new_state = self.collision_control()
            is_col = True

        # 非线性项，规避障碍物
        if self.obs_check_func is not None:
            del_vx, del_vy = self.obstacle_detour_maneuver(
                    r_margin=METADATA['target_speed_limit']*self.sampling_period*2)
            new_state[2] += del_vx
            new_state[3] += del_vy

        self.state = new_state
        self.range_check()
        return is_col

    def range_check(self):
        """状态范围检查，将位置裁剪到大小限制中，将速度的大小缩放到最大速度 limit[1][2] 上
        """
        # 位置
        self.state[:2] = np.clip(self.state[:2], self.limit[0][:2], self.limit[1][:2])

        # 速度
        v_square = np.sum(self.state[2:]**2)
        if v_square > self.limit[1][2]**2:
            factor = np.sqrt(self.limit[1][2] / v_square)
            self.state[2] *= factor
            self.state[3] *= factor

    def collision_control(self):
        """碰撞控制函数，返回一个新速度使得智能体远离最近的障碍物 with an angle (pi/2, pi)
        """
        odom = [self.state[0], self.state[1], np.arctan2(self.state[3], self.state[2])]

        # 障碍物位置
        obs_pos = self.obs_check_func(odom)
        v = np.sqrt(np.sum(np.square(self.state[2:]))) + np.random.normal(0.0, 1.0)
        if obs_pos[1] >= 0:
            th = obs_pos[1] - (1 + np.random.random()) * np.pi/2
        else:
            th = obs_pos[1] + (1 + np.random.random()) * np.pi/2

        state = np.array([self.state[0], self.state[1], v * np.cos(th + odom[2]), v * np.sin(th + odom[2])])
        return state

    def obstacle_detour_maneuver(self, r_margin=1.0):
        """
        Returns del_vx, del_vy which will be added to the new state.
        This provides a repultive force from the closest obstacle point based
        on the current velocity, a linear distance, and an angular distance.

        Parameters:
        ----------
        r_margin : float. A margin from an obstalce that you want to consider
        as the minimum distance the target can get close to the obstacle.
        """
        odom = [self.state[0], self.state[1], np.arctan2(self.state[3], self.state[2])]
        # 最近的障碍物
        obs_pos = self.obs_check_func(odom)

        speed = np.sqrt(np.sum(self.state[2:]**2))

        rot_ang = np.pi/2 * (1. + 1./(1. + np.exp(-(speed-0.5*METADATA['target_speed_limit']))))
        if obs_pos is not None:
            acc = max(0.0, speed * np.cos(obs_pos[1])) / max(METADATA['margin2wall'], obs_pos[0] - r_margin)
            th = obs_pos[1] - rot_ang if obs_pos[1] >= 0 else obs_pos[1] + rot_ang
            del_vx = acc * np.cos(th + odom[2]) * self.sampling_period
            del_vy = acc * np.sin(th + odom[2]) * self.sampling_period
            return del_vx, del_vy
        else:
            return 0., 0.


class AgentSE2(Agent):
    def __init__(self,
                 init_state,
                 sampling_period,
                 limit,
                 collision_func,
                 margin=METADATA['margin']):
        Agent.__init__(self, init_state, 3, sampling_period, limit, collision_func, margin=margin)

    def update(self, control_input, margin_pos=None):
        """状态更新

        Parameters
        ----------
        control_input : list. [linear_velocity, angular_velocity]
            控制输入，线速度和角速度

        Return
        ------
        is_col : bool
            是否碰撞
        """
        # 状态更新
        if self.dim == 3:
            new_state = SE2Dynamics(self.state, self.sampling_period, control_input)
        elif self.dim == 5:
            new_state = SE2DynamicsVel(self.state, self.sampling_period, control_input)
        
        # 碰撞检测
        is_col = False
        if self.collision_check(new_state[:2]):
            # 如果碰撞，位置保持不变
            is_col = True
            new_state[:2] = self.state[:2]
        
        self.state = new_state
        self.range_check()

        return is_col


class Agent2DFixedPath(Agent):
    """按照预定轨迹行动的智能体
    """
    def __init__(self, dim, sampling_period, limit, collision_func, path, margin=METADATA['margin']):
        Agent.__init__(self, path[0], dim, sampling_period, limit, collision_func, margin=margin)
        self.path = path
        self.t = 0

    def update(self):
        self.t = (self.t + 1) % len(self.path)
        self.state = self.path[self.t]
        
    def reset(self):
        self.t = 0


def SE2Dynamics(x, dt, u):
    """SE2 动力学模型的更新函数

    Parameters
    ----------
    x : array of shape (3, )
        智能体状态
    dt : float
        时间间隔
    u : array of shape (2, )
        线速度和角速度

    Returns
    -------
    array of shape (3, )
        新状态
    """
    assert(len(x)==3)
    # 角度偏移
    tw = dt * u[1]

    # 更新状态
    if abs(tw) < 0.001:
        diff = np.array([dt*u[0]*np.cos(x[2]+tw/2),
                         dt*u[0]*np.sin(x[2]+tw/2),
                         tw])
    else:
        diff = np.array([u[0]/u[1]*(np.sin(x[2]+tw) - np.sin(x[2])),
                         u[0]/u[1]*(np.cos(x[2]) - np.cos(x[2]+tw)),
                         tw])
    new_x = x + diff
    new_x[2] = util.wrap_around(new_x[2])
    return new_x


def SE2DynamicsVel(x, dt, u=None):
    """
    update dynamics function for contant linear and angular velocities
    """
    assert(len(x)==5) # x = [x,y,theta,v,w]
    if u is None:
        u = x[-2:]
    odom = SE2Dynamics(x[:3], dt, u)
    return np.concatenate((odom, u))
