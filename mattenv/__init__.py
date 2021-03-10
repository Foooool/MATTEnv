from gym.wrappers import TimeLimit
from mattenv.multiagent_target_tracking import MultiAgentTargetTrackingEnv1


def make(env_name,
         num_agents=1,
         num_targets=1,
         map_name='empty',
         seed=None,
         reward_function=None,
         known_noise=True,
         max_episode_steps=1000,
         render=False,
         figID=0,
         local_view=0,
         ros=False,
         record=False,
         record_dir='.',
         **kwargs):
    """创建环境

    Parameters
    ----------
    env_name : str
        环境名称
    num_agents : int, optional
        智能体个数, by default 1
    num_targets : int, optional
        目标个数, by default 1
    map_name : str, optional
        环境地图名称, by default 'empty'
    seed : int, optional
        随机数种子, by default None
    reward_function : callable, optional
        回报函数, by default None
    known_noise : bool, optional
        是否已知方差, by default True
    max_episode_steps : int, optional
        每个 episode 最大步数, by default 1000
    render : bool, optional
        是否渲染, by default False
    figID : int, optional
        figid, by default 0
    local_view : int, optional
        是否显示局部视野, by default 0
    ros : bool, optional
        是否 Ros, by default False
    record : bool, optional
        是否保存视频, by default False
    record_dir : str, optional
        视频保存目录, by default '.'

    Returns
    -------
    gym.Env
        环境对象
    """
    if env_name == 'MultiAgentTargetTracking-v1':
        env0 = MultiAgentTargetTrackingEnv1(num_agents=num_agents,
                                            num_targets=num_targets,
                                            map_name=map_name,
                                            seed=seed,
                                            reward_function=reward_function,
                                            known_noise=known_noise,
                                            **kwargs)
    else:
        raise ValueError('Env name 错误：没有这个环境')

    # Wrappers
    # 每个 episode 最大步数 wrapper
    env = TimeLimit(env0, max_episode_steps=max_episode_steps)

    if ros:
        from mattenv.ros_wrapper import Ros
        env = Ros(env)

    if render:
        from mattenv.display_wrapper import Display2D
        env = Display2D(env, figID=figID, local_view=local_view)
        
    if record:
        from mattenv.display_wrapper import Video2D
        env = Video2D(env, dirname=record_dir, local_view=local_view)

    return env
