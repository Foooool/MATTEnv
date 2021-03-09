from gym.wrappers import TimeLimit
from ttenv.multiagent_target_tracking import MultiAgentTargetTrackingEnv1


def make(env_name,
         num_targets=1,
         num_agents=1,
         render=False,
         figID=0,
         record=False,
         ros=False,
         directory='',
         max_episode_steps=1000,
         seed=None,
         **kwargs):
    """创建环境

    Parameters
    ----------
    env_name : str
        name of an environment. (e.g. 'TargetTracking-v0')
    render : bool
        wether to render.
    figID : int
        figure ID for rendering and/or recording.
    record : bool
        whether to record a video.
    ros : bool
        whether to use ROS.
    directory :str
        a path to store a video file if record is True.
    T_steps : int
        the number of steps per episode.
    num_targets : int
        the number of targets

    Return
    ------
    Gym.Env
    """
    local_view = 0
    if env_name == 'MultiAgentTargetTracking-v1':
        env0 = MultiAgentTargetTrackingEnv1(num_agents=num_agents,
                                            num_targets=num_targets,
                                            seed=seed,
                                            **kwargs)
    # elif env_name == 'TargetTracking-info1':
    #     from ttenv.infoplanner_python.target_tracking_infoplanner import TargetTrackingInfoPlanner1
    #     env0 = TargetTrackingInfoPlanner1(num_targets=num_targets, **kwargs)
    # elif env_name == 'TargetTracking-info2':
    #     from ttenv.infoplanner_python.target_tracking_infoplanner import TargetTrackingInfoPlanner2
    #     env0 = TargetTrackingInfoPlanner2(num_targets=num_targets, **kwargs)
    else:
        raise ValueError('没有这个环境')

    # Wrappers
    # 每个 episode 最大步数 wrapper
    env = TimeLimit(env0, max_episode_steps=max_episode_steps)

    if ros:
        from ttenv.ros_wrapper import Ros
        env = Ros(env)

    if render:
        from ttenv.display_wrapper import Display2D
        env = Display2D(env, figID=figID, local_view=local_view)
        
    if record:
        from ttenv.display_wrapper import Video2D
        env = Video2D(env, dirname=directory, local_view=local_view)

    return env
