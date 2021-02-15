# Multi-Agent Target Tracking Environment (OpenAI gym framework)
多智能体目标追踪环境，基于 OpenAI gym 框架开发，主要参考 [ttenv](https://github.com/coco66/ttenv.git) 环境，将其扩展到多智能体环境。

## 安装
* Python 3.x
安装 OpenAI gym (http://gym.openai.com/docs/)
```
pip install gym
```
克隆该仓库
```
git clone https://github.com/Foooool/MATTEnv.git
```
`run_example.py` 展示了该环境的基本用法

```
python run_example.py --render 1
```

## 实验环境
* MultiAgentTargetTracking-v1 : 双重积分目标模型，卡尔曼滤波。
目标采用双重积分动力学模型，使用卡尔曼滤波作为估计器。每个目标的状态为 $ (x,y,\dot x, \dot y) $ 。


## Metadata
Some metadata examples for the environment setup are presented in ttenv/ttenv/metatdata.py.

## 代码结构

+ init.py: 环境接口
  + make 环境接口函数
+ agent_model: 动力学模型
  + Agent
  + AgentDoubleInt2D
  + AgentDoubleInt2D_Nonlinear
  + AgentSE2
  + Agent2DFixedPath
  + SE2Dynamics
+ base: 环境基类
  + MultiAgentTargetTrackingBase
    + reset
    + step
    + get_init_pose
    + gen_rand_pose
    + get_init_pose_random
    + add_history_to_state
    + set_target_path
    + observation
    + observation_noise
    + observe_and_update_belief
    + get_reward
  + sharing_reward_fun: 所有智能体共享的回报函数
+ belief_tracker: 滤波器
  + KFbelief: 卡尔曼滤波
  + UKFbelief: 无迹卡尔曼滤波
+ display_wrapper: 渲染包装器
  + Display2D
  + Video2D
+ metadata: 基本设定
+ polices:  一些简单策略
  + RandomPolicy
+ multiagent_target_tracking: 环境
  + MultiAgentTargetTrackingEnv1 : 二次积分目标( $ x, y,\dot x, \dot y$) + KF
+ utils:  工具函数

## Running with RL
Examples of learning a deep reinforcement learning policy can be found in the ADFQ repository (https://github.com/coco66/ADFQ).
* DQN : ADFQ/deep_adfq/baselines0/deepq/run_tracking.py
* Double DQN : ADFQ/deep_adfq/baselines0/deepq/run_tracking.py --double_q=1
* Deep ADFQ : ADFQ/deep_adfq/run_tracking.py
