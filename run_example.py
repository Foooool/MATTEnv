import mattenv
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', type=str, default='MultiAgentTargetTracking-v1')
parser.add_argument('--render', help='whether to render', type=int, default=0)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--ros', help='whether to use ROS', type=int, default=0)
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=2)
parser.add_argument('--nb_agents', help='the number of agents', type=int, default=2)
parser.add_argument('--log_dir', help='a path to a directory to log your data', type=str, default='.')
parser.add_argument('--map', type=str, default="empty")
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--im_size', type=int, default=28)
parser.add_argument('--max_episode_steps', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
# args = parser.parse_args()
args = parser.parse_args(['--render', '1'])


def main():
    # 创建环境
    env = mattenv.make(args.env,
                    render=args.render,
                    record=args.record,
                    ros=args.ros,
                    map_name=args.map,
                    directory=args.log_dir,
                    num_targets=args.nb_targets,
                    num_agents=args.nb_agents,
                    is_training=False,
                    im_size=args.im_size,
                    max_episode_steps=args.max_episode_steps
                    )

    for _ in range(args.repeat):  # for each episode
        env.reset()
        done = False
        step_counter = 0
        while not done and step_counter < 1000:
            # 渲染
            if args.render:
                env.render()
            
            # 执行动作
            _, rew, done, _ = env.step([env.action_space.sample() for _ in range(args.nb_agents)])

            # reward
            print('Step {}, reward {}'.format(step_counter, rew))

            step_counter += 1

if __name__ == "__main__":
    main()
