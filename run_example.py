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
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
# args = parser.parse_args(['--render', '1'])


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
                    )

    for _ in range(args.repeat):  # for each episode
        nlogdetcov = []
        env.reset()
        done = False
        while not done:
            # 渲染
            if args.render:
                env.render()
            
            # 执行动作
            obs, rew, done, info = env.step([env.action_space.sample() for _ in range(args.nb_agents)])

            # 记录平均 log det 协方差
            nlogdetcov.append(info['mean_nlogdetcov'])

        print("Sum of negative logdet of the target belief covariances : {:.2f}".format(np.sum(nlogdetcov)))

if __name__ == "__main__":
    main()
