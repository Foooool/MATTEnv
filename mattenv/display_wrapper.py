import os
from random import shuffle

import matplotlib
from matplotlib import animation
from matplotlib import patches
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from gym import Wrapper

from mattenv.metadata import METADATA


matplotlib.use('TkAgg')


class Display2D(Wrapper):
    def __init__(self, env, skip=1, confidence=0.95, local_view=0):
        super(Display2D, self).__init__(env)
        self.mapmin = self.env.MAP.mapmin
        self.mapmax = self.env.MAP.mapmax
        self.mapres = self.env.MAP.mapres

        # Figure 对象
        self.figID = 0
        self.fig = plt.figure(self.figID, figsize=(8, 6))
        # Local view
        self.local_view = local_view
        if local_view:
            self.local_view_fig = plt.figure(self.figID + 1)
            self.local_idx_map = [(1, 1), (1, 0), (1, 2), (0, 1), (2, 1)]
        
        self.n_frames = 0 # 当前帧数
        self.skip = skip  # 帧率
        self.c_cf = np.sqrt(-2 * np.log(1 - confidence)) / 10
        self.traj_num = 0

        # 颜色列表
        self.colors = [key for key, _ in mcolors.CSS4_COLORS.items()]
        shuffle(self.colors)

    def close(self):
        plt.close(self.fig)
        if self.local_view:
            plt.close(self.local_view_fig)

    def step(self, actions):
        # 目标真实位置
        target_true_pos = [target.state[:2] for target in self.env.targets]

        # 记录智能体轨迹
        for i in range(self.env.num_agents):
            self.traj[i][0].append(self.env.agents[i].state[0])
            self.traj[i][1].append(self.env.agents[i].state[1])

        # 记录目标轨迹
        for i in range(self.env.num_targets):
            self.traj_y[i][0].append(target_true_pos[i][0])
            self.traj_y[i][1].append(target_true_pos[i][1])
        return self.env.step(actions)

    def render(self, record=False):
        # 智能体当前状态
        states = [agent.state for agent in self.env.agents]

        # 目标真实位置
        num_targets = self.env.num_targets
        target_true_pos = [target.state[:2] for target in self.env.targets]
        
        # 显示帧率
        if self.n_frames % self.skip == 0:
            # 清空画布
            self.fig.clf()
            ax = self.fig.subplots()
            if self.local_view:
                self.local_view_fig.clf()
                if self.local_view == 1:
                    local_view_ax = self.local_view_fig.subplots()
                elif self.local_view == 5:
                    local_view_ax = self.local_view_fig.subplots(3, 3)
                    [[local_view_ax[r][c].set_aspect('equal', 'box') for r in range(3)] for c in range(3)]
                else:
                    raise ValueError('Display2D 错误：localview 应该在 [0, 1, 5] 中选择，当前为 {}'.format(self.local_view))

            # 绘制背景
            if self.env.MAP.visit_freq_map is not None:
                background_map = self.env.MAP.visit_freq_map.T
                if self.env.MAP.map is not None:
                    background_map += 2 * self.env.MAP.map
            else:
                if self.env.MAP.map is not None:
                    background_map = 2 * self.env.MAP.map
                else:
                    background_map = np.zeros(self.env.MAP.mapdim)
            ax.imshow(background_map, cmap='gray_r', origin='lower',
                      vmin=0, vmax=2, extent=[self.mapmin[0], self.mapmax[0],
                                              self.mapmin[1], self.mapmax[1]])
            
            # 绘制智能体与置信
            for i, state in enumerate(states):
                # 智能体以蓝色三角形表示
                ax.plot(state[0], state[1], marker=(3, 0, state[2] / np.pi * 180 - 90),
                        markersize=10, linestyle='None', markerfacecolor='b',
                        markeredgecolor='b')
                # 智能体轨迹，蓝色小点
                ax.plot(self.traj[i][0], self.traj[i][1], 'b.', markersize=2)

                # 传感器观测范围，灰色空心扇形
                sensor_arc = patches.Arc((state[0], state[1]), METADATA['sensor_r']*2, METADATA['sensor_r']*2,
                                         angle=state[2]/np.pi*180, theta1=-METADATA['fov']/2, theta2=METADATA['fov']/2, facecolor='gray')
                ax.add_patch(sensor_arc)
                ax.plot([state[0], state[0]+METADATA['sensor_r']*np.cos(state[2]+0.5*METADATA['fov']/180.0*np.pi)],
                        [state[1], state[1]+METADATA['sensor_r']*np.sin(state[2]+0.5*METADATA['fov']/180.0*np.pi)], 'k', linewidth=0.5)
                ax.plot([state[0], state[0]+METADATA['sensor_r']*np.cos(state[2]-0.5*METADATA['fov']/180.0*np.pi)],
                        [state[1], state[1]+METADATA['sensor_r']*np.sin(state[2]-0.5*METADATA['fov']/180.0*np.pi)], 'k', linewidth=0.5)

                # 置信
                for j in range(num_targets):
                    # 位置置信，中心为置信中心，半轴为协方差的特征值，同一种颜色属于同一个智能体
                    eig_val, eig_vec = LA.eig(
                        self.env.belief_targets[i, j].cov[:2, :2])
                    belief_target = patches.Ellipse(
                        (self.env.belief_targets[i, j].state[0],
                         self.env.belief_targets[i, j].state[1]),
                        2*np.sqrt(eig_val[0])*self.c_cf,
                        2*np.sqrt(eig_val[1])*self.c_cf,
                        angle=180/np.pi*np.arctan2(np.real(eig_vec[0][1]),
                                                   np.real(eig_vec[0][0])), fill=True, zorder=2,
                        facecolor=self.colors[i], alpha=0.5)
                    ax.add_patch(belief_target)

                    # # 关于速度的 belief
                    # if target_cov[i].shape[0] == 4:
                    #     eig_val, eig_vec = LA.eig(target_cov[i][2:, 2:])
                    #     belief_target_vel = patches.Ellipse(
                    #         (target_b_state[i][0], target_b_state[i][1]),
                    #         2*np.sqrt(eig_val[0])*self.c_cf,
                    #         2*np.sqrt(eig_val[1])*self.c_cf,
                    #         angle=180/np.pi*np.arctan2(np.real(eig_vec[0][1]),
                    #                                 np.real(eig_vec[0][0])), fill=True, zorder=2,
                    #         facecolor='m', alpha=0.5)
                    #     ax.add_patch(belief_target_vel)

                    # 强调置信中心，空心小圆
                    ax.plot(self.env.belief_targets[i, j].state[0], self.env.belief_targets[i, j].state[1], marker='o',
                            markersize=5, linewidth=4, markerfacecolor='none',
                            markeredgecolor=self.colors[i])

            # 目标轨迹及真实目标
            for i in range(num_targets):
                # 目标轨迹：红色小点
                ax.plot(self.traj_y[i][0], self.traj_y[i][1], 'r.', markersize=2)

                # 目标当前位置：红色圆圈
                ax.plot(target_true_pos[i][0], target_true_pos[i][1], marker='o',
                        markersize=5, linestyle='None', markerfacecolor='r',
                        markeredgecolor='r')
            
            # 在右边显示速度
            obj_counter = 1
            for i in range(self.env.num_targets):
                ax.text(self.mapmax[0]+1., self.mapmax[1]-5*obj_counter,
                        'Target {}: {:.2f}'.format(i+1, np.sqrt(np.sum(self.env.targets[i].state[2:]**2))))
                obj_counter += 1
            for i in range(self.env.num_agents):
                ax.text(self.mapmax[0]+1., self.mapmax[1]-5*obj_counter,
                        'Agent {}: {:.2f}'.format(i+1, self.env.last_actions[i][0]))
                obj_counter += 1
            ax.set_xlim((self.mapmin[0], self.mapmax[0]))
            ax.set_ylim((self.mapmin[1], self.mapmax[1]))
            ax.set_title("Trajectory {}".format(self.traj_num))
            ax.set_aspect('equal', 'box')
            ax.grid()

            # 局部视野
            if self.local_view:
                im_size = self.env.im_size
                for j in range(self.local_view):
                    local_rect = patches.Rectangle(
                        self.env.local_mapmin_g[j],
                        width=im_size*self.mapres[0],
                        height=im_size*self.mapres[0],
                        angle=(state[2]-np.pi/2)/np.pi*180,
                        fill=False, edgecolor='b')
                    ax.add_patch(local_rect)

            if self.local_view == 1:
                local_mapmin = np.array([-im_size/2*self.mapres[0], 0.0])
                local_view_ax.imshow(
                    np.reshape(self.env.local_map[0], (im_size, im_size)),
                    cmap='gray_r', origin='lower', vmin=-1, vmax=1,
                    extent=[local_mapmin[0], -local_mapmin[0],
                            0.0, -local_mapmin[0]*2])
            elif self.local_view == 5:
                local_mapmin = np.array([-im_size/2*self.mapres[0], 0.0])
                [local_view_ax[self.local_idx_map[j][0]][self.local_idx_map[j][1]].imshow(
                    np.reshape(self.env.local_map[j], (im_size, im_size)),
                    cmap='gray_r', origin='lower', vmin=-1, vmax=1,
                    extent=[local_mapmin[0], -local_mapmin[0],
                            0.0, -local_mapmin[0]*2]) for j in range(self.local_view)]
            
            if not record:
                plt.draw()
                plt.pause(0.0001)

        self.n_frames += 1

    def reset(self):
        self.traj_num += 1
        self.traj = [[[], []]] * self.env.num_agents  # 智能体轨迹
        self.traj_y = [[[], []]] * self.env.num_targets  # 目标轨迹
        return self.env.reset()


class Video2D(Wrapper):
    def __init__(self, env, dirname='', skip=1, dpi=80, local_view=0):
        super(Video2D, self).__init__(env)
        self.local_view = local_view
        self.skip = skip
        self.moviewriter = animation.FFMpegWriter()
        fnum = np.random.randint(0, 1000)
        fname = os.path.join(dirname, 'train_%d.mp4' % fnum)
        self.moviewriter.setup(fig=env.fig, outfile=fname, dpi=dpi)
        if self.local_view:
            self.moviewriter0 = animation.FFMpegWriter()
            self.moviewriter0.setup(fig=env.fig0,
                                    outfile=os.path.join(
                                        dirname, 'train_%d_local.mp4' % fnum),
                                    dpi=dpi)
        self.n_frames = 0

    def render(self, *args, **kwargs):
        if self.n_frames % self.skip == 0:
            # if traj_num % self.skip == 0:
            self.env.render(record=True, *args, **kwargs)
        self.moviewriter.grab_frame()
        if self.local_view:
            self.moviewriter0.grab_frame()
        self.n_frames += 1

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def finish(self):
        self.moviewriter.finish()
        if self.local_view:
            self.moviewriter0.finish()
