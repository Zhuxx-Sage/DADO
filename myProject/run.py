import copy
import numpy as np
import os
from environments.TrafficEnvironment import TrafficEnvironment
from Utils import *
from Brain import Brain
from A2C import Net
from RolloutStorage import RolloutStorage
import torch
from config import Config

config = Config()


class Environment:
    '''主要运行：'''

    def run(self):
        # 为同时执行的环境数生成envs
        envs = [config.ENV for i in range(config.NUM_PROCESSES)]
        # 生成所有Agent的共享的脑Brain
        n_in = envs[0].observation_space  # 状态数量是4*361
        n_out = envs[0].action_space  # 动作数量是1
        # 加载模型

        # 生成存储变量
        obs_shape = 10 * n_in
        current_obs = torch.zeros(config.NUM_PROCESSES, obs_shape).to(config.device)  # 当前agent的状态
        rollouts = RolloutStorage(config.NUM_ADVANCED_STEP, config.NUM_PROCESSES, obs_shape)  # rollouts对象
        episode_rewards = torch.zeros([config.NUM_PROCESSES, 1])  # 保存当前试验的奖励
        final_rewards = torch.zeros([config.NUM_PROCESSES, 1])  # 保存最后试验的奖励
        obs_np = np.zeros([config.NUM_PROCESSES, obs_shape])
        reward_np = np.zeros([config.NUM_PROCESSES, 1])
        done_np = np.zeros([config.NUM_PROCESSES, 1])
        each_step = np.zeros(config.NUM_PROCESSES)  # 记录每个环境中的step数
        episode = 0  # 回合数

        # 初始状态
        obs = [envs[i].reset() for i in range(config.NUM_PROCESSES)]
        # print(obs[0][0].sum(axis=(0,1)))
        obs = np.array(obs).flatten()
        obs = torch.from_numpy(obs).float()
        current_obs = obs  # 存储最新的obs

        # 将当前状态保存到对象rollouts的第一个状态进行advantage学习
        rollouts.observations[0].copy_(current_obs)

        actor_critic_list = {}
        global_brain_list = {}

        # 初始化神经网络及大脑列表
        for initI in range(config.ENV.step_nums):
            actor_critic = Net(n_in, n_out).to(config.device)  # 到哪个阶段就加载哪个参数
            brain = Brain(actor_critic)
            actor_critic_list[initI] = actor_critic
            global_brain_list[initI] = brain

        # 运行循环
        # for j in range(config.NUM_EPISODES * config.NUM_PROCESSES):
        while episode < (config.NUM_EPISODES * config.NUM_PROCESSES):
            # 计算advanced学习的每个step数
            # print("Episode " + episode.__str__() + " is start!")
            # for step_t in range(config.ENV.step_nums):  # 对每个时间步
            #     actor_critic.load_state_dict(torch.load('parameters/A2C_parameters_step_' + (step_t + 1).str + '.pkl'))
            #     global_brain = Brain(actor_critic)
            step_t = 0
            brain = None
            ac = None
            while (step_t < config.ENV.step_nums):
                for step in range(config.NUM_ADVANCED_STEP):
                    # 求取动作
                    # actor_critic = Net(n_in, n_out).to(config.device)
                    # 判断一下路径是否存在
                    flag = os.path.exists(
                        'parameters/A2C_parameters_step_' + (step_t).__str__() + '.pkl')  # True/False
                    if flag:
                        ac = actor_critic_list[step_t]
                        ac.load_state_dict(
                            torch.load('parameters/A2C_parameters_step_' + (step_t).__str__() + '.pkl'))
                    else:
                        ac = actor_critic_list[step_t]
                    brain = global_brain_list[step_t]
                    # print(step_t)
                    # for pa in actor_critic.parameters():
                    #     print(pa)

                    with torch.no_grad():
                        # 求动作
                        action = actor_critic.act(rollouts.observations[step])
                        step_t += 1
                    actions = torch.clamp(action, 0, 6).cpu().numpy()
                    # tensor到numpy
                    # actions = action.squeeze(1).numpy()

                    # 运行1步
                    for i in range(config.NUM_PROCESSES):
                        obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(actions)
                        # 判断当前回合是够终止以及是否有下一个状态
                        if done_np[i]:  # 若终止
                            # if i == 0:  # 仅在初始环境输出
                            # print("%d Episode: Finished after %d steps" % (episode, each_step[i] + 1))
                            print("Episode " + episode.__str__() + " is end!")
                            print('Epoch {}, loss:{:.4f}'.format(episode + 1, brain.total_loss.sum()))
                            config.writer.add_scalar('Train/loss', brain.total_loss.sum(), episode)
                            # 保存模型
                            episode += 1
                            each_step[i] = 0
                            obs_np[i] = envs[i].reset().flatten()
                        else:
                            each_step[i] += 1

                        # 将奖励转换为tensor并添加到试验总奖励中
                    reward = torch.from_numpy(reward_np).float()
                    episode_rewards += reward
                    # 对每个执行环境，如果done，将mask置为0，如果继续，则设置为1
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])
                    # 更新最后一次试验的总奖励
                    final_rewards *= masks  # 若正在进行，乘1保持原样，否则重置为0
                    # 如果完成，重置为0
                    final_rewards += (1 - masks) * episode_rewards

                    # 更新试验的总奖励
                    episode_rewards *= masks
                    masks = masks.to(config.device)

                    current_obs = current_obs.view(1, -1).to(config.device) * masks
                    # 更新current_obs
                    obs = torch.from_numpy(obs_np).float()
                    current_obs = obs  # 存储最新obs

                    rollouts.insert(current_obs, torch.Tensor(actions), reward, masks)

                    # 保存参数
                    torch.save(actor_critic.state_dict(),
                               'parameters/A2C_parameters_step_' + (step_t).__str__() + '.pkl')

                # 从advanced的最终step开始计算状态价值
                with torch.no_grad():
                    next_value = actor_critic.get_value(rollouts.observations[-1]).detach()

                # 计算所有步骤的折扣奖励总和并更新rollouts的变量returns
                # print(next_value.shape) # torch.size[10,1]
                next_value = next_value.view(1, -1)
                rollouts.compute_returns(next_value)  # torch.size[1,10]
                # 网络和rollouts的更新
                brain.update(rollouts)
                rollouts.after_update()


if __name__ == '__main__':
    en = Environment()
    en.run()
