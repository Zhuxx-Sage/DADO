import torch
from torch import optim
import torch.nn as nn

from config import Config
config = Config()

class Brain(object):

    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)
        self.total_loss = torch.Tensor(0)
        self.total_loss.to(config.device)

    def update(self, rollouts):
        '''对使用Advantage计算的所有5个步骤进行更新'''
        obs_shape = rollouts.observations.size()[2:]  # size:14440
        num_steps = config.NUM_ADVANCED_STEP
        num_processes = config.NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 14440),  # 状态 torch.size[5,1,14440] - > [5,14440]
            rollouts.actions.view(-1, 1)  # 动作 torch.size[5,1,10]  - > [5,10]
        )

        values = values.view(num_steps, num_processes, 10)

        action_log_probs = action_log_probs.view(num_steps, num_processes, 10)

        # advantage = v_t - value(critic)
        advantages = rollouts.returns[:-1] - values
        # critic_loss = advantage^2
        value_loss = advantages.pow(2).mean()
        # 计算actor的gain，作为loss
        action_gain = (action_log_probs * advantages.detach()).mean() + config.entropy_coef * entropy
        action_gain = - action_gain
        # 误差函数总和
        self.total_loss = (value_loss * config.value_loss_coef + action_gain)

        # 更新连接参数
        self.actor_critic.train()  # 在训练模式中
        self.optimizer.zero_grad()  # 重置梯度
        self.total_loss.backward(torch.ones_like(self.total_loss))  # 计算反向传播
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), config.max_grad_norm)  # 使梯度大小最大为0.5，以便连接参数不会一下改变太多

        self.optimizer.step()  # 更新连接参数


