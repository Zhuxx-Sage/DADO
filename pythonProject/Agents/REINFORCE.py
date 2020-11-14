import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from Agents.Base_Agent import Base_Agent
from Agents import Neural_Network



class REINFORCE(Base_Agent):
    agent_name = "REINFORCE"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.agent_name = "REINFORCE"
        self.policy = self.create_NN(config.state_size, config.action_size, config.learning_rate)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.hyperparameters["learning_rate"])
        self.episode_rewards = []
        self.episode_log_probabilities = []

    def reset_game(self):
        '''
        :return: 重新进行一回合
        '''
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = None
        self.total_episode_score_so_far = 0
        self.episode_rewards = []
        self.episode_log_probabilities = []
        self.episode_step_number = 0

    def step(self):
        '''
        :return: 对环境执行动作
        '''
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            # self.update_next_state_reward_done_and_score()
            self.store_reward()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state
            self.episode_step_number += 1
        self.episode_number += 1

    def pick_and_conduct_action_and_save_log_probabilities(self):
        '''
        :return: 选择并执行动作，并且保存对数概率
        '''
        action, log_probabilities = self.pick_action_and_get_log_probabilities()
        self.store_log_probabilities(log_probabilities)
        self.store_action(action)
        self.conduct_action()

    def pick_action_and_get_log_probabilities(self):
        '''
        :return: 选择动作并且得到对数概率
        '''
        # state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        state = torch.from_numpy(self.state).float().to(self.device)
        action_probabilities = self.policy.forward(state).cpu()
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def store_log_probabilities(self, log_probabilities):
        '''
        :return: 存储对数概率
        '''
        self.episode_log_probabilities.append(log_probabilities)

    def store_action(self, action):
        '''
        :param action: 选择的动作
        :return: 存储选择的动作
        '''
        self.action = action

    def store_reward(self):
        self.episode_rewards.append(self.reward)

    def actor_learn(self):
        '''
        :return:
        '''
        total_discounted_reward = self.calculate_episode_discounted_reward()
        policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def calculate_episode_discounted_reward(self):
        '''
        :return: 计算累积奖励
        '''
        discounts = self.config.hyperparameters["discount_rate"] ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)
        return total_discounted_reward

    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        '''
        :param total_discounted_reward:  总的累积奖励
        :return: 计算损失
        '''
        policy_loss = []
        for log_prob in self.episode_log_probabilities:
            policy_loss.append(-log_prob * total_discounted_reward)
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss

    def time_to_learn(self):
        return self.done
