import time
import numpy as np
from Agents.Neural_Network import NerualNetwork
from nn_builder.pytorch.NN import NN


class Base_Agent(object):

    def __init__(self, config):
        '''
        :param config: 配置文件
        '''
        # self.logger = self.setup_logger()  # 输出运行日志
        self.config = config  # 配置文件
        self.environment = config.environment  # 交通环境
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = config.action_size
        self.state_size = config.state_size
        self.hyperparameters = config.hyperparameters
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = 0

        self.episode_number = 0  # 当前回合
        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.global_step_number = 0
        self.turn_off_exploration = False

        self.neural_network = None

    def step(self):
        '''
        :return: 子类必须实现step类
        '''
        raise ValueError("Step needs to be implemented by the agent")

    def reset_game(self):
        '''
        :return: 为新的回合重置游戏
        '''
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = None
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_observations = []
        self.memory = []

    def track_episodes_data(self):
        '''
        :return: 保存最近回合的数据
        '''
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            if save_and_print_results: self.save_and_print_result()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def save_and_print_result(self):
        '''
        :return: 保存并打印游戏结果
        '''
        self.save_result()

    def save_result(self):
        '''
        :return: 保存一回合的游戏结果
        '''
        self.game_full_episode_scores.append(self.total_episode_score_so_far)

    def conduct_action(self, action):
        '''
        :param action: agent执行的动作
        :return:  对环境执行动作，环境进行转换
        '''
        self.next_state, self.reward, self.done, info = self.environment.step(action)
        self.total_episode_score_so_far += self.reward

    def enough_experience_to_learning_from(self):
        '''
        :return: 缓冲池是否有足够的经验去学习
        '''
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory=None, experience=None):
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss):
        network = [network]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 需要考虑
    def create_NN(self, input_dim, output_dim):
        model = NerualNetwork(input_dim, output_dim).to(self.device)
        return model



    # def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
    #     if hyperparameters is None: hyperparameters = self.hyperparameters
    #     if key_to_use: hyperparameters = hyperparameters[key_to_use]
    #     if override_seed:
    #         seed = override_seed
    #     else:
    #         seed = self.config.seed
    #
    #     default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
    #                                       "initialiser": "default", "batch_norm": False,
    #                                       "columns_of_data_to_be_embedded": [],
    #                                       "embedding_dimensions": [], "y_range": ()}
    #
    #     return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
    #               output_activation=hyperparameters["final_layer_activation"],
    #               batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
    #               hidden_activations=hyperparameters["hidden_activations"]).to(self.device)
    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero: from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())