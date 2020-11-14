from Utilities.Config import Config
from environments.TrafficEnvironment import *
from Agents.REINFORCE import REINFORCE
from Agents.A3C import A3C
from Agents.A2C import A2C
from Agents.PPO import PPO
# from Agents.A3C import *
from Trainer import Trainer
import Agents.Neural_Network

road_matrix = [[0, Edge(0, 0, 1, 4, 1), Edge(1, 0, 2, 5, 1),
                Edge(2, 0, 3, 3, 1)],
               [Edge(3, 1, 0, 8, 1), 0, 0, Edge(4, 1, 3, 5, 1)],
               [Edge(5, 2, 0, 6, 1), 0, 0, Edge(6, 2, 3, 2, 1)],
               [Edge(7, 3, 0, 7, 1), Edge(8, 3, 1, 4, 1),
                Edge(9, 3, 2, 12, 1), 0]]

urban = UrbanNetGraph(road_matrix, 4, 10)

config = Config()
config.seed = 1
config.environment = TrafficEnvironment(urban)
config.state_size = 4 * 361
config.action_size = 1
config.num_episodes_to_run = 100
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

config.hyperparameters = {
     "Policy_Gradient_Agents": {
            "learning_rate": 0.05,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.9,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 10,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": False
        },


    "Actor_Critic_Agents": {

        "learning_rate": 0.0005,
        "linear_hidden_units": [150, 30, 30, 30],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 25.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 10.0,
        "normalise_rewards": False,
        "automatically_tune_entropy_hyperparameter": True,
        "add_extra_noise": False,
        "min_steps_before_learning": 4,
        "do_evaluation_iterations": True,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.001,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.01,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "batch_size": 3,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "HER_sample_proportion": 0.8,
        "exploration_worker_difference": 1.0
    }
}

if __name__ == "__main__":
    AGENTS = [A2C]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
