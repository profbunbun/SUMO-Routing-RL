
from sumo_mmrl import PERAgent, Env
from .net_parser import NetParser



def create_env(config):

    path = config['training_settings']['experiment_path']
    sumo_config_path = path + config['training_settings']['sumoconfig']
    parser = NetParser(sumo_config_path)
    edge_locations = (
            parser.get_edge_pos_dic()
        )  

    out_dict = parser.get_out_dic()
    index_dict = parser.get_edge_index()

    return Env(config, edge_locations,  out_dict, index_dict)


def create_agent(config):
  
    experiment_path = config['training_settings']['experiment_path']
    learning_rate = config['agent_hyperparameters']['learning_rate']
    gamma = config['agent_hyperparameters']['gamma']
    epsilon_decay = config['agent_hyperparameters']['epsilon_decay']
    batch_size = config['agent_hyperparameters']['batch_size']
    memory_size = config['agent_hyperparameters']['memory_size']
    epsilon_max = config['agent_hyperparameters']['epsilon_max']
    epsilon_min = config['agent_hyperparameters']['epsilon_min']

    return PERAgent(19, 6, experiment_path,
                    learning_rate,
                    gamma, 
                    epsilon_decay, 
                    epsilon_max, 
                    epsilon_min, 
                    memory_size, 
                    batch_size,)

def create_PPOagent(config):
  
    experiment_path = config['training_settings']['experiment_path']
    learning_rate = config['agent_hyperparameters']['learning_rate']
    gamma = config['agent_hyperparameters']['gamma']
    epsilon_decay = config['agent_hyperparameters']['epsilon_decay']
    batch_size = config['agent_hyperparameters']['batch_size']
    memory_size = config['agent_hyperparameters']['memory_size']
    epsilon_max = config['agent_hyperparameters']['epsilon_max']
    epsilon_min = config['agent_hyperparameters']['epsilon_min']

    return PERAgent(19, 6, experiment_path,
                    learning_rate,
                    gamma, 
                    epsilon_decay, 
                    epsilon_max, 
                    epsilon_min, 
                    memory_size, 
                    batch_size,)





