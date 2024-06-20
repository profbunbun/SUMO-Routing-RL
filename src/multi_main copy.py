import argparse
from utils import Utils
from environment import sim_manager as so
import wandb




def parse_args():
    """
    Parse command-line arguments for hyperparameters and configurations.

    .. todo:: Determine what command line args are necesary for this project.
    """
    parser = argparse.ArgumentParser(description="Reinforcement Learning for SUMO Traffic Simulation")
    parser.add_argument('--config', type=str, default='src/configurations/config.yaml', help='Path to the configuration file')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate for the agent')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--epsilon_decay', type=float, default=None, help='Epsilon decay rate')
    parser.add_argument('--num_agents', type=int, default=1, help='Number of agents')
    return parser.parse_args()

def load_and_override_config(args):
    """
    Load configuration from a YAML file and override with command-line arguments if provided.
    """
    config = Utils.load_yaml_config(args.config)
    if args.episodes is not None:
        config['training_settings']['episodes'] = args.episodes
    if args.learning_rate is not None:
        config['agent_hyperparameters']['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['agent_hyperparameters']['batch_size'] = args.batch_size
    if args.epsilon_decay is not None:
        config['agent_hyperparameters']['epsilon_decay'] = args.epsilon_decay
    return config

def main_training_loop(config):
    """
    Main training loop for the reinforcement learning agent.


    .. todo:: Determine best usage of wandb for this project
    """
    env = so.create_env(config=config)
    dagent = so.create_agent(config=config)
    best_reward = float('-inf')

    for episode in range(config['training_settings']['episodes']):
        route_taken = []
        cumulative_reward = 0
        if (episode) % 1000 == 0:
            env.render("human")
        else:
            env.render()
        # env.render("human")
        state = env.reset()
        done = 0
        
        while not done:
            action = dagent.choose_action(state)
            next_state, new_reward, done, info = env.step(action)
            edge = info
            route_taken.append(edge)
            dagent.remember(state, action, new_reward, next_state, done)
            cumulative_reward += new_reward

            if len(dagent.memory) > dagent.batch_size:
                dagent.replay(dagent.batch_size)
                dagent.hard_update()

            state = next_state

        wandb.log({
            "cumulative_reward": cumulative_reward,
            "epsilon": dagent.get_epsilon(),
            "episode": episode,
            "agent_steps": env.agent_step,
            "simulation_steps": env.sumo.simulation.getTime(),
            "Distance": env.distance_traveled
        })

        dagent.decay()
        env.close(episode, cumulative_reward, dagent.get_epsilon())
        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            dagent.save_model(episode)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    try:
        config = load_and_override_config(args)
        wandb.init(project=config['wandb']['project_name'],
                   entity=config['wandb']['entity'],
                   name=config['wandb']['name'],
                   group=config['wandb']['group'],
                   config=config)
        main_training_loop(config)
    except Exception as e:
        print(f"Error occurred: {e}")
        wandb.finish(exit_code=1)
