import argparse
from utils import Utils
from environment import sim_manager as so
import wandb

def parse_args():
    """
    Parse command-line arguments for hyperparameters and configurations.
    """
    parser = argparse.ArgumentParser(description="Reinforcement Learning for SUMO Traffic Simulation")
    parser.add_argument('--config', type=str, default='src/configurations/config.yaml', help='Path to the configuration file')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate for the agent')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--epsilon_decay', type=float, default=None, help='Epsilon decay rate')
    parser.add_argument('--num_agents', type=int, default=None, help='Number of agents')
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
    if args.num_agents is not None:
        config['env']['num_agents'] = args.num_agents
    return config

def main_training_loop(config):
    """
    Main training loop for the reinforcement learning agents.
    """
    env = so.create_env(config=config)
    agents = [so.create_agent(config=config) for _ in range(config['env']['num_agents'])]
    best_reward = float('-inf')

    for episode in range(config['training_settings']['episodes']):
        cumulative_rewards = [0] * config['env']['num_agents']
        route_taken = [[] for _ in range(config['env']['num_agents'])]

        if episode % 1000 == 0:
            env.render("human")
        else:
            env.render()
        # env.render("human")

        states = env.reset()
        dispatched_taxis = [i for i, dispatched in enumerate(env.dispatched) if dispatched]

        
        dones = [0] * len(dispatched_taxis)

        while not all(dones):
            
            # for i in enumerate(dispatched_taxis):
                
            active_taxis_indices = [i for i in dispatched_taxis if not dones[i]]

            actions = [agents[i].choose_action(states[i]) if not dones[i] else 'None' for i in dispatched_taxis]

            if not actions.count(None) == len(actions):
                next_states, rewards, dones, infos = env.step(actions)

            for j, idx in enumerate(active_taxis_indices):
                agents[idx].remember(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])
                cumulative_rewards[idx] += rewards[idx]

                if len(agents[idx].memory) > agents[idx].batch_size:
                    agents[idx].replay(agents[idx].batch_size)
                    agents[idx].hard_update()

                states[idx] = next_states[idx]
                route_taken[idx].append(infos[idx])



        for i in dispatched_taxis:
            wandb.log({
                f"cumulative_reward_agent_{i}": cumulative_rewards[i],
                f"epsilon_agent_{i}": agents[i].get_epsilon(),
                f"episode_agent_{i}": episode,
                f"agent_steps_agent_{i}": env.vehicles[i].agent_step,
                f"simulation_steps_agent_{i}": env.sumo.simulation.getTime(),
                f"Distance_agent_{i}": env.vehicles[i].get_dist()
            })

            agents[i].decay()
            env.pre_close(episode,i, cumulative_rewards[i], agents[i].get_epsilon())
            if cumulative_rewards[i] > best_reward:
                best_reward = cumulative_rewards[i]
                agents[i].save_model(episode)
        env.quiet_close()

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
