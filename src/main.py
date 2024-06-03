
from sumo_mmrl import Utils
from sumo_mmrl.environment import sim_manager as so
import wandb

config = Utils.load_yaml_config('config.yaml')
EPISODES = config['training_settings']['episodes']


wandb.init(project=config['wandb']['project_name'],
           entity=config['wandb']['entity'],
           name=config['wandb']['name'],
           group=config['wandb']['group'],
           config=config)

def main_training_loop():
    env = so.create_env(config=config)
    dagent = so.create_agent(config=config)




    for episode in range(EPISODES):
        route_taken = []
        cumulative_reward = 0
        env.render()
        # env.render("gui")
        state = env.reset()

        done = 0
        
        while done != 1:

            action= dagent.choose_action(state)
            next_state, new_reward, done,info = env.step(action)
            edge = info
            route_taken.append(edge)
            dagent.remember(state, action, new_reward, next_state, done )
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
        dagent.save_model(episode)
    
    wandb.finish()

if __name__ == "__main__":
    main_training_loop()
