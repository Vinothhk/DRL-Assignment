import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from env.ant_grid_env import AntGridEnv
import pygame
import wandb
import os

# Make sure this callback class is in your script
class RenderCallback(BaseCallback):
    """
    Callback for rendering the environment during training.
    """
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Render every 4 steps to avoid slowing down training too much
        if self.n_calls % 4 == 0:
            self.model.env.render()
            
            # Check for Pygame quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Return False to stop the training
                    return False
        return True

def main():
    # 0. Initialize WandB
    # Make sure you are logged in to WandB (e.g., `wandb login`)
    run = wandb.init(
        project="ant-grid-rl-training",
        sync_tensorboard=True,  # Syncs TensorBoard logs to WandB
    )
    
    # 1. Instantiate the custom environment with the correct rendering mode.
    walls = [(4, i) for i in range(3, 7)] + [(i, 2) for i in range(3, 6)] + [(8, i) for i in range(4, 7)]
    env = AntGridEnv(size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=walls, render_mode="human")
    
    # Check the environment to ensure it's compatible with the Gymnasium API
    check_env(env)

    # 2. Instantiate a Stable Baselines3 algorithm.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")

    # 3. Instantiate the rendering and WandB callbacks
    render_callback = RenderCallback()
    wandb_callback = WandbCallback(
        gradient_save_freq=100, # Save gradients every 100 steps
        model_save_path=f"models/{run.id}", # Save models to a unique path for this run
        verbose=2,
    )

    # Combine the callbacks into a list
    callback_list = CallbackList([render_callback, wandb_callback])

    # 4. Train the agent with the list of callbacks
    print("Starting training...")
    model.learn(total_timesteps=100_000, callback=callback_list)
    print("Training finished. Saving the final model...")

    # 5. Save the final model
    model.save("ppo_ant_grid")
    
    # Clean up
    env.close()
    wandb.finish()

if __name__ == '__main__':
    main()
