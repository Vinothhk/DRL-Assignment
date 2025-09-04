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
import time
import numpy as np
import torch
import random

# --- Reproducibility Function ---
def set_seed(seed: int):
    """
    Sets the seed for reproducibility across libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

# --- Baseline Agent (Random Policy) ---
class RandomAgent:
    """
    A simple agent that chooses random actions from the environment's action space.
    Used as a baseline for comparison.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=False):
        # Choose a random action from the action space
        return self.action_space.sample(), None
    
def main():
    # --- 0. Hyperparameters and Reproducibility ---
    seed = 42
    set_seed(seed)
    
    hyperparameters = {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "total_timesteps": 100_000,
        "n_steps": 2048,
        "gae_lambda": 0.95,
        "seed": seed
    }
    
    # Initialize WandB
    run = wandb.init(
        project="ANTGRID-RL-Training",
        name="PPO",
        config=hyperparameters,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=["PPO", "AntGridEnv", "Reproducibility"],
        notes="Experiment to document hyperparameters, ensure reproducibility, and track compute time."
    )
    
    # Log library versions for reproducibility
    wandb.config.update({
        "wandb_version": wandb.__version__,
        "stable_baselines3_version": "2.2.1", # Example, replace with actual version
        "gymnasium_version": "0.29.1", # Example, replace with actual version
        "pygame_version": pygame.__version__,
    })

    # --- 1. Instantiate the custom environment ---
    walls = [(4, i) for i in range(3, 7)] + [(i, 2) for i in range(3, 6)] + [(8, i) for i in range(4, 7)]
    env = AntGridEnv(size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=walls, render_mode="human")
    
    # Check the environment to ensure it's compatible with the Gymnasium API
    check_env(env)

    # --- 2. Instantiate a Stable Baselines3 PPO algorithm ---
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparameters["learning_rate"],
        gamma=hyperparameters["gamma"],
        n_steps=hyperparameters["n_steps"],
        gae_lambda=hyperparameters["gae_lambda"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}"
    )

    # --- 3. Instantiate the rendering and WandB callbacks ---
    render_callback = RenderCallback()
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    callback_list = CallbackList([render_callback, wandb_callback])

    # --- 4. Train the agent with the list of callbacks ---
    print("Starting training...")
    start_time = time.time()
    # The seed is passed here for the initial environment reset
    model.learn(total_timesteps=hyperparameters["total_timesteps"], callback=callback_list)
    end_time = time.time()
    total_compute_time = end_time - start_time
    print("Training finished. Saving the final model...")
    
    # Log total compute time to WandB
    wandb.log({"total_compute_time_seconds": total_compute_time})

    # --- 5. Save the final model ---
    model.save("ppo_ant_grid")
    
    # --- 6. Evaluate and Compare ---
    def evaluate_and_log(agent, env, name, num_episodes=5):
        """Helper function to evaluate an agent and log the results."""
        total_rewards = []
        total_steps = []
        for i in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                if terminated or truncated:
                    done = True
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        
        wandb.log({
            f"{name}_avg_reward": avg_reward,
            f"{name}_avg_steps": avg_steps
        })
        print(f"{name} Evaluation: Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.2f}")

    # Evaluate the trained PPO agent
    evaluate_and_log(model, env, "PPO_Trained_Agent")
    
    # Evaluate the random baseline agent
    random_agent = RandomAgent(env.action_space)
    evaluate_and_log(random_agent, env, "Random_Baseline")
    
    # Clean up
    env.close()
    wandb.finish()

if __name__ == '__main__':
    main()
