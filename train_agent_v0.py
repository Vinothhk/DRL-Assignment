import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from env.ant_grid_env import AntGridEnv
import pygame

# Make sure this callback class is in your script
class RenderCallback(BaseCallback):
    """
    Callback for rendering the environment during training.
    """
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.last_render_time = 0

    def _on_step(self) -> bool:
        if self.n_calls % 10 == 0:
            self.model.env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
        return True

def main():
    # Make sure pygame is initialized
    pygame.init()
    
    # 1. Instantiate the custom environment with the correct rendering mode.
    walls = [(4, i) for i in range(3, 7)] + [(i, 2) for i in range(3, 6)] + [(8, i) for i in range(4, 7)]
    env = AntGridEnv(size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=walls, render_mode="human")
    
    check_env(env)

    # 2. Instantiate a Stable Baselines3 algorithm.
    model = PPO("MlpPolicy", env, verbose=1)

    # 3. Instantiate the rendering callback
    render_callback = RenderCallback()

    # 4. Train the agent with the callback
    print("Starting training...")
    # Pass the callback to the learn method
    model.learn(total_timesteps=100_000, callback=render_callback)
    print("Training finished. Saving the model...")

    # 5. Save and evaluate as before
    model.save("ppo_ant_grid")
    # ... (evaluation code remains the same)
    
    env.close()

if __name__ == '__main__':
    main()