import gymnasium as gym
from stable_baselines3 import PPO
from env.ant_grid_env import AntGridEnv
import pygame

def evaluate_model(model_path: str, num_episodes: int = 5):
    """
    Loads a trained PPO model and evaluates its performance on the AntGridEnv.
    
    Args:
        model_path (str): The file path to the saved PPO model.
        num_episodes (int): The number of episodes to run for evaluation.
    """
    # 1. Instantiate the environment in "human" render mode to visualize the agent
    walls = [(4, i) for i in range(3, 7)] + [(i, 2) for i in range(3, 6)] + [(8, i) for i in range(4, 7)]
    env = AntGridEnv(size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=walls, render_mode="human")
    
    print(f"Loading model from {model_path}...")
    try:
        # 2. Load the trained model
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and the model was trained successfully.")
        env.close()
        return

    print(f"Evaluating the model for {num_episodes} episodes...")
    
    # Run a loop for the specified number of evaluation episodes
    for episode in range(num_episodes):
        # obs, info = env.reset()
        # done = False
        # total_reward = 0
        # steps = 0
        
        # # Scenario 1: Deterministic evaluation (as before)
        # # This will show the agent getting stuck if its learned policy is flawed
        # print(f"\n--- Episode {episode + 1}: Deterministic Policy (Stuck Behavior) ---")
        # while not done:
        #     action, _ = model.predict(obs, deterministic=True)
        #     obs, reward, terminated, truncated, info = env.step(action)
        #     total_reward += reward
        #     steps += 1
        #     env.render()
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             return # Exit gracefully
        #     if terminated or truncated:
        #         done = True
        # print(f"Finished in {steps} steps with total reward {total_reward:.2f}")

        # Scenario 2: Stochastic evaluation (simulates training exploration)
        # This will show the agent trying to move even after a wall hit
        obs, info = env.reset() # Reset for the new scenario
        done = False
        total_reward = 0
        steps = 0
        print(f"\n--- Episode {episode + 1}: Stochastic Policy (Continues After Wall Hit) ---")
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return # Exit gracefully
            if terminated or truncated:
                done = True
        print(f"Finished in {steps} steps with total reward {total_reward:.2f}")


    # 4. Clean up the environment
    env.close()
    
if __name__ == '__main__':
    pygame.init()
    evaluate_model("archives/ppo_ant_grid.zip")
    pygame.quit()
