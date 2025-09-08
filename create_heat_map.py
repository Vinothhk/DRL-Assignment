import gymnasium as gym
from stable_baselines3 import DQN
from env.ant_grid_env import AntGridEnv
import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch

def create_heatmap(model, env):
    """
    Generates and saves a heatmap of the maximum Q-values for each state.

    Args:
        model: The trained DQN model.
        env: The environment instance.
    """
    rows, cols = env.size
    q_values = np.zeros((rows, cols))
    
    print("Generating Q-value heatmap...")
    # Iterate through every possible position on the grid
    for y in range(rows):
        for x in range(cols):
            # Check the Q-value for both 'with food' and 'without food' states
            obs_no_food = np.array([x, y, 0])
            # The ._predict() method returns raw actions, we need the Q-values from the network
            q_values_no_food = model.q_net.forward(torch.tensor(obs_no_food, dtype=torch.float32).unsqueeze(0).to(model.device))

            obs_with_food = np.array([x, y, 1])
            q_values_with_food = model.q_net.forward(torch.tensor(obs_with_food, dtype=torch.float32).unsqueeze(0).to(model.device))

            # The final value for the heatmap is the max Q-value
            # for the most likely state (e.g., without food)
            q_values[y, x] = max(q_values_no_food.detach().cpu().numpy().flatten())


    # --- Rotate the heatmap for correct visual orientation ---
    # Flip the array 180 degrees to match the environment's top-left origin
    rotated_q_values = np.flipud(q_values)


    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 12))
    # Use the rotated data and set origin to 'upper'
    im = ax.imshow(rotated_q_values, cmap='viridis', origin='upper')

    # Draw grid lines
    ax.set_xticks(np.arange(cols+1)-.5, minor=True)
    ax.set_yticks(np.arange(rows+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Plot food and home positions
    ax.text(env.food_pos[0], env.food_pos[1], 'Food', ha='center', va='center', color='white', fontsize=12)
    ax.text(env.home_pos[0], env.home_pos[1], 'Home', ha='center', va='center', color='white', fontsize=12)
    
    # Plot walls
    for pos in env.wall_pos:
        ax.add_patch(plt.Rectangle((pos[0]-0.5, pos[1]-0.5), 1, 1, color='gray'))

    ax.set_title("DQN Learned Q-Value Heatmap (Rotated)")
    plt.colorbar(im, ax=ax, label="Max Q-Value")
    plt.savefig("dqn_heatmap.png")
    plt.show()

def main():
    # Load the trained DQN model
    model_path = "models/dqn_ant_grid.zip"
    try:
        model = DQN.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the DQN model file exists.")
        return

    # Create a dummy environment to get the grid dimensions
    walls = [(4, i) for i in range(3, 7)] + [(i, 2) for i in range(3, 6)] + [(8, i) for i in range(4, 7)]
    env = AntGridEnv(size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=walls, render_mode="human")
    # Generate and display the heatmap
    create_heatmap(model, env)
    
if __name__ == "__main__":
    main()
