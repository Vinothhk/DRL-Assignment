# import numpy as np
# import time
# import os
# import pygame
# import gymnasium as gym
# from gymnasium import spaces

# # --- Constants for Pygame Visualization ---
# CELL_SIZE = 50  # Size of each grid cell in pixels
# FPS = 10        # Frames per second
# FONT_SIZE = 14

# # The main Ant on a Grid RL Environment (simplified)
# class AntGridEnv(gym.Env):
#     def __init__(self, size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=None):
#         super().__init__()
#         self.size = size
#         self.food_pos = food_pos
#         self.home_pos = home_pos
#         self.wall_pos = set(wall_pos) if wall_pos else set()

#         self.ant_pos = None
#         self.has_food = False
#         self.steps = 0
#         self.max_steps = 100

#         # Define Gymnasium action and observation spaces
#         self.action_space = spaces.Discrete(5) # 0:up, 1:down, 2:left, 3:right, 4:stay
#         self.observation_space = spaces.Box(low=0, high=max(size[0], size[1]), shape=(3,), dtype=np.int32)
        
#         # --- Pygame and Image Loading ---
#         pygame.init()
#         self.screen_width = self.size[0] * CELL_SIZE
#         self.screen_height = self.size[1] * CELL_SIZE + FONT_SIZE * 2
#         self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
#         pygame.display.set_caption("Ant on a Grid RL Environment")
#         self.clock = pygame.time.Clock()

#         # Load and Scale Images
#         self.images = None
#         try:
#             ant_image = pygame.transform.scale(pygame.image.load('ant.jpeg'), (CELL_SIZE, CELL_SIZE))
#             home_image = pygame.transform.scale(pygame.image.load('home.png'), (CELL_SIZE, CELL_SIZE))
#             food_image = pygame.transform.scale(pygame.image.load('ant_food.jpeg'), (CELL_SIZE, CELL_SIZE))
            
#             self.images = {
#                 'ant': ant_image,
#                 'home': home_image,
#                 'food': food_image
#             }

#         except pygame.error as e:
#             print(f"Error loading image: {e}")
#             print("Please ensure 'ant.jpeg', 'home.png', and 'ant_food.jpeg' are in the same directory.")
            
#             # --- Fallback to simple shapes if images could not be loaded ---
#             print("Using fallback shapes for visualization.")
#             self.images = {
#                 'ant': self._create_shape(self._create_ant_shape),
#                 'home': self._create_shape(self._create_home_shape),
#                 'food': self._create_shape(self._create_food_shape)
#             }
        
#     def _create_shape(self, draw_func):
#         """Helper to create a surface and draw a shape on it."""
#         surface = pygame.Surface((CELL_SIZE, CELL_SIZE))
#         surface.fill((255, 255, 255)) # Fill with white to ensure no black background
#         draw_func(surface)
#         return surface

#     def _create_ant_shape(self, surface):
#         """Draws a blue ant shape."""
#         pygame.draw.circle(surface, (50, 50, 255), (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 3)

#     def _create_home_shape(self, surface):
#         """Draws a green home shape."""
#         pygame.draw.circle(surface, (0, 200, 0), (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 3)

#     def _create_food_shape(self, surface):
#         """Draws a red food shape."""
#         pygame.draw.circle(surface, (255, 100, 100), (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 3)

#     def reset(self, seed=None, options=None):
#         """Resets the environment to its initial state."""
#         super().reset(seed=seed)
#         self.ant_pos = (self.np_random.integers(self.size[0]), self.np_random.integers(self.size[1]))
#         while self.ant_pos in self.wall_pos or self.ant_pos == self.food_pos or self.ant_pos == self.home_pos:
#             self.ant_pos = (self.np_random.integers(self.size[0]), self.np_random.integers(self.size[1]))
        
#         self.has_food = False
#         self.steps = 0
        
#         observation = np.array([self.ant_pos[0], self.ant_pos[1], int(self.has_food)], dtype=np.int32)
#         info = {}
#         return observation, info

#     def step(self, action):
#         """
#         Takes an action and updates the environment.
#         Returns: next_state, reward, terminated, truncated, info
#         """
#         reward = -0.1  # Time penalty

#         old_pos = self.ant_pos
#         x, y = old_pos

#         if action == 0:  # Up
#             y -= 1
#         elif action == 1:  # Down
#             y += 1
#         elif action == 2:  # Left
#             x -= 1
#         elif action == 3:  # Right
#             x += 1
#         elif action == 4:  # Stay
#             pass

#         new_pos = (x, y)

#         if new_pos not in self.wall_pos and 0 <= x < self.size[0] and 0 <= y < self.size[1]:
#             self.ant_pos = new_pos
#         else:
#             reward -= 1.0  # Penalty for hitting a wall

#         if not self.has_food:
#             old_dist = np.linalg.norm(np.array(old_pos) - np.array(self.food_pos))
#             new_dist = np.linalg.norm(np.array(self.ant_pos) - np.array(self.food_pos))
#             if new_dist < old_dist:
#                 reward += 0.05
#         else:
#             old_dist = np.linalg.norm(np.array(old_pos) - np.array(self.home_pos))
#             new_dist = np.linalg.norm(np.array(self.ant_pos) - np.array(self.home_pos))
#             if new_dist < old_dist:
#                 reward += 0.05

#         terminated = False
#         if self.ant_pos == self.food_pos and not self.has_food:
#             self.has_food = True
#             reward += 1.0
            
#         if self.ant_pos == self.home_pos and self.has_food:
#             self.has_food = False
#             reward += 10.0
#             terminated = True # Episode ends when food is delivered
            
#         truncated = False
#         self.steps += 1
#         if self.steps >= self.max_steps:
#             truncated = True

#         observation = np.array([self.ant_pos[0], self.ant_pos[1], int(self.has_food)], dtype=np.int32)
#         info = {}
        
#         return observation, reward, terminated, truncated, info

#     def render(self):
#         """Renders the current state of the environment using Pygame."""
#         self.screen.fill((255, 255, 255))
        
#         rows, cols = self.size
        
#         for y in range(rows):
#             for x in range(cols):
#                 rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#                 pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

#         for pos in self.wall_pos:
#             x, y = pos
#             rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#             pygame.draw.rect(self.screen, (100, 100, 100), rect)
            
#         home_rect = pygame.Rect(self.home_pos[0] * CELL_SIZE, self.home_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#         self.screen.blit(self.images['home'], home_rect.topleft)

#         food_rect = pygame.Rect(self.food_pos[0] * CELL_SIZE, self.food_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#         self.screen.blit(self.images['food'], food_rect.topleft)

#         ant_x, ant_y = self.ant_pos
#         ant_rect = pygame.Rect(ant_x * CELL_SIZE, ant_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#         self.screen.blit(self.images['ant'], ant_rect.topleft)

#         font = pygame.font.Font(None, FONT_SIZE)
#         text_surface = font.render(f"Step: {self.steps} | Has Food: {self.has_food}", True, (0, 0, 0))
#         self.screen.blit(text_surface, (10, self.size[1] * CELL_SIZE + 5))

#         pygame.display.flip()
#         self.clock.tick(FPS)

#     def close(self):
#         pygame.quit()
        
# if __name__ == "__main__":
#     walls = [(4, i) for i in range(3, 7)] + [(i, 2) for i in range(3, 6)] + [(8, i) for i in range(4, 7)] 
#     env = AntGridEnv(size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=walls)
    
#     running = True
#     while running:
#         observation, info = env.reset()
#         done = False
#         while not done:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#                     done = True
            
#             action = env.action_space.sample()
#             observation, reward, terminated, truncated, info = env.step(action)
            
#             env.render()
            
#             if terminated or truncated:
#                 done = True
#                 print(f"Episode finished in {env.steps} steps.")
#                 time.sleep(2)
        
#     env.close()

import numpy as np
import time
import os
import pygame
import gymnasium as gym
from gymnasium import spaces

# --- Constants for Pygame Visualization ---
CELL_SIZE = 50  # Size of each grid cell in pixels
FPS = 10        # Frames per second
FONT_SIZE = 14

# The main Ant on a Grid RL Environment (simplified)
class AntGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=None, render_mode=None):
        super().__init__()
        self.size = size
        self.food_pos = food_pos
        self.home_pos = home_pos
        self.wall_pos = set(wall_pos) if wall_pos else set()
        self.render_mode = render_mode

        self.ant_pos = None
        self.has_food = False
        self.steps = 0
        self.max_steps = 100

        # Define Gymnasium action and observation spaces
        self.action_space = spaces.Discrete(5) # 0:up, 1:down, 2:left, 3:right, 4:stay
        self.observation_space = spaces.Box(low=0, high=max(size[0], size[1]), shape=(3,), dtype=np.int32)
        
        # --- Pygame and Image Loading ---
        if self.render_mode == "human":
            pygame.init()
            self.screen_width = self.size[0] * CELL_SIZE
            self.screen_height = self.size[1] * CELL_SIZE + FONT_SIZE * 2
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Ant on a Grid RL Environment")
            self.clock = pygame.time.Clock()

        self.images = None
        try:
            ant_image = pygame.transform.scale(pygame.image.load('ant.jpeg'), (CELL_SIZE, CELL_SIZE))
            home_image = pygame.transform.scale(pygame.image.load('home.png'), (CELL_SIZE, CELL_SIZE))
            food_image = pygame.transform.scale(pygame.image.load('ant_food.jpeg'), (CELL_SIZE, CELL_SIZE))
            
            self.images = {
                'ant': ant_image,
                'home': home_image,
                'food': food_image
            }

        except pygame.error as e:
            print(f"Error loading image: {e}")
            print("Please ensure 'ant.jpeg', 'home.png', and 'ant_food.jpeg' are in the same directory.")
            
            # --- Fallback to simple shapes if images could not be loaded ---
            print("Using fallback shapes for visualization.")
            self.images = {
                'ant': self._create_shape(self._create_ant_shape),
                'home': self._create_shape(self._create_home_shape),
                'food': self._create_shape(self._create_food_shape)
            }
        
    def _create_shape(self, draw_func):
        """Helper to create a surface and draw a shape on it."""
        surface = pygame.Surface((CELL_SIZE, CELL_SIZE))
        surface.fill((255, 255, 255)) # Fill with white to ensure no black background
        draw_func(surface)
        return surface

    def _create_ant_shape(self, surface):
        """Draws a blue ant shape."""
        pygame.draw.circle(surface, (50, 50, 255), (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 3)

    def _create_home_shape(self, surface):
        """Draws a green home shape."""
        pygame.draw.circle(surface, (0, 200, 0), (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 3)

    def _create_food_shape(self, surface):
        """Draws a red food shape."""
        pygame.draw.circle(surface, (255, 100, 100), (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 3)

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.ant_pos = (self.np_random.integers(self.size[0]), self.np_random.integers(self.size[1]))
        while self.ant_pos in self.wall_pos or self.ant_pos == self.food_pos or self.ant_pos == self.home_pos:
            self.ant_pos = (self.np_random.integers(self.size[0]), self.np_random.integers(self.size[1]))
        
        self.has_food = False
        self.steps = 0
        
        observation = np.array([self.ant_pos[0], self.ant_pos[1], int(self.has_food)], dtype=np.int32)
        info = {}
        return observation, info

    def step(self, action):
        """
        Takes an action and updates the environment.
        Returns: next_state, reward, terminated, truncated, info
        """
        reward = -0.1  # Time penalty

        old_pos = self.ant_pos
        x, y = old_pos

        if action == 0:  # Up
            y -= 1
        elif action == 1:  # Down
            y += 1
        elif action == 2:  # Left
            x -= 1
        elif action == 3:  # Right
            x += 1
        elif action == 4:  # Stay
            pass

        new_pos = (x, y)

        if new_pos not in self.wall_pos and 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.ant_pos = new_pos
        else:
            reward -= 1.0  # Penalty for hitting a wall

        if not self.has_food:
            old_dist = np.linalg.norm(np.array(old_pos) - np.array(self.food_pos))
            new_dist = np.linalg.norm(np.array(self.ant_pos) - np.array(self.food_pos))
            if new_dist < old_dist:
                reward += 0.05
        else:
            old_dist = np.linalg.norm(np.array(old_pos) - np.array(self.home_pos))
            new_dist = np.linalg.norm(np.array(self.ant_pos) - np.array(self.home_pos))
            if new_dist < old_dist:
                reward += 0.05

        terminated = False
        if self.ant_pos == self.food_pos and not self.has_food:
            self.has_food = True
            reward += 1.0
            
        if self.ant_pos == self.home_pos and self.has_food:
            self.has_food = False
            reward += 10.0
            terminated = True
            
        truncated = False
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        observation = np.array([self.ant_pos[0], self.ant_pos[1], int(self.has_food)], dtype=np.int32)
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the current state of the environment using Pygame."""
        if self.render_mode != "human":
            return
            
        self.screen.fill((255, 255, 255))
        
        rows, cols = self.size
        
        for y in range(rows):
            for x in range(cols):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        for pos in self.wall_pos:
            x, y = pos
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, (100, 100, 100), rect)
            
        home_rect = pygame.Rect(self.home_pos[0] * CELL_SIZE, self.home_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        self.screen.blit(self.images['home'], home_rect.topleft)

        food_rect = pygame.Rect(self.food_pos[0] * CELL_SIZE, self.food_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        self.screen.blit(self.images['food'], food_rect.topleft)

        ant_x, ant_y = self.ant_pos
        ant_rect = pygame.Rect(ant_x * CELL_SIZE, ant_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        self.screen.blit(self.images['ant'], ant_rect.topleft)

        font = pygame.font.Font(None, FONT_SIZE)
        text_surface = font.render(f"Step: {self.steps} | Has Food: {self.has_food}", True, (0, 0, 0))
        self.screen.blit(text_surface, (10, self.size[1] * CELL_SIZE + 5))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
        
if __name__ == "__main__":
    walls = [(4, i) for i in range(3, 7)] + [(i, 2) for i in range(3, 6)] + [(8, i) for i in range(4, 7)] 
    env = AntGridEnv(size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=walls, render_mode="human")
    
    running = True
    while running:
        observation, info = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
            
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            env.render()
            
            if terminated or truncated:
                done = True
                print(f"Episode finished in {env.steps} steps.")
                time.sleep(2)
        
    env.close()
