import pygame # type: ignore
import numpy as np
import time
from ant_grid_env import AntGridEnv

# --- Constants for Pygame Visualization ---
CELL_SIZE = 50  # Size of each grid cell in pixels
FPS = 10        # Frames per second
FONT_SIZE = 14

# --- Pygame Colors (RGB) ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
WALL_COLOR = (100, 100, 100)
ANT_COLOR = (50, 50, 255)
FOOD_COLOR = (255, 100, 100)
HOME_COLOR = (0, 200, 0)

def draw_grid(screen, env,images):
    """Draws the entire grid, including all elements."""
    screen.fill(WHITE)
    
    rows, cols = env.size
    
    for y in range(rows):
        for x in range(cols):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 1)

    # --- Draw Walls ---
    for pos in env.wall_pos:
        x, y = pos
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, WALL_COLOR, rect)

    # --- Draw Special Locations using images ---
    # Home
    home_rect = pygame.Rect(env.home_pos[0] * CELL_SIZE, env.home_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(images['home'], home_rect.topleft)

    # Food
    food_rect = pygame.Rect(env.food_pos[0] * CELL_SIZE, env.food_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(images['food'], food_rect.topleft)

    # --- Draw Ant ---
    ant_x, ant_y = env.ant_pos
    ant_rect = pygame.Rect(ant_x * CELL_SIZE, ant_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(images['ant'], ant_rect.topleft)
        
    # # --- Draw Special Locations ---
    # food_rect = pygame.Rect(env.food_pos[0] * CELL_SIZE, env.food_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    # pygame.draw.circle(screen, FOOD_COLOR, food_rect.center, CELL_SIZE // 3)
    
    # home_rect = pygame.Rect(env.home_pos[0] * CELL_SIZE, env.home_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    # pygame.draw.circle(screen, HOME_COLOR, home_rect.center, CELL_SIZE // 3)

    # # --- Draw Ant ---
    # ant_x, ant_y = env.ant_pos
    # ant_rect = pygame.Rect(ant_x * CELL_SIZE, ant_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    # pygame.draw.circle(screen, ANT_COLOR, ant_rect.center, CELL_SIZE // 3)

    # --- Draw Info Text ---
    font = pygame.font.Font(None, FONT_SIZE)
    text_surface = font.render(f"Step: {env.steps} | Has Food: {env.has_food}", True, BLACK)
    screen.blit(text_surface, (10, env.size[1] * CELL_SIZE + 5))

    pygame.display.flip()

def main():
    """Main function to run the visualization and simulation loop."""
    # walls = [(4, i) for i in range(3, 7)] + [(i, 6) for i in range(3, 8)]
    walls = [(4, i) for i in range(3, 7)] + [(i, 2) for i in range(3, 6)] + [(8, i) for i in range(4, 7)] 
    env = AntGridEnv(size=(10, 10), food_pos=(8, 8), home_pos=(1, 1), wall_pos=walls)
    
    pygame.init()
    
    screen_width = env.size[0] * CELL_SIZE
    screen_height = env.size[1] * CELL_SIZE + FONT_SIZE * 2
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Ant on a Grid RL Environment")
    clock = pygame.time.Clock()


    # --- Load and Scale Images ---
    try:
        # Assuming the generated image is saved as 'ant_home_food.png'
        # The image needs to be cropped and scaled
        # original_image = pygame.image.load('ant_home_food.png').convert_alpha()
        ant_image = pygame.transform.scale(pygame.image.load('ant.jpeg'), (CELL_SIZE, CELL_SIZE))
        home_image = pygame.transform.scale(pygame.image.load('home.png'), (CELL_SIZE, CELL_SIZE))
        food_image = pygame.transform.scale(pygame.image.load('ant_food.jpeg'), (CELL_SIZE, CELL_SIZE))
        
        images = {
            'ant': ant_image,
            'home': home_image,
            'food': food_image
        }

    except pygame.error as e:
        print(f"Error loading image: {e}")
        print("Please ensure 'ant_home_food.png' is in the same directory.")
        return # Exit if image fails to load


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        action = np.random.choice(env.action_space)
        state, reward, done, _ = env.step(action)
        
        draw_grid(screen, env,images)
        
        if done:
            print(f"Episode finished in {env.steps} steps.")
            time.sleep(2)
            env.reset()
            
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()