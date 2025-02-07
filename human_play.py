import gymnasium as gym
import pygame

# Initialize environment
env = gym.make("LunarLander-v3", render_mode="human")

# Initialize pygame for keyboard input
pygame.init()
screen = pygame.display.set_mode((400, 300))  # Dummy window to capture input
pygame.display.set_caption("Lunar Lander Controls")
clock = pygame.time.Clock()

# Mapping keys to actions
KEYS_TO_ACTIONS = {
    pygame.K_d: 1,  # Fire left engine
    pygame.K_a: 3,  # Fire right engine
    pygame.K_w: 2,  # Fire main engine
}

done = False
obs, info = env.reset()
action = 0  # Default: do nothing

while not done:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key in KEYS_TO_ACTIONS:
                action = KEYS_TO_ACTIONS[event.key]
        elif event.type == pygame.KEYUP:
            action = 0  # Stop firing when key is released

    # Step the environment
    obs, reward, terminated, truncated, _ = env.step(action)

    # Reset the environment if done
    if terminated or truncated:
        obs, info = env.reset()

    clock.tick(30)  # Limit FPS

env.close()
pygame.quit()
