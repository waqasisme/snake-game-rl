import time
import gym
from gym import spaces
import numpy as np

from snake_game import Snake_Game


BOUNDS = (50, 50)
TILE_SIZE = 10


class CustomSnakeEnvCNN(gym.Env):
    def __init__(self, title: str = 'SNAKE_GAME', bounds: tuple[int, int] = BOUNDS):
        super(CustomSnakeEnvCNN, self).__init__()
        self.title = title
        self.bounds = bounds
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(bounds[0], bounds[1], 1), dtype=np.uint8)
        self.game = Snake_Game(self.title, list(map(lambda x: x * TILE_SIZE, self.bounds)))
        self.seed(23)
        
    def seed(self, seed=None):
        np.random.seed(seed)
        
    def step(self, action: int):
        self.game.update(action)
        
        info = {}
        observation, ate_apple, is_dead = self.state()
        if ate_apple:
            reward = 1 
        else: 
            reward = -0.001  # DECAY
        if is_dead:
            reward = -1

        done = is_dead
        
        return observation, reward, done, info

    def reset(self):
        self.game.reset()
        observation, _, _ = self.state()
        return observation

    def state(self):
        head, body, apple, ate_apple, is_dead, snake_len, score = self.game.state() 

        observation = np.zeros((self.bounds[0], self.bounds[1], 1), dtype=np.uint8)  # image representation

        observation[int(head[0] / TILE_SIZE), int(head[1] / TILE_SIZE), 0] = 255 

        for part in body:
            observation[int(part[0] / TILE_SIZE), int(part[1] / TILE_SIZE), 0] = 255

        observation[int(apple[0] / 10), int(apple[1] / 10), 0] = 255

        return observation, ate_apple, is_dead

    def render(self):
        init_time = time.time()
        while init_time + 0.03 > time.time():
            pass
        self.game.draw()
    
    def close(self):
        self.game.close()
