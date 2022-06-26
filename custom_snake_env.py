from collections import deque
import time
import gym
from gym import spaces
import numpy as np

from snake_game import Snake_Game


BOUNDS = (50, 50)
TILE_SIZE = 10


class CustomSnakeEnv(gym.Env):
    def __init__(self, title: str = 'SNAKE_GAME', bounds: tuple[int, int] = BOUNDS, with_history=True):
        super(CustomSnakeEnv, self).__init__()
        self.title = title
        self.bounds = bounds
        self.with_history = with_history
        self.history = []
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(bounds[0], bounds[1]), 
                                            shape=(15 if self.with_history else 5, ), dtype=np.float32)
        self.game = Snake_Game(self.title, list(map(lambda x: x * TILE_SIZE, self.bounds)))
        self.seed(23)
        
    def seed(self, seed=None):
        np.random.seed(seed)
        
    def step(self, action: int):
        if self.with_history:
            self.history.append(action)
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
        self.history = deque([0] * 10, maxlen=10)
        observation, _, _ = self.state()
        return observation

    def render(self):
        init_time = time.time()
        while init_time + 0.03 > time.time():
            pass
        self.game.draw()

    def state(self):
        head, body, apple, ate_apple, is_dead, snake_len, score = self.game.state() 

        observation = np.array(
            [head[0] / TILE_SIZE, head[1] / TILE_SIZE, 
             apple[0] / TILE_SIZE, apple[1] / TILE_SIZE, 
             snake_len * 2 / (self.bounds[0])] + list(self.history if self.with_history else []), 
            dtype=np.float32)
        return observation, ate_apple, is_dead
    
    def close(self):
        self.game.close()
