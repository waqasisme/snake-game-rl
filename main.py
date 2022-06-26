import argparse
import os
import time

from stable_baselines3 import DQN, PPO

from custom_snake_env import CustomSnakeEnv
from stable_baselines3.common.env_checker import check_env

from custom_snake_env_cnn import CustomSnakeEnvCNN


parser = argparse.ArgumentParser(description='Parse agent parameters')
parser.add_argument('--mode', choices=['train', 'test', 'check'], default='check', type=str, required=True)
parser.add_argument('--policy', choices=['MLP', 'CNN'], type=str, required=True)
parser.add_argument('--agent', choices=['PPO', 'DQN'], type=str, 
                    help='The kind of agent to train or load', required=True)
parser.add_argument('--load', type=str, help="specify the name of the model file to load (located in models/...")
parser.add_argument('--tag', type=str)

args = parser.parse_args()

env = CustomSnakeEnv('SNAKE_GAME_MLP', (50, 50), False) if args.policy == 'MLP' \
    else CustomSnakeEnvCNN('SNAKE_GAME_CNN', (50, 50)) if args.policy == 'CNN' \
    else None

if args.mode == 'train':

    _, _, model_folder, model_timestep = [None, None, None, None] if args.load is None else args.load.split('\\')
    model_timestep = None if model_timestep is None else model_timestep.split('.')[0]
    
    models_dir = f'models/{args.agent}'
    models_dir += f'/{int(time.time()) if model_folder is None else model_folder}'
    models_dir += '' if args.tag is None else f'_{args.tag}'  

    logs_dir = f'logs/{args.agent}'
    logs_dir += f'/{int(time.time()) if model_folder is None else model_folder}'
    logs_dir += '' if args.tag is None else f'_{args.tag}'

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    if args.load is None:
        policy = "MlpPolicy" if args.policy == 'MLP' else "CnnPolicy" if args.policy == 'CNN' else None
        
        model = PPO(policy, env, verbose=1, tensorboard_log=logs_dir) if args.agent == 'PPO' \
            else DQN(policy, env, verbose=1, tensorboard_log=logs_dir) \
            if args.agent == 'DQN' \
            else None
    else:
        model = PPO.load(args.load, env, verbose=1, tensorboard_log=logs_dir) if args.agent == 'PPO' \
            else DQN.load(args.load, env, verbose=1, tensorboard_log=logs_dir) if args.agent == 'DQN' \
            else None
            
    TIMESTEPS = 25000
    counter = 1 if model_timestep is None else (int(int(model_timestep) / TIMESTEPS) + 1)
    while True:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{args.agent}_{args.policy}")
        model.save(f"{models_dir}/{TIMESTEPS*counter}")
        counter += 1

elif args.mode == 'test':
    model_path = f'{args.load}'.replace('\\', '/').replace('.zip', '')
    print(model_path)
    model = PPO.load(args.load, env, verbose=1) if args.agent == 'PPO' \
        else DQN.load(args.load, env, verbose=1) if args.agent == 'DQN' \
        else None
    
    episodes = 10
    for episode in range(episodes):
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

elif args.mode == 'check':
    check_env(env)
    episodes = 10

    for episode in range(episodes):
        done = False
        obs = env.reset()
        while not done:
            env.render()
            random_action = env.action_space.sample()
            print("action", random_action)
            obs, reward, done, info = env.step(random_action)
            print('reward', reward)

env.close()
