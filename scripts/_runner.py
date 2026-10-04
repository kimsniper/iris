from pathlib import Path
import yaml
from rl.environments.balancing_env import BalancingEnv

def run(controller, episodes=3):
    env=BalancingEnv()
    for ep in range(episodes):
        obs,_=env.reset(seed=ep); controller.reset(); done=False; steps=0
        while not done:
            action=controller.compute(obs,env.dt); obs,_,terminated,truncated,_=env.step([float(action[0])]); done=terminated or truncated; steps+=1
        print({'episode':ep,'steps':steps,'seconds':steps*env.dt,'fell':terminated})
