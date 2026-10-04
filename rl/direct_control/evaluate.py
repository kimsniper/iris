import argparse
from stable_baselines3 import PPO
from rl.environments.balancing_env import BalancingEnv

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--episodes", type=int, default=10); args = parser.parse_args()
    env = BalancingEnv(); model = PPO.load(args.model); returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=1000 + episode); done = False; total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True); obs, reward, terminated, truncated, _ = env.step(action); total += reward; done = terminated or truncated
        returns.append(total)
    print({"episodes": args.episodes, "mean_return": sum(returns)/len(returns)})
if __name__ == "__main__": main()
