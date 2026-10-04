import argparse
from pathlib import Path
from stable_baselines3 import PPO
from rl.environments.mpc_tuning_env import MPCTuningEnv

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--total-timesteps", type=int, default=10000); parser.add_argument("--seed", type=int, default=0); args = parser.parse_args()
    env = MPCTuningEnv(); model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.total_timesteps)
    out = Path("models/mpc_tuning"); out.mkdir(parents=True, exist_ok=True); model.save(out / f"seed{args.seed}_final")

if __name__ == "__main__": main()
