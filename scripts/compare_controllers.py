import csv
from pathlib import Path
import yaml
from controllers.pid.pid_controller import PIDController
from controllers.mpc.mpc_controller import MPCController
from rl.environments.balancing_env import BalancingEnv

def evaluate(name, controller, episodes=5):
    rows=[]; env=BalancingEnv()
    for ep in range(episodes):
        obs,_=env.reset(seed=1000+ep); controller.reset(); done=False; steps=0
        while not done:
            command=controller.compute(obs,env.dt); obs,_,terminated,truncated,_=env.step([float(command[0])]); done=terminated or truncated; steps+=1
        rows.append({'controller':name,'episode':ep,'balance_seconds':steps*env.dt,'fell':terminated})
    return rows

def main():
    with open('controllers/pid/tuned_gains.yaml') as f: p=yaml.safe_load(f)['pid']; p.pop('source',None)
    with open('controllers/mpc/tuned_parameters.yaml') as f: m=yaml.safe_load(f)['mpc']; m.pop('source',None)
    rows=evaluate('pid',PIDController(**p))+evaluate('mpc',MPCController(**m))
    out=Path('results/comparisons/initial_comparison.csv'); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as stream: writer=csv.DictWriter(stream,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(out)
if __name__=='__main__': main()
