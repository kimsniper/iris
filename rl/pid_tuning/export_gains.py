import argparse, yaml
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--kp',type=float,required=True); p.add_argument('--ki',type=float,required=True); p.add_argument('--kd',type=float,required=True); a=p.parse_args()
    Path('controllers/pid/tuned_gains.yaml').write_text(yaml.safe_dump({'pid':{'kp':a.kp,'ki':a.ki,'kd':a.kd,'integral_limit':0.5,'output_limit':2.0,'source':'ppo_qualified'}}))
if __name__=='__main__': main()
