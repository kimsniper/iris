import argparse, yaml
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--q-pitch',type=float,required=True); p.add_argument('--q-pitch-rate',type=float,required=True); p.add_argument('--r-input',type=float,required=True); a=p.parse_args()
    values={'dt':0.02,'horizon':20,'q_pitch':a.q_pitch,'q_pitch_rate':a.q_pitch_rate,'q_position':1.0,'q_velocity':1.0,'r_input':a.r_input,'output_limit':2.0,'source':'ppo_qualified'}
    Path('controllers/mpc/tuned_parameters.yaml').write_text(yaml.safe_dump({'mpc':values}))
if __name__=='__main__': main()
