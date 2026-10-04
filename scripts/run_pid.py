import yaml
from controllers.pid.pid_controller import PIDController
from scripts._runner import run
with open('controllers/pid/tuned_gains.yaml') as f: cfg=yaml.safe_load(f)['pid']
run(PIDController(**{k:v for k,v in cfg.items() if k!='source'}))
