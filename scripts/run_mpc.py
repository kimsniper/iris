import yaml
from controllers.mpc.mpc_controller import MPCController
from scripts._runner import run
with open('controllers/mpc/tuned_parameters.yaml') as f: cfg=yaml.safe_load(f)['mpc']
run(MPCController(**{k:v for k,v in cfg.items() if k != 'source'}))
