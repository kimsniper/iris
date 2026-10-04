import argparse
from controllers.ppo.ppo_controller import PPOController
from scripts._runner import run
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); a=p.parse_args(); run(PPOController(a.model))
