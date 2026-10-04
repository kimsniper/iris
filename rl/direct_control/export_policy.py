import argparse, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); a=p.parse_args(); out=Path('models/direct_control/best_model.zip'); out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(a.model,out); print(out)
if __name__=='__main__': main()
