import argparse, pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv'); a=p.parse_args(); df=pd.read_csv(a); print(df.groupby('controller').mean(numeric_only=True))
if __name__=='__main__': main()
