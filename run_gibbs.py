import argparse
import numpy as np
import pandas as pd
import simulate_data as sim


def main():
    parser = argparse.ArgumentParser(description="Mnozy n razy d")
    parser.add_argument("-n", type=float, required=True, help="Wartosc n")
    parser.add_argument("-d", type=float, required=True, help="Wartosc d")

    args = parser.parse_args()

    wynik = args.n * args.d
    print(wynik)

    df = pd.DataFrame([{"x":1, "y":2}, {"x":1, "y":2}])
    df.to_csv("xxx.csv")

    X = sim.X_basic(int(args.n), int(args.d))
    

if __name__ == "__main__":
    main()