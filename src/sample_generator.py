import argparse
import math
import os
import random
import sys

import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))
text_column = 'review'
output_path = os.path.normpath(os.path.join(this_dir, '..', 'data', 'output'))

s_min = 0
s_max = 2**32 - 1

confidence_levels = [90, 95, 99]
z_scores = [1.645, 1.96, 2.576]
st_dev = 0.5


def main(data, conf_level, conf_interval):
    # 1. Handle arguments and edge cases
    try:
        if data is None or conf_level is None or conf_interval is None:
            raise ValueError()
        if conf_level not in confidence_levels:
            raise ValueError()
        if conf_interval < 0:
            raise ValueError()
    except ValueError:
        print_usage()
        sys.exit(1)

    print(f"[CONFIDENCE LEVEL]: {conf_level}%")
    print(f"[CONFIDENCE INTERVAL]: {conf_interval}%\n")

    conf_interval = float(conf_interval / 100)

    # 2. Dataset loading
    print(f"Loading data from {data}...")
    try:
        if data.endswith('.csv'):
            df = pd.read_csv(data)
        elif data.endswith('.xlsx'):
            df = pd.read_excel(data)
        else:
            print("Unsupported file format. Use CSV or XLSX.")
            return
    except Exception as e:
        print(f"Error while opening file: {e}")
        return

    population = len(df)

    # 3. Calculate sample size
    ## Algorithm -> (n = N * Z^2 * p * (1 - p)) / (e^2 * (N - 1) + Z^2 * p * (1 - p))
    ### N = Population, Z = Z-Score, p = Standard Deviation, e = Confidence Interval
    z_sc = z_scores[confidence_levels.index(conf_level)]
    n_instances = math.ceil((population * pow(z_sc, 2) * st_dev * (1 - st_dev)) /
                   (pow(conf_interval, 2) * (population - 1) + pow(z_sc, 2) * st_dev * (1 - st_dev)))

    print(f"\n[SAMPLE SIZE]: {n_instances} instances")

    # 4. Sample generation and saving
    sample_df = df.sample(n=n_instances, random_state=random.randint(s_min, s_max))[['id', text_column]]
    sample_df['id'] = pd.factorize(sample_df['id'])[0] + 1

    output_file = os.path.join(output_path, "sample_data.csv")
    sample_df.to_csv(output_file, index=False)

    print(f"\nSample generation completed! Data has been saved to {output_file}")


def print_usage():
    print("ERROR: Invalid program execution\n")
    print("Basic usage: python sample_generator.py --level [CONFIDENCE_LEVEL] (90 || 95 || 99) "
          "--interval [CONFIDENCE_INTERVAL] (int)")
    print("Complete usage: python sample_generator.py --data [SAMPLE_PATH] --level [CONFIDENCE_LEVEL] (90 || 95 || 99) "
          "--interval [CONFIDENCE_INTERVAL] (int)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Maker and Analyzer")

    parser.add_argument(
        "--data",
        type=str,
        default=os.path.normpath(os.path.join(this_dir, '..', 'data', 'raw_data.csv')),
        help="Path to the input data (must be in CSV or XLSX format)"
    )

    parser.add_argument(
        "--level",
        type=int,
        help="Confidence Level desired"
    )

    parser.add_argument(
        "--interval",
        type=int,
        help="Confidence Interval (margin of error)"
    )

    # Handle not enough args passed
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    args = parser.parse_args()

    main(args.data, args.level, args.interval)