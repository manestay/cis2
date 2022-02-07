"""
Formats a raw GLUCOSE dataset into a CIS^2 formatted dataset.
"""


import argparse
import os

from local_vars import SAVE_DIR, SEED, TEST_PATH
from preprocess import format_data

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', default=TEST_PATH)
parser.add_argument('--seed', type=int, default=SEED)
parser.add_argument('--out_path', default=f'{SAVE_DIR}/test_cis2.tsv')

def main(args):
    pass

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    df, _, _ = format_data(args.test_path, 'A',
                                split_val=False, seed=args.seed, is_test=True)
    df = df[['input', 'output', 'output_orig','sim_score']]
    df.to_csv(args.out_path, sep='\t', index=True)
