# Adapted from test/evaluate_GLUCOSE_submissions.ipynb.

import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd
import sacrebleu
import numpy as np

from local_vars import GLUCOSE_DIR, ALL_RESULTS_PATH, RESULTS_COLS
from utils import to_canonical
from utils import add_results_row, get_all_results_df, save_results_df

submissions_path = f"{GLUCOSE_DIR}/outputs/exp0_base/"
key_file_path = f"{GLUCOSE_DIR}/data/nov27_key_final_copy.csv"


parser = argparse.ArgumentParser()
parser.add_argument('--submission_path', '-s', default=submissions_path)
parser.add_argument('--key_path', '-k', default=key_file_path)
parser.add_argument('--count_escaped', '-ce', action='store_true',
                    help='count `escaped` in the BLEU calucation. This is the original behavior, '
                         'but we consider it incorrect.')
parser.add_argument('--all_checkpoints', action='store_true', help='evaluate all checkpoints')
parser.add_argument('--all_results', default=ALL_RESULTS_PATH, help='path to JSON with all results')

# this ID is always `escaped`, so it has no rows in the expanded test CSV. We add it in as a row
# in the canonical CSV, so we can compare the same length.
EMPTY_ID = "6f55e830-3f7b-4b96-b9f2-022e01dca25b__4"

def main(submissions_path, key_path, count_escaped=False, all_checkpoints=False):
    model_evaluation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    df_key = pd.read_csv(key_path, encoding="utf-8")


    empty = df_key[df_key['unique_id'] == EMPTY_ID].iloc[0]
    assert empty.value_counts()['escaped'] == 20
    empty_loc = empty.name
    if all_checkpoints:
        itr = glob.glob(os.path.join(submissions_path, "**/predictions_test.csv"))
    else:
        itr = [os.path.join(submissions_path, 'model/predictions_test.csv')]
    for path in itr:
        model_name = path.rsplit('/', 3)[-3]
        print(f'processing {model_name}...')
        df_generation = pd.read_csv(path, encoding="utf-8")

        if '1_specificNL' not in df_generation.columns:
            # convert to canonical format if needed
            df_generation = to_canonical(df_generation)

        bleu_avg_gen = 0 # accumulate BLEU scores
        bleu_avg_spec = 0
        test_size = 0 # count number of examples
        for dim in range(10):
            dim = dim + 1
            for mode in ["specific", "general"]:
                predictions = df_generation[str(dim)+"_"+mode+"NL"].values
                predictions = np.insert(predictions, empty_loc, "escaped")
                references = [[], [], []]  # 3 references per each test case
                dropped_idxs = set()
                for idx, one_row in enumerate(df_key[str(dim)+"_"+mode+"NL"].values):
                    if one_row != "escaped":
                        splitted = one_row.split("****")
                        for i in range(3):
                            references[i].append(splitted[i])
                    else:
                        if count_escaped:
                            for i in range(3):
                                references[i].append("escaped")
                        else:
                            dropped_idxs.add(idx)
                if not count_escaped:
                    preds_new = []
                    for i in range(len(predictions)):
                        if i in dropped_idxs:
                            continue
                        if predictions[i] == 'escaped':
                            # print('no prediction here, using blank string')
                            preds_new.append('')
                        else:
                            preds_new.append(predictions[i])
                    preds_old = predictions
                    predictions = preds_new
                bleu = sacrebleu.corpus_bleu(predictions, references).score
                model_evaluation_results["dim"+str(dim)][mode][model_name]["bleu"] = bleu
                preds_in_dim = len(predictions)
                if mode == 'general':
                    bleu_avg_gen += bleu * preds_in_dim
                elif mode == 'specific':
                    bleu_avg_spec += bleu * preds_in_dim
            test_size += preds_in_dim
            # print(f'{preds_in_dim} in dimension {dim}')
        print(f'tested {test_size} total')
        model_evaluation_results["avg"]['general'][model_name]["bleu"] = bleu_avg_gen / test_size
        model_evaluation_results["avg"]['specific'][model_name]["bleu"] = bleu_avg_spec / test_size
    return model_evaluation_results


if __name__ == "__main__":
    args = parser.parse_args()
    model_evaluation_results = main(args.submission_path, args.key_path, args.count_escaped, args.all_checkpoints)

    rows = defaultdict(lambda: dict.fromkeys(RESULTS_COLS))
    for dim, dim_d in model_evaluation_results.items():
        print(dim)
        for mode, bleu_d in dim_d.items():
            print('  ', mode)
            for model, score in bleu_d.items():
                print('    ', model, round(score['bleu'], 2))
                row = rows[model]
                row['model'] = model
                row['split'] = 'test'
                row['is_baseline'] = True
                row[f'{mode}_{dim}'] = score['bleu']
        print('-' * 10)

    df_results = get_all_results_df(args.all_results)
    for row in rows.values():
        add_results_row(df_results, row)
    save_results_df(df_results, args.all_results)
