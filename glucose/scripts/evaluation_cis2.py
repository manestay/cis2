import argparse
from pathlib import Path
import os

import pandas as pd

from transformers.trainer_utils import set_seed
from nltk.tokenize import sent_tokenize

from local_vars import CIS2_RESULTS_PATH, CIS2_TEST_PATH, EXP_NUMS, SEED, GLUCOSE_DIR, DELIM
from preprocess import get_spec_col
from utils import lemmatize_sents, split_output, infer_both_cis2_label

parser = argparse.ArgumentParser()
parser.add_argument('exp_num', choices=EXP_NUMS)
parser.add_argument('--input_csv', '-i')
parser.add_argument('--model_size', '-ms', required=True)
parser.add_argument('--results', default=CIS2_RESULTS_PATH, help='path to TSV with CIS2 results')
parser.add_argument('--ref', default=CIS2_TEST_PATH)

ABS = set(['A', 'cis2'])

def fix_spacing(s):
    return s.replace('>>', '> >').replace('><', '> <')

if __name__ == "__main__":
    set_seed(SEED)
    args = parser.parse_args()

    exp_name = f'exp{args.exp_num}_{args.model_size}'
    if not args.input_csv:
        args.input_csv = os.path.join(GLUCOSE_DIR, 'outputs', exp_name, 'model/predictions_test.csv')

    df_ref = pd.read_csv(args.ref, sep='\t')
    df_pred = pd.read_csv(args.input_csv)
    df_pred['selected_index'] = df_ref['selected_index']
    df_pred['dim'] = df_ref['dim']
    df_pred['sents'] = df_ref['story']
    df_pred['lemmatized'] = df_ref['lemmatized'].str.split(DELIM)

    preds = df_pred['output_pred']
    if args.exp_num not in ABS: # convert predicted specific rule to CIS2 labels
        split_output(df_pred, 'output_pred', check_len=False)
        df_pred['spec_lemmatized'] = get_spec_col(df_pred['output_spec'])

        print('selecting most likely...')
        df_pred['cis2_pred'], df_pred['sim_score'] = zip(*df_pred.progress_apply(infer_both_cis2_label, axis=1))

    else:
        df_pred['cis2_pred'] = df_pred['output_pred'].apply(fix_spacing)
    print(sum(df_pred['cis2_pred'] == df_ref['output']))
