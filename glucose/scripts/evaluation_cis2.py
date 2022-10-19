import argparse
from pathlib import Path
import os

import dill as pickle
import pandas as pd

from transformers.trainer_utils import set_seed
from nltk.tokenize import sent_tokenize

from local_vars import CIS2_TEST_PATH, EXP_NUMS, SEED, GLUCOSE_DIR, DELIM, METRICS, \
                       TEST_STORY_VECS
from preprocess import get_spec_col
from utils import split_output, infer_both_cis2_label, get_sent_vecs, get_exp_name

parser = argparse.ArgumentParser()
parser.add_argument('exp_num', choices=EXP_NUMS)
parser.add_argument('--input_csv', '-i')
parser.add_argument('--model_size', '-ms', required=True)
parser.add_argument('--ref', default=CIS2_TEST_PATH)
parser.add_argument('--sim_metric', '-sm', default='bleu', choices=METRICS)
parser.add_argument('--recalc-vecs', action='store_true')

ABS = set(['A', 'cis2'])

def fix_spacing(s):
    return s.replace('>>', '> >').replace('><', '> <')

if __name__ == "__main__":
    set_seed(SEED)
    args = parser.parse_args()

    exp_name = get_exp_name(args.exp_num, args.model_size, args.sim_metric)
    if not args.input_csv:
        args.input_csv = os.path.join(GLUCOSE_DIR, 'outputs', exp_name, 'model/predictions_test.csv')
        print(f'loading from {args.input_csv}')

    df_ref = pd.read_csv(args.ref, sep='\t')
    df_pred = pd.read_csv(args.input_csv)
    df_pred['selected_index'] = df_ref['selected_index']
    df_pred['dim'] = df_ref['dim']

    if args.exp_num not in ABS: # convert predicted specific rule to CIS2 labels
        if args.sim_metric == 'bleu':
            df_pred['lemmatized'] = df_ref['lemmatized'].str.split(DELIM)
        elif args.sim_metric == 'sent_vecs':
            df_pred['sents'] = df_ref['sents'].str.split(DELIM).apply(tuple)
            if args.recalc_vecs or not os.path.exists(TEST_STORY_VECS):
                print('calculating sentence vectors')
                df_pred['sent_vecs'] = df_pred['sents'].progress_apply(get_sent_vecs)
                with open(TEST_STORY_VECS, 'wb') as f:
                    pickle.dump(df_pred['sent_vecs'], f)
                    print(f'wrote test story sent vectors to {TEST_STORY_VECS}')
            else:
                print(f'loading sentence vectors from {TEST_STORY_VECS}, pass --recalc-vecs to not load')
                with open(TEST_STORY_VECS, 'rb') as f:
                    df_pred['sent_vecs'] = pickle.load(f)
        split_output(df_pred, 'output_pred', check_len=False)
        if args.sim_metric == 'bleu':
            df_pred['spec_lemmatized'] = get_spec_col(df_pred['output_spec'])

        print('selecting most likely...')
        df_pred['cis2_pred'], df_pred['sim_score'] = zip(*df_pred.progress_apply(infer_both_cis2_label, args=(args.sim_metric,), axis=1))

    else:
        df_pred['cis2_pred'] = df_pred['output_pred'].apply(fix_spacing)

    num_correct = sum(df_pred['cis2_pred'] == df_ref['output'])
    # num_correct = sum(df_pred['output_true'] == df_pred['output_pred'])
    print(f'correct: {num_correct}/{len(df_pred)}')
    score = num_correct / len(df_pred)
    print(f'CIS2 score: {score*100:.1f}')
