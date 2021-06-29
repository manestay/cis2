"""
Evaluation script for our experiments.
"""

import argparse
import os

import pandas as pd
from sacrebleu import sentence_bleu

from transformers.trainer_utils import set_seed

from local_vars import SEED, GLUCOSE_DIR, ALL_RESULTS_PATH, EXP_NUMS
from utils import add_results_row, get_all_results_df, save_results_df, select_most_likely

T5_HEADER = ['input', 'output']

parser = argparse.ArgumentParser()
parser.add_argument('exp_num')
parser.add_argument('--input_csv', '-i')
parser.add_argument('--model_size', '-ms')
group = parser.add_mutually_exclusive_group()
# group.add_argument('--baseline', action='store_true')
# group.add_argument('--random', action='store_true')
parser.add_argument('--all_results', default=ALL_RESULTS_PATH, help='path to TSV with all results')

NO_GEN = set(['1', 'A'])

def exact_match(row, hyp_name='output_pred', ref_name='output_true'):
    if row[hyp_name] == row[ref_name]:
        return 1
    return 0

def eval_bleu(row, hyp_name='output_pred', ref_name='output_true'):
    if pd.isna(row[hyp_name]): return 0
    return sentence_bleu(row[hyp_name], [row[ref_name]]).score

def print_sample(df, size=5, orig_refs=[]):
    samples = df.sample(5, random_state=SEED)
    for i, sample in enumerate(samples.itertuples()):
        print(f"example {i}")
        print(f'INPUT:  {sample.input}')
        print(f'GOLD:   {sample.output_true}')
        print(f'PRED:   {sample.output_pred}')
        if len(orig_refs):
            print(f'ORIG GOLD: {orig_refs[sample.Index]}')


def run_eval(preds_path, exp):
    df = pd.read_csv(preds_path)
    if exp in NO_GEN:
        df['exact_spec'] = df.apply(exact_match, axis=1)
        df['bleu_spec'] = df.apply(eval_bleu, axis=1)
        df['exact_gen'] = None
        df['bleu_gen'] = None
    elif exp in EXP_NUMS:
        con_gen_true = df['output_true'].str.split(' \*\* ')
        df['true_spec'] = con_gen_true.str[0]
        df['true_gen'] = con_gen_true.str[1]

        con_gen_pred = df['output_pred'].str.split(' \*\* ')
        df['pred_spec'] = con_gen_pred.str[0]
        df['pred_gen'] = con_gen_pred.str[1]
        df['exact_spec'] = df.apply(exact_match, args=('pred_spec', 'true_spec'), axis=1)
        df['bleu_spec'] = df.apply(eval_bleu, args=('pred_spec', 'true_spec'), axis=1)
        df['exact_gen'] = df.apply(exact_match, args=('pred_gen', 'true_gen'), axis=1)
        df['bleu_gen'] = df.apply(eval_bleu, args=('pred_gen', 'true_gen'), axis=1)
    else:
        return

    return df

if __name__ == "__main__":
    set_seed(SEED)
    args = parser.parse_args()
    ### TODO: rework this section, doesn't work well atm
    # if args.baseline or args.random:
    #     data_path = '/home1/b/bryanli/projects/stories/glucose/t5_data/t5_training_data.tsv'
    #     df_all = pd.read_csv(data_path, sep='\t', names=T5_HEADER)
    #     with open('/home1/b/bryanli/projects/stories/glucose/outputs/exp2a/ids_val.txt') as f:
    #         ids_val = [x.strip() for x in f.readlines()]
    #     df_val = df_all.iloc[ids_val].copy()
    #     #split_sents(df_val)
    #     most_likely = df_val.apply(select_most_likely, axis=1, include_random=args.random)
    #     df_val['most_likely'] = most_likely
    #     df_val['score'] = df_val.apply(eval_bleu, args=('most_likely', 'output_sent'), axis=1)
    #####

    exp_name = f'exp{args.exp_num}_{args.model_size}'
    if not args.input_csv:
        args.input_csv = os.path.join(GLUCOSE_DIR, 'outputs', exp_name, 'model/predictions_val.csv')

    exp = args.exp_num
    df = run_eval(args.input_csv, exp)
    if df is None:
        print(f'{args.input_csv} could not be read')
        exit()

    bleu_gen = df["bleu_gen"].mean()
    bleu_spec = df["bleu_spec"].mean()

    if exp not in NO_GEN:
        print(f'{args.input_csv}', '-' * 50)
        print(f'EM general:   {df["exact_gen"].mean()*100:.2f}')
        print(f'BLEU general: {bleu_gen:.2f}')

    print('-' * 10, f'{args.input_csv}', '-' * 10)

    if exp == 'A':
        # TODO: make a param instead of hardcoded
        ref_path = 'outputs/baseline/model/predictions_test.csv'
        ref_df = pd.read_csv(ref_path)
        orig_refs = ref_df['output_true']
    else:
        orig_refs = []
    print_sample(df, 3, orig_refs)
    print(f'EM specific:   {df["exact_spec"].mean()*100:.2f}')
    print(f'BLEU specific: {bleu_spec:.2f}')


    # get BLEU for individual dimensions
    df['dim'] = df['input'].str.split(':', 1).str[0].str.slice(1,).astype(int)
    bleu_spec_d = {}
    bleu_gen_d = {}
    for dim in range(1, 11):
        bleu_gen_d[dim] = df[df['dim'] == dim]['bleu_gen'].mean()
        bleu_spec_d[dim] = df[df['dim'] == dim]['bleu_spec'].mean()
    split = os.path.basename(args.input_csv).split('_', 1)[1].split('.', 1)[0]
    row = [exp_name, split, False, bleu_gen, bleu_spec] + [bleu_gen_d[dim] for dim in range(1, 11)] + \
        [bleu_spec_d[dim] for dim in range(1, 11)]

    df_results = get_all_results_df(args.all_results)
    df_results = add_results_row(df_results, row)
    save_results_df(df_results, args.all_results)
