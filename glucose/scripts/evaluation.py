"""
Evaluation script for our experiments.
"""

import argparse
import os

import pandas as pd
from sacrebleu import sentence_bleu, corpus_bleu

from transformers.trainer_utils import set_seed

from local_vars import SEED, GLUCOSE_DIR, ALL_RESULTS_PATH, EXP_NUMS
from utils import add_results_row, get_all_results_df, save_results_df, select_most_likely

T5_HEADER = ['input', 'output']

parser = argparse.ArgumentParser()
parser.add_argument('exp_num')
parser.add_argument('--input_csv', '-i')
parser.add_argument('--model_size', '-ms')
parser.add_argument('--all_results', default=ALL_RESULTS_PATH, help='path to TSV with all results')

NO_GEN = set(['1', 'A'])

def eval_bleu(hyps, refs):
    return corpus_bleu(hyps.fillna(''), [refs]).score

def eval_bleu2(row, hyp_name='output_pred', ref_name='output_true'):
    if pd.isna(row[hyp_name]): return 0
    return sentence_bleu(row[hyp_name], [row[ref_name]]).score

def exact_match(row, hyp_name='output_pred', ref_name='output_true'):
    if row[hyp_name] == row[ref_name]:
        return 1
    return 0

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
        df['exact_gen'] = None
        bleu_spec = eval_bleu(df['output_pred'], df['output_true'])
        bleu_gen = 0.0
    elif exp in EXP_NUMS:
        con_gen_true = df['output_true'].str.split(' \*\* ')
        df['true_spec'] = con_gen_true.str[0]
        df['true_gen'] = con_gen_true.str[1]

        con_gen_pred = df['output_pred'].str.split(' \*\* ')
        df['pred_spec'] = con_gen_pred.str[0]
        df['pred_gen'] = con_gen_pred.str[1]
        df['exact_spec'] = df.apply(exact_match, args=('pred_spec', 'true_spec'), axis=1)
        df['exact_gen'] = df.apply(exact_match, args=('pred_gen', 'true_gen'), axis=1)
        bleu_spec = eval_bleu(df['pred_spec'], df['true_spec'])
        bleu_gen = eval_bleu(df['pred_gen'], df['true_gen'])
    else:
        return

    return df, bleu_spec, bleu_gen


if __name__ == "__main__":
    set_seed(SEED)
    args = parser.parse_args()

    exp_name = f'exp{args.exp_num}_{args.model_size}'
    if not args.input_csv:
        args.input_csv = os.path.join(GLUCOSE_DIR, 'outputs', exp_name, 'model/predictions_val.csv')

    exp = args.exp_num
    df, bleu_spec, bleu_gen = run_eval(args.input_csv, exp)
    if df is None:
        print(f'{args.input_csv} could not be read')
        exit()

    if exp == 'A':
        # TODO: make a param instead of hardcoded
        if 'val' in args.input_csv:
            ref_path = 'outputs/baseline/model/predictions_val.csv'
        else:
            ref_path = 'outputs/baseline/model/predictions_test.csv'
        ref_df = pd.read_csv(ref_path)
        orig_refs = ref_df['output_true']
    else:
        orig_refs = []
    print_sample(df, 3, orig_refs)

    print('-' * 10, f'{args.input_csv}', '-' * 10)
    print(f'EM specific:   {df["exact_spec"].mean()*100:.2f}')
    print(f'BLEU specific: {bleu_spec:.2f}')

    if exp not in NO_GEN:
        print(f'{args.input_csv}', '-' * 50)
        print(f'EM general:   {df["exact_gen"].mean()*100:.2f}')
        print(f'BLEU general: {bleu_gen:.2f}')


    # get BLEU for individual dimensions
    df['dim'] = df['input'].str.split(':', 1).str[0].str.slice(1,).astype(int)
    bleu_spec_d = {}
    bleu_gen_d = {}
    for dim in range(1, 11):
        df_dim = df[df['dim'] == dim]
        if exp in NO_GEN:
            bleu_spec_d[dim] = eval_bleu(df_dim['output_pred'], df_dim['output_true'])
            bleu_gen_d[dim] = 0.0
        else:
            bleu_spec_d[dim] = eval_bleu(df_dim['pred_spec'], df_dim['true_spec'])
            bleu_gen_d[dim] = eval_bleu(df_dim['pred_gen'], df_dim['true_gen'])

    df_dim_15 = df[df['dim'] <= 5]
    df_dim_610 = df[df['dim'] >= 6]

    if exp in NO_GEN:
        bleu_spec_15 = eval_bleu(df_dim_15['output_pred'], df_dim_15['output_true'])
        bleu_spec_610 = eval_bleu(df_dim_610['output_pred'], df_dim_610['output_true'])
        bleu_gen_15 = 0.0
        bleu_gen_610 = 0.0
    else:
        bleu_spec_15 = eval_bleu(df_dim_15['pred_spec'], df_dim_15['true_spec'])
        bleu_spec_610 = eval_bleu(df_dim_610['pred_spec'], df_dim_610['true_spec'])
        bleu_gen_15 = eval_bleu(df_dim_15['pred_gen'], df_dim_15['true_gen'])
        bleu_gen_610 = eval_bleu(df_dim_610['pred_gen'], df_dim_610['true_gen'])

    split = os.path.basename(args.input_csv).split('_', 1)[1].split('.', 1)[0]
    row = [exp_name, split, False, bleu_spec, bleu_gen] + [bleu_spec_d[dim] for dim in range(1, 11)] \
        + [bleu_gen_d[dim] for dim in range(1, 11)] \
        + [bleu_spec_15, bleu_spec_610, bleu_gen_15, bleu_gen_610]

    df_results = get_all_results_df(args.all_results)
    df_results = add_results_row(df_results, row)
    save_results_df(df_results, args.all_results)
