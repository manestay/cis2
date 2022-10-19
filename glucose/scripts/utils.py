import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from local_vars import CANONICAL_COLS, RESULTS_COLS
from cis2_lib import *

tqdm.pandas()


def split_output(df, output_label='output', check_len=True):
    def standardize_len(row, std_len):
        if len(row) < std_len:
            return row + [''] * (std_len - len(row))
        return row[:std_len]

    print('splitting outputs')
    output_rules = df[output_label].str.split(' \*\* ').str[0]
    output_spec = output_rules.str.split('>').map(lambda x: [s.strip() for s in x])
    lens = output_spec.apply(len)
    if check_len:
        assert lens.nunique() == 1  # with this split, shoudl always have len == 3
    else:
        output_spec = output_spec.apply(standardize_len, args=(3,))
    df.loc[:, 'output_spec'] = output_spec


def canonical_dict():
    return dict.fromkeys(CANONICAL_COLS)

def get_story_id(row):
    if 'story_id' in row:
        story_id = row.story_id
    else:
        story_id = row.unique_id.split('__')[0]
    return story_id

def to_canonical(preds):
    dim_story = preds['input'].str.split(":", 1)
    preds['dim'] = dim_story.str[0].str.lstrip("#")
    preds['story'] = dim_story.str[1].str.strip()
    preds['story_id'] = preds.apply(get_story_id, axis=1)

    canon_row = None
    canon_rows = []
    unique_id_prev = None
    preds = preds.sort_values('unique_id')
    for row in preds.itertuples():
        if row.unique_id != unique_id_prev:
            if canon_row:
                for k, v in canon_row.items():
                    if not v:
                        canon_row[k] = 'escaped'
                canon_rows.append(canon_row)

            canon_row = canonical_dict()
        unique_id_prev = row.unique_id
        # now we add to each row
        canon_row['story_id'] = row.story_id
        canon_row['unique_id'] = row.unique_id
        general = f'{row.dim}_generalNL'
        specific = f'{row.dim}_specificNL'
        pred_split = row.output_pred.split("**", 1)
        # output is rarely incorrectly formatted, try to split
        if len(pred_split) == 1:
            pred_split = row.output_pred.split("*", 1)

        # if failed to split, take whole row as specific
        if len(pred_split) == 1:
            pred_spec = pred_split[0]
            pred_gen = ''
            print(f'WARNING: could not split row {row.Index} into specific and general')
        else:
            pred_spec, pred_gen = [x.strip() for x in pred_split]
        canon_row[general] = pred_gen
        canon_row[specific] = pred_spec
    if canon_row:
        for k, v in canon_row.items():
            if not v:
                canon_row[k] = 'escaped'
        canon_rows.append(canon_row)
    df = pd.DataFrame(canon_rows)

    return df

def load_tokenizer(model_size, exp_num):
    tokenizer = AutoTokenizer.from_pretrained(model_size)
    if exp_num == '2b':
        special_tokens_dict = {'additional_special_tokens': ['<mask_sent>']}
        tokenizer.add_special_tokens(special_tokens_dict)
    elif exp_num == 'cis2':
        sent_tokens = ['<s0>', '<s1>', '<s2>', '<s3>', '<s4>']
        rel_tokens = ['>Causes/Enables>', '>Enables>', '>Results in>', '>Motivates>', '>Causes>']
        tokenizer.add_tokens(sent_tokens)
        tokenizer.add_tokens(rel_tokens)
    return tokenizer

## functions for working with the results df
def get_all_results_df(path):
    if not os.path.exists(path):
        df = pd.DataFrame([], columns=RESULTS_COLS)
        df.set_index(['split', 'is_baseline', 'model'], inplace=True)
        return df

    df = pd.read_csv(path, sep='\t')
    df.set_index(['split', 'is_baseline', 'model'], inplace=True)
    return df

def add_results_row(df, row):
    if isinstance(row, dict):
        row = [row[x] for x in RESULTS_COLS]

    idx = tuple(row[0:3])
    row_scores = row[3:]
    if idx in df.index:
        action = input(f'overwrite {idx} in results df? (y/n,  or enter suffix beginning with _)\n')
        if action == 'y':
            pass
        elif action.startswith('_'):
            idx = tuple(row[0:2] + [row[2] + action])
        else:
            return df
    df.loc[idx] = row_scores
    return df

def save_results_df(df, path, round=2):
    if round:
        df = df.round(round)
    df.sort_index(inplace=True)
    df.to_csv(path, sep='\t')

def get_exp_name(exp_num, model_size, sim_metric, specific_only=False):
    if exp_num == 'baseline':
        return exp_num
    exp_name = f'exp{exp_num}_{model_size}'
    if exp_num == 'cis2':
        exp_name += f'_{sim_metric}'
    if specific_only:
        exp_name += '_specific_only'
    return exp_name
