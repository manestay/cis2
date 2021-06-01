"""
Preprocesses the datasets, and also tokenizes them.

NOTE: may want to merge preprocess.py and format_data.py
"""

import argparse
import logging
import os
import re

import datasets
import pandas as pd
from sklearn import model_selection
from transformers import AutoTokenizer

from local_vars import EXP_NUMS, SAVE_DIR, TRAIN_PATH, TEST_PATH, BATCH_SIZE_ENCODE, COLS_TO_FORMAT, SEED

parser = argparse.ArgumentParser()
parser.add_argument('exp_num', choices=EXP_NUMS)
parser.add_argument('--train_path', default=TRAIN_PATH)
parser.add_argument('--test_path', default=TEST_PATH)
parser.add_argument('--seed', type=int, default=SEED)
parser.add_argument('--split_val', action='store_true',
                   help='split a validation set from train (deprecated)')
parser.add_argument('--val_ids', default=None, help='specify story IDs to use as the validation set. '
                    'Supersedes --split_val if both are used.')

parser.add_argument('--no_logging', dest='logging', action='store_false')

# uses the parser from format_data.py, with additional arguments
parser.add_argument('--dataset_dir')
parser.add_argument('--model_size', '-m', default='t5-base')
parser.add_argument('--out_location', default=SAVE_DIR)
parser.set_defaults(val_ids=f'{SAVE_DIR}/val_ids.txt')

def get_src_tgt_len(source_text, target_text, tokenizer):
    tokenized_source_text = tokenizer(list(source_text), truncation=False, padding=False)
    tokenized_target_text = tokenizer(list(target_text), truncation=False, padding=False)

    max_source = 0
    for item in tokenized_source_text['input_ids']:
        if len(item) > max_source:
            max_source = len(item)

    max_target = 0
    for item in tokenized_target_text['input_ids']:
        if len(item) > max_target:
            max_target = len(item)
    return max_source, max_target

def encode(batch, tokenizer, max_source, max_target):
    inp = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=max_source)
    if 'output' in batch:
        outp = tokenizer(batch['output'], padding='max_length', truncation=True, max_length=max_target)
        inp['labels'] = outp['input_ids']
    return inp

def preprocess(args):
    # format data for the given experiment
    logging.debug(f'formatting data for experiment {args.exp_num}')
    df_train, df_val, ids_val = format_data(args.train_path, args.exp_num, val_ids=args.val_ids, seed=args.seed)
    df_test, _, _ = format_data(args.test_path, args.exp_num, split_val=False, seed=args.seed, is_test=True)

    logging.debug(f"size of train: {len(df_train)}")
    logging.debug(f"size of validation: {len(df_val)}")

    ex = df_train.iloc[200]
    logging.debug('sample input/output:')
    logging.debug(f'input: {ex["input"]}')
    logging.debug(f'output: {ex["output"]}')

    # tokenize the data
    logging.debug(f'loading tokenizer for {args.model_size}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_size)

    if args.exp_num == '2b':
        special_tokens_dict = {'additional_special_tokens': ['<mask_sent>']}
        add_toks = tokenizer.add_special_tokens(special_tokens_dict)

    ds_train = datasets.Dataset.from_pandas(df_train)
    ds_val = datasets.Dataset.from_pandas(df_val)
    ds_test = datasets.Dataset.from_pandas(df_test)

    logging.debug('calculating max sequence length...')
    max_source, max_target = get_src_tgt_len(df_train['input'], df_train['output'], tokenizer)
    logging.debug(f'number of tokens:\nmax input tokens: {max_source}\nmax output tokens: {max_target}')

    logging.debug('tokenizing datasets...')
    kwargs = dict(max_source=max_source, max_target=max_target, tokenizer=tokenizer)
    ds_train = ds_train.map(encode, batched=True, batch_size=BATCH_SIZE_ENCODE, fn_kwargs=kwargs)
    ds_val = ds_val.map(encode, batched=True, batch_size=BATCH_SIZE_ENCODE, fn_kwargs=kwargs)
    ds_test = ds_test.map(encode, batched=True, batch_size=BATCH_SIZE_ENCODE, fn_kwargs=kwargs)

    ds_train.set_format(type='torch', columns=COLS_TO_FORMAT)
    ds_val.set_format(type='torch', columns=COLS_TO_FORMAT)
    ds_test.set_format(type='torch', columns=[x for x in COLS_TO_FORMAT if x != 'labels'])

    # verify proper encoding
    logging.debug('example text after encoding and decoding:')
    logging.debug(tokenizer.decode(ds_val[0]['input_ids']))
    logging.debug(tokenizer.decode(ds_val[0]['labels']))

    logging.debug(f'saving tokenized datasets to disk at {args.dataset_dir}')
    ds_train.save_to_disk(f'{args.dataset_dir}/ds_train')
    ds_val.save_to_disk(f'{args.dataset_dir}/ds_val')
    ds_test.save_to_disk(f'{args.dataset_dir}/ds_test')


def get_in_out_df(df, exp_num):
    if exp_num == '1':
        return get_in_out_df_exp1(df)
    elif exp_num == '2a':
        return get_in_out_df_exp2a(df)
    elif exp_num == '2b':
        return get_in_out_df_exp2b(df)
    elif exp_num == '3a':
        return get_in_out_df_exp3a(df)
    elif exp_num == '3b':
        return get_in_out_df_exp3b(df)
    else:
        print('invalid exp num!')


def get_in_out_df_exp1(df):
    # for next sentence task, we exclude cases where there are no sentences before
    # doesn't work well because we're not using the causal information
    df = df[(df['story_before'] != '')].reset_index()
    df['input'] = df['dim'] + ': ' + df['story_before'].str.strip()
    df['output'] = df['target']
    return df


def get_in_out_df_exp2a(df):
    df = df[(df['story_before'] != '')].reset_index()
    df['input'] = df['dim'] + ': ' + df['story_before'].str.strip()
    df['output'] = df['output_orig']
    return df


def get_in_out_df_exp2b(df):
    df = df[(df['story_before'] != '')].reset_index()
    df['input'] = df['dim'] + ': ' + df['story_before'].str.strip() + ' <mask_sent> ' + \
        df['story_after'].str.strip()
    if 'output_orig' in df.columns:
        df['output'] = df['output_orig']
    return df


def get_in_out_df_exp3a(df):
    # after instead of before, since we have at least 1
    df = df[(df['story_after'] != '')].reset_index()
    target_highlighted = df['target'].apply(lambda x: f' *{x}*')
    df['input'] = df['dim'] + ': ' + df['story_before'].str.strip() + target_highlighted
    df['output'] = df['output_orig']
    return df


def get_in_out_df_exp3b(df):
    # after instead of before, since we have at least 1
    df = df[(df['story_after'] != '')].reset_index()
    target_highlighted = df['target'].apply(lambda x: f' *{x}*')
    df['input'] = df['dim'] + ': ' + df['story_before'].str.strip() + target_highlighted
    df['output'] = df['output_orig']
    return df

def format_for_t5(df, is_test=False):
    def rename_NL(col):
        if not col[0].isdigit():
            return col
        if col[0:2] == '10':
            return f"{col[3:]}_{col[0:2]}"
        return f"{col[2:]}_{col[0]}"

    df = df.rename(columns=lambda x: rename_NL(x))
    rows_expanded = []

    for row in df.itertuples():
        for i in range(1, 11):
            row_ex = []
            if is_test:
                row_ex.append(row.unique_id)
                row_ex.append(row.unique_id.split("__", 1)[0])
            if not is_test:
                row_ex.append(row.experiment_id)
                row_ex.append(row.story_id)

            specific_name = f"specificNL_{i}"
            general_name = f"generalNL_{i}"
            specific = getattr(row, specific_name)
            general = getattr(row, general_name)
            if specific == 'escaped' or general == 'escaped':
                continue

            story = row.story.replace('****', ' ') # test has this
            selected_sentence = row.selected_sentence
            escaped = re.escape(selected_sentence)
            story = re.sub(escaped, f"*{selected_sentence}*", story, 1)
            story = f"#{i}: {story}"
            row_ex.append(story)
            # assert story.count("*") == 2

            if is_test:
                for specific_st, general_st in zip(specific.split("****"), general.split("****")):
                    row_ex.append(specific_st)
                    row_ex.append(general_st)
            else:
                row_ex.append(specific)
                row_ex.append(general)

            rows_expanded.append(row_ex)

    if is_test:
        COLS = ['experiment_id', 'story_id', 'input', 'specific_ref1', 'general_ref1', 'specific_ref2', 'general_ref2',
                'specific_ref3', 'general_ref3']
        df_expanded = pd.DataFrame(rows_expanded, columns=COLS)
        df_expanded['output'] = df_expanded['specific_ref1'] + ' ** ' + df_expanded['general_ref1']

    else:
        COLS = ['experiment_id', 'story_id', 'input', 'specific', 'general']
        df_expanded = pd.DataFrame(rows_expanded, columns=COLS)
        df_expanded['output'] = df_expanded['specific'] + ' ** ' + df_expanded['general']
    return df_expanded


def split_contexts(df):
    '''
    Creates an intermediate df, used for later formatting of input/output. Assigns a unique `story_id` to each story

    Args:
        df (pd.Series): T5 GLUCOSE dataset, where each row is 1 dimension for a selected experiment
    '''
    X_input = df['input']
    X_output = df['output']
    X_split = X_input.str.split(': ', 1, expand=True)
    dim, story = X_split[0], X_split[1]
    selected_split = story.str.split('*', 2, expand=True)
    story_before, target_sentence, story_after = selected_split[0], selected_split[1], selected_split[2]
    story = story_before + target_sentence + story_after
    story_id = df['story_id']
    experiment_id = df['experiment_id']
    d = {'dim': dim, 'story_before': story_before, 'target': target_sentence,
         'story_after': story_after, 'story': story, 'story_id': story_id,
         'experiment_id': experiment_id, 'output_orig': X_output}
    df_new = pd.DataFrame(d)
    return df_new


def manual_fix(df_train):
    bad_index = df_train['output'][df_train['output'].str.match(".*foodd.*\)")].index[0]
    old = df_train.loc[bad_index]['output']
    # remove random letters at end of example 4744
    fixed = re.sub(r'foodd.*\)', 'food)', old)
    df_train.loc[bad_index, 'output'] = fixed

    # clean up consecutive spaces
    df_train['input'] = df_train['input'].str.replace(r'\s\s+', ' ', regex=True)
    df_train['output'] = df_train['output'].str.replace(r'\s\s+', ' ', regex=True)

    # some inputs are all capitalized -- change them to lower so BPE works
    all_upper = df_train['input'].apply(lambda x: not any(char.islower() for char in x))
    all_upper_ind = df_train[all_upper].loc[:, 'input'].index
    df_train.loc[all_upper_ind, 'input'] = df_train.loc[all_upper_ind, 'input'].str.lower()


def format_data(train_path, exp_num, split_val=False, val_ids=None, seed=0, is_test=False):
    if val_ids and split_val:
        print('WARNING: both split val and val_ids were specified; only using val_ids')
    exp_num = str(exp_num)
    df_train_orig = pd.read_csv(train_path)
    logging.debug(f"loaded original train CSV {train_path} ({len(df_train_orig)} rows)")

    df_train_ex = format_for_t5(df_train_orig, is_test=is_test)
    if not is_test:
        manual_fix(df_train_ex)
    logging.debug(f"expanded each experiment to 1 dimension per row {train_path} ({len(df_train_ex)} rows)")
    if val_ids:
        logging.debug(f'loading validation ids from {val_ids}...')
        with open(val_ids) as f:
            ids_val = [x.strip() for x in f.readlines()]
        df_val_ex = df_train_ex[df_train_ex['story_id'].isin(ids_val)]
        df_train_ex = df_train_ex[~df_train_ex['story_id'].isin(ids_val)]
    elif split_val:  # split off a validation set from train based on ID
        logging.debug(f'splitting validation set from train...')
        story_ids = df_train_ex['story_id'].unique()
        ids_train, ids_val = model_selection.train_test_split(
            story_ids, test_size=.1, random_state=seed)
        df_val_ex = df_train_ex[df_train_ex['story_id'].isin(ids_val)]
        df_train_ex = df_train_ex[df_train_ex['story_id'].isin(ids_train)]
    else:
        df_val_ex = None
        ids_val = []

    if exp_num == '0': # original task, just return
        return df_train_ex, df_val_ex, ids_val

    # else we need to reformat
    df_train = split_contexts(df_train_ex)
    df_val = None
    if split_val or val_ids:
        df_val = split_contexts(df_val_ex)
    logging.debug(f"split stories into before/target sentence/after")

    if exp_num == '0':
        return df_train, df_val, ids_val

    df_train1 = get_in_out_df(df_train, exp_num)
    df_val1 = None
    if split_val or val_ids:
        df_val1 = get_in_out_df(df_val, exp_num)
    return df_train1, df_val1, ids_val

if __name__ == "__main__":
    args = parser.parse_args()
    if args.logging:
        logging.basicConfig(level=logging.DEBUG)
    if not args.dataset_dir:
        args.dataset_dir = f'{SAVE_DIR}/exp{args.exp_num}_{args.model_size}'
    os.makedirs(args.dataset_dir, exist_ok=True)
    preprocess(args)