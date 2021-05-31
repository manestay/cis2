import argparse
import logging
import re

import pandas as pd
from sklearn import model_selection

parser = argparse.ArgumentParser()
parser.add_argument('exp_num', choices=['0', '1', '2a', '2b', '3a', '3b'])
parser.add_argument('--train_path', default='data_final/GLUCOSE_training_data_final.csv')
parser.add_argument('--test_path', default='data_final/nov27_key_final_copy.csv')
parser.add_argument('--no_logging', dest='logging', action='store_false')

parser.add_argument('--seed', type=int, default=2557)
parser.add_argument('--split_val', action='store_true',
                   help='split a validation set from train (deprecated)')


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
    # import pdb; pdb.set_trace()
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
         'story_after': story_after, 'story': story, 'story_id': story_id, 'output_orig': X_output,
         'experiment_id': experiment_id}
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


def format_data(train_path, exp_num, split_val=False, orig_val=True, seed=0, is_test=False):
    exp_num = str(exp_num)
    df_train_orig = pd.read_csv(train_path)
    logging.debug(f"loaded original train CSV {train_path} ({len(df_train_orig)} rows)")

    df_train_ex = format_for_t5(df_train_orig, is_test=is_test)
    if not is_test:
        manual_fix(df_train_ex)
    logging.debug(f"expanded each experiment to 1 dimension per row {train_path} ({len(df_train_ex)} rows)")

    if split_val:  # split off a validation set from train based on ID
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
    if split_val:
        df_val = split_contexts(df_val_ex)
    logging.debug(f"split stories into before/target sentence/after")

    if exp_num == '0':
        return df_train, df_val, ids_val

    df_train1 = get_in_out_df(df_train, exp_num)
    df_val1 = None
    if split_val:
        df_val1 = get_in_out_df(df_val, exp_num)
    return df_train1, df_val1, ids_val


if __name__ == "__main__":
    args = parser.parse_args()
    if args.logging:
        logging.basicConfig(level=logging.DEBUG)

    df_train, df_val, ids_val = format_data(args.train_path, args.exp_num,
                                            split_val=args.split_val, seed=args.seed)
    df_test, _, _ = format_data(args.test_path, args.exp_num,
                                            split_val=False, seed=args.seed, is_test=True)
