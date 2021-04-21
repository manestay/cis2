import argparse
import pdb
import pandas as pd
import re

from sklearn import model_selection

T5_HEADER = ['input', 'output']

parser = argparse.ArgumentParser()
parser.add_argument('exp_num', choices=['0', '1', '2a', '2b'])
parser.add_argument('train_path')
parser.add_argument('--val', action='store_true', help='split a validation set from train')
parser.add_argument('--seed', type=int, default=2557)


def get_story_ids(story_col):
    stories = story_col.unique()
    story2id = {story: i for i, story in enumerate(stories)}
    return story_col.map(story2id)


def get_in_out_df(df, exp_num):
    if exp_num == '1':
        return get_in_out_df_exp1(df)
    elif exp_num == '2a':
        return get_in_out_df_exp2a(df)
    elif exp_num == '2b':
        return get_in_out_df_exp2b(df)
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
    # discuss
    df = df[(df['story_before'] != '')].reset_index()
    df['input'] = df['dim'] + ': ' + df['story_before'].str.strip()
    df['output'] = df['output_orig']
    return df


def get_in_out_df_exp2b(df):
    # discuss
    df = df[(df['story_before'] != '')].reset_index()
    df['input'] = df['dim'] + ': ' + df['story_before'].str.strip() + ' <mask_sent> ' + \
        df['story_after'].str.strip()
    df['output'] = df['output_orig']
    return df


def make_df(df):
    '''
    Creates an intermediate df, used for later formatting of input/output. Assigns a unique `story_id` to each story

    Args:
        df (pd.Series): original T5 GLUCOSE dataset
    '''
    X_input = df['input']
    X_output = df['output']
    X_split = X_input.str.split(': ', 1, expand=True)
    dim, story = X_split[0], X_split[1]
    selected_split = story.str.split('*', 2, expand=True)
    story_before, target_sentence, story_after = selected_split[0], selected_split[1], selected_split[2]
    story = story_before + target_sentence + story_after
    story_id = get_story_ids(story)
    d = {'dim': dim, 'story_before': story_before, 'target': target_sentence,
         'story_after': story_after, 'story': story, 'story_id': story_id, 'output_orig': X_output}
    df_new = pd.DataFrame(d)
    return df_new


def manual_fix(df_train):
    old = df_train.iloc[4744].output
    # remove random letters at end of example 4744
    fixed = re.sub(r'foodd.*\)', 'food)', old)
    df_train.iloc[4744].output = fixed

    # fix double ** in a story
    df_train['input'][df_train['input'].str.count('\*') == 4] = df_train['input'][df_train['input'].str.count(
        '\*') == 4].str.replace("thing *John went for a bike ride.*", "thing John went for a bike ride.", regex=False)

    # clean up consecutive spaces
    df_train['input'] = df_train['input'].str.replace(r'\s+', ' ', regex=True)
    df_train['output'] = df_train['output'].str.replace(r'\s+', ' ', regex=True)

def format_data(train_path, exp_num, split_val, seed):
    exp_num = str(exp_num)
    df_train_orig = pd.read_csv(train_path, sep='\t', names=T5_HEADER)

    manual_fix(df_train_orig)

    df_train_orig['input'] = '#' + df_train_orig['input']

    if exp_num == 0 and not split_val:  # can just return dfs
        return df_train_orig, None, None
    df_train = make_df(df_train_orig)

    ids_val = None
    if split_val:  # split off a validation set from train based on ID
        story_ids = df_train['story_id'].unique()
        ids_train, ids_val = model_selection.train_test_split(
            story_ids, test_size=.1, random_state=seed)
        df_val = df_train[df_train['story_id'].isin(ids_val)]
        df_train = df_train[df_train['story_id'].isin(ids_train)]

    if exp_num == '0':
        return df_train, df_val, ids_val

    df_train1 = get_in_out_df(df_train, exp_num)

    df_val1 = None
    if split_val:
        df_val1 = get_in_out_df(df_val, exp_num)

    return df_train1, df_val1, ids_val


if __name__ == "__main__":
    args = parser.parse_args()
    df_train, df_val = format_data(args.train_path, args.exp_num,
                                   split_val=args.val, seed=args.seed)
