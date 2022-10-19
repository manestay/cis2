import sys

import pandas as pd

sys.path.append('scripts')
from utils import get_all_results_df

def rename_dim2cols(df, cols):
    remap = {}
    for i, label in enumerate(cols):
        _, dim = label.split('_')
        if dim.startswith('dim'):
            dim = dim[3:]
        remap[label] = dim
    return df[['model'] + cols].rename(remap, axis=1)

def print_latex(df):
    print()
    print(df.round(1).to_latex(index=False))
    print()

ALL_RESULTS = '/home1/b/bryanli/projects/stories/glucose/outputs/all_results.tsv'
df = get_all_results_df(ALL_RESULTS)
df = df.reset_index()
gen_columns = [x for x in df.columns if x.startswith('general') and '-' not in x]
spec_columns = [x for x in df.columns if x.startswith('specific') and '-' not in x]
columns_avg = ['model'] + [x for x in df.columns if 'specific_avg' in x] + [x for x in df.columns if 'general_avg' in x]

# generate the baseline tables first
df_baseline = df[df['is_baseline']]
df_baseline = df_baseline.drop(['split', 'is_baseline'], axis=1)

df_baseline_spec = rename_dim2cols(df_baseline, spec_columns)
print('specific baseline')
print_latex(df_baseline_spec)

df_baseline_gen = rename_dim2cols(df_baseline, gen_columns)
print('general baseline')
print_latex(df_baseline_gen)

# generate tables for our experiments
df_exp = df[(df['is_baseline']) & (df['split'] == 'test')]
df_exp = df_exp.drop(['split', 'is_baseline'], axis=1)

df_exp_avg = df_exp[columns_avg]
print('average exps')
print_latex(df_exp_avg)
