import os
import re
from functools import lru_cache

import pandas as pd
from sacrebleu import sentence_bleu
from tqdm import tqdm
from transformers import AutoTokenizer

from local_vars import CANONICAL_COLS, RESULTS_COLS

import nltk
from nltk.corpus import wordnet

tqdm.pandas()

def canonical_dict():
    return dict.fromkeys(CANONICAL_COLS)


def to_canonical(preds):
    dim_story = preds['input'].str.split(": ", 1)
    preds['dim'] = dim_story.str[0].str.lstrip("#")
    preds['story'] = dim_story.str[1]

    story = ''
    canon_row = canonical_dict()
    canon_rows = []
    seen = set()
    for row in preds.itertuples():
        if 'story_id' in row._fields:
            story_id = row.story_id
        else:
            story_id = row.unique_id.split('__')[0]

        if row.story != story:
            if story:
                for k, v in canon_row.items():
                    if not v:
                        canon_row[k] = 'escaped'
                canon_rows.append(canon_row)
                # assert story_id not in seen
                seen.add(story_id)

            canon_row = canonical_dict()

        # now we add to each row
        story = row.story
        canon_row['story_id'] = story_id
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
            print(f'WARNING: could not split row {row.Index} into specific and general:')
            print(row.output_pred)
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
    elif exp_num == 'A':
        extra_tokens = ['<s0>', '<s1>', '<s2>', '<s3>', '<s4>']
        tokenizer.add_tokens(extra_tokens)
    return tokenizer

## functions for working with the results df
def get_all_results_df(path):
    if not os.path.exists(path):
        df = pd.DataFrame([], columns=RESULTS_COLS)
        df.set_index(['model', 'split', 'is_baseline'], inplace=True)
        return df

    df = pd.read_csv(path, sep='\t')
    df.set_index(['model', 'split', 'is_baseline'], inplace=True)
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
            idx = tuple([row[0] + action] + row[1:3])
        else:
            return df
    df.loc[idx] = row_scores
    return df

def save_results_df(df, path, round=2):
    if round:
        df = df.round(round)
    df.sort_index(inplace=True)
    df.to_csv(path, sep='\t')

##
def find_sim(sent, other, mode='bleu'):
    if mode == 'bleu':
        return sentence_bleu(sent, [other]).score

def select_most_likely(row):
    # for the given row, generates the 3-token CIS^2 and the similarity score
    # it is most likely because we use BLEU as the heuristic to find the other sentence

    # sents = row['sents']
    # spec = row["output_spec"]
    sents = row['lemmatized']
    spec = row['spec_lemmatized']

    sel_idx = row['selected_index'] # this is given from annotations
    dim = int(row['dim'].strip('#'))
    if 1 <= dim <= 5:
        other_pos, sel_pos = 0, 2
    else: # 6 to 10
        other_pos, sel_pos = 2, 0
    relation = f'>{spec[1]}>'
    other = spec[other_pos]
    # use heuristic to find most likely match
    best_inp = ''
    best_idx = -1
    best_score = -1
    for i, sent in enumerate(sents):
        if i == sel_idx: # can't select same sentence twice
            continue
        score = find_sim(sent, other, 'bleu')
        if score > best_score:
            best_inp = sent
            best_idx = i
            best_score = score
    # create the 3-token sequence
    rel = [None, relation, None]
    rel[other_pos] = f'<s{best_idx}>'
    rel[sel_pos] = f'<s{sel_idx}>'
    return ' '.join(rel), round(best_score, 2)

def split_output(df, output_label='output'):
    print('splitting outputs')
    output_rules = df[output_label].str.split(' \*\* ').str[0]
    output_spec = output_rules.str.split('>')
    lens = output_spec.apply(len)
    assert lens.nunique() == 1 # with this split, shoudl always have len == 3
    df.loc[:, 'output_spec'] = output_spec

####
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)
pos_tag = lru_cache(maxsize=50000)(nltk.pos_tag)

def wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

@lru_cache(maxsize=50000)
def lemmatize_sent(sent):
    pos_tagged = pos_tag(tuple(nltk.word_tokenize(sent)))
    wordnet_tagged = list(map(lambda x: (x[0], wordnet_pos(x[1])), pos_tagged))
    sent_lem = [lemmatize(word, tag) if tag else lemmatize(word) for word, tag in wordnet_tagged]
    if len(sent_lem) and sent_lem[-1] == '.':
        del sent_lem[-1]
    sent_lem = ' '.join(sent_lem)

    return sent_lem

def lemmatize_sents(sents):
    sents_lem = []
    for sent in sents:
        sent_lem = lemmatize_sent(sent)
        sents_lem.append(sent_lem)
    return sents_lem
