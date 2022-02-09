import re
from functools import lru_cache

import nltk
from nltk.stem import WordNetLemmatizer
from sacrebleu import sentence_bleu


def find_sim(sent, other, mode='bleu'):
    if other == '':
        return 0
    if mode == 'bleu':
        return sentence_bleu(sent, [other]).score


def select_best_match(to_match, sents, seen):
    best_inp = ''
    best_idx = -1
    best_score = -1
    for i, sent in enumerate(sents):
        if i in seen:  # can't select same sentence twice
            continue
        score = find_sim(sent, to_match, 'bleu')
        if score > best_score:
            best_inp = sent
            best_idx = i
            best_score = score
    return best_idx, best_score


def infer_cis2_label(row):
    # for the given row, generates the 3-token CIS^2 and the similarity score
    # it is most likely because we use BLEU as the heuristic to find the other sentence

    # sents = row['sents']
    # spec = row["output_spec"]
    sents = row['lemmatized']
    spec = row['spec_lemmatized']

    sel_idx = row['selected_index']  # this is given from annotations
    dim = int(row['dim'].strip('#'))
    if 1 <= dim <= 5:
        other_pos, sel_pos = 0, 2
    else:  # 6 to 10
        other_pos, sel_pos = 2, 0
    relation = f'>{spec[1]}>'
    other = spec[other_pos]
    # use heuristic to find most likely match
    best_idx, best_score = select_best_match(other, sents, [sel_idx])

    # create the 3-token sequence
    rel = [None, relation, None]
    rel[other_pos] = f'<s{best_idx}>'
    rel[sel_pos] = f'<s{sel_idx}>'
    return ' '.join(rel), round(best_score, 2)


def infer_both_cis2_label(row):
    # find both sentences using a heuristic
    # used when selected index is not known

    # sents = row['sents']
    # spec = row["output_spec"]
    sents = row['lemmatized']
    spec = row['spec_lemmatized']

    relation = f'>{spec[1]}>'

    statement_a, statement_b = spec[0], spec[2]
    best_score_tot = -1
    for ordering in [[statement_a, statement_b, False], [statement_b, statement_a, True]]:
        best_idx0, best_score0 = select_best_match(ordering[0], sents, set())
        best_idx1, best_score1 = select_best_match(ordering[1], sents, set([best_idx0]))
        score_tot = best_score0 + best_score1
        if ordering[2]:  # then we flip the order!
            best_idx0, best_idx1 = best_idx1, best_idx0
        if score_tot > best_score_tot:
            best_score_tot = score_tot
            idxs = [best_idx0, best_idx1]

    # create the 3-token sequence
    rel = [f'<s{idxs[0]}>', relation, f'<s{idxs[1]}>']
    return ' '.join(rel), round(best_score_tot, 2)


def split_output(df, output_label='output', check_len=True):
    def standardize_len(row, std_len):
        if len(row) < std_len:
            return row + [''] * (std_len - len(row))
        return row[:std_len]

    print('splitting outputs')
    output_rules = df[output_label].str.split(' \*\* ').str[0]
    output_spec = output_rules.str.split('>')
    lens = output_spec.apply(len)
    if check_len:
        assert lens.nunique() == 1  # with this split, shoudl always have len == 3
    else:
        output_spec = output_spec.apply(standardize_len, args=(3,))
    df.loc[:, 'output_spec'] = output_spec


####
wnl = WordNetLemmatizer()
lemmatize = lru_cache()(wnl.lemmatize)
pos_tag = lru_cache()(nltk.pos_tag)


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


@lru_cache()
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