import re
from functools import lru_cache

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sacrebleu import sentence_bleu
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch.cuda import is_available

model = SentenceTransformer('all-MiniLM-L6-v2')

@lru_cache()
def get_sent_vecs(sents):
    return model.encode(sents, show_progress_bar=False, convert_to_tensor=True)

def find_sim(sent, other, sim_metric):
    '''
    sent and other are of type str if sim_metric == 'bleu', and
    of type np.array if sim_metric == 'sent_vecs'
    '''
    if other == '':
        return 0
    if sim_metric == 'sent_vecs':
        return cos_sim(sent, other).item()
    if sim_metric == 'bleu':
        return sentence_bleu(sent, [other]).score


def select_best_match(to_match, sents, seen, sim_metric):
    best_inp = ''
    best_idx = -1
    best_score = -1
    for i, sent in enumerate(sents):
        if i in seen:  # can't select same sentence twice
            continue
        score = find_sim(sent, to_match, sim_metric)
        if score > best_score:
            best_inp = sent
            best_idx = i
            best_score = score
    return best_idx, best_score


def infer_cis2_label(row, sim_metric):
    # for the given row, generates the 3-token CIS^2 and the similarity score
    # it is most likely because we use BLEU as the heuristic to find the other sentence
    if sim_metric == 'sent_vecs':
        sents = row['sent_vecs']
        spec = row["output_spec"]
    elif sim_metric == 'bleu':
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
    # print(row['output_spec'][other_pos])
    # print(row['sents'])
    if sim_metric == 'sent_vecs':
        other = get_sent_vecs(other)
    # use heuristic to find most likely match
    best_idx, best_score = select_best_match(other, sents, [sel_idx], sim_metric)

    # create the 3-token sequence
    rel = [None, relation, None]
    rel[other_pos] = f'<s{best_idx}>'
    rel[sel_pos] = f'<s{sel_idx}>'
    return ' '.join(rel), round(best_score, 2)


def infer_both_cis2_label(row, sim_metric):
    # find both sentences using a heuristic
    # used when selected index is not known

    if sim_metric == 'sent_vecs':
        sents = row['sent_vecs']
        spec = row["output_spec"]
        if is_available():
            sents = sents.to('cuda')

    elif sim_metric == 'bleu':
        sents = row['lemmatized']
        spec = row['spec_lemmatized']

    relation = f'>{spec[1]}>'

    statement_a, statement_b = spec[0], spec[2]
    if sim_metric == 'sent_vecs':
        statement_a = get_sent_vecs(statement_a)
        statement_b = get_sent_vecs(statement_b)
    best_score_tot = -1
    for ordering in [[statement_a, statement_b, False], [statement_b, statement_a, True]]:
        best_idx0, best_score0 = select_best_match(ordering[0], sents, set(), sim_metric)
        best_idx1, best_score1 = select_best_match(ordering[1], sents, set([best_idx0]), sim_metric)
        score_tot = best_score0 + best_score1
        if ordering[2]:  # then we flip the order!
            best_idx0, best_idx1 = best_idx1, best_idx0
        if score_tot > best_score_tot:
            best_score_tot = score_tot
            idxs = [best_idx0, best_idx1]

    # create the 3-token sequence
    rel = [f'<s{idxs[0]}>', relation, f'<s{idxs[1]}>']
    return ' '.join(rel), round(best_score_tot, 2)


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
