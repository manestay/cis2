# TODO: convert the entire TF_HF.ipynb notebook to a script. For now we just have some useful functions.

import transformers

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
