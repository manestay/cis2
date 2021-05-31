import argparse
from pathlib import Path
from os.path import isdir
import pdb

import datasets
import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.trainer_utils import set_seed

MODEL_DIR = "/home1/b/bryanli/projects/stories/glucose/outputs/exp0"
DATA_PATH = f"{MODEL_DIR}/ds_val"


parser = argparse.ArgumentParser()
parser.add_argument('exp_num', type=str)
parser.add_argument('--model_dir', '-m', default=MODEL_DIR)
parser.add_argument('--dataset_dir', '-d', default=DATA_PATH,
                    help='the dataset, if a dir, assumed to be val, if a .csv, assumed to be test')

parser.add_argument('--all', action='store_true')
parser.add_argument('--batch_size', '-bs', default=384, type=int)
parser.add_argument('--canonical', '-c', action='store_true',
                    help='output CSV in canonical GLUCOSE format for evaluation_baseline.py')
parser.add_argument('--seed', type=int, default=2557)

CANONICAL_COLS = [
    'story_id', 'unique_id', 'story', 'selected_sentence',
    '1_specificNL', '1_generalNL', '2_specificNL', '2_generalNL', '3_specificNL', '3_generalNL',
    '4_specificNL', '4_generalNL', '5_specificNL', '5_generalNL', '6_specificNL', '6_generalNL',
    '7_specificNL', '7_generalNL', '8_specificNL', '8_generalNL', '9_specificNL', '9_generalNL',
    '10_specificNL', '10_generalNL']

kwargs = dict(
    top_k=15,
    do_sample=True,
    max_length=256)

def generate_from_sentence(model, tokenizer, input):
    inputs = tokenizer.encode(input, return_tensors='pt')
    output_sequences = model.generate(
        inputs.to(model.device),
        pad_token_id=tokenizer.eos_token_id,
        **kwargs
    )

    return [tokenizer.decode(x, skip_special_tokens=True) for x in output_sequences]


def generate_from_dataset(model, tokenizer, dataset, batch_size=128, skip=True):
    output_sequences_all = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        # for i in tqdm(range(0, 1000, batch_size)):
        batch = dataset[i:i+batch_size]
        output_sequences = model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        output_sequences_all.extend(output_sequences)
    return output_sequences_all


def decode_seqs(seqs, tokenizer, skip):
    return [tokenizer.decode(x, skip_special_tokens=skip) for x in seqs]


def run_inference(dataset, model, tokenizer, exp_num, batch_size=128, seed=0):
    set_seed(seed)
    preds = generate_from_dataset(model, tokenizer, dataset, batch_size=batch_size)
    preds_decoded = decode_seqs(preds, tokenizer, True)

    # dataset_orig = dataset
    # dataset = dataset[0:len(preds_decoded)]

    if exp_num == '2b':
        sources_decoded = decode_seqs(dataset['input_ids'], tokenizer, False)
        sources_decoded = [x.split('</s>', 1)[0] for x in sources_decoded]
    else:
        sources_decoded = decode_seqs(dataset['input_ids'], tokenizer, True)
    if 'labels' in dataset:
        labels_decoded = decode_seqs(dataset['labels'], tokenizer, True)
    else:
        labels_decoded = [None] * len(preds_decoded)

    d = {'input': sources_decoded, 'output_true': labels_decoded,
                          'output_pred': preds_decoded}
    d['story_id'] = dataset['story_id']
    d['unique_id'] = dataset['experiment_id']
    output = pd.DataFrame(d)

    return output


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
        if row.story != story:
            if story:
                for k, v in canon_row.items():
                    if not v:
                        canon_row[k] = 'escaped'
                canon_rows.append(canon_row)
                # assert row.story_id not in seen
                seen.add(row.story_id)

            canon_row = canonical_dict()

        # now we add to each row
        story = row.story
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

if __name__ == "__main__":
    args = parser.parse_args()
    model_size = 't5-large' if 'large' in args.model_dir else 't5-base'
    suffix = '_' + args.dataset_dir.rsplit('_', 1)[-1]
    if suffix == '_test' and not args.canonical:
        print(f'WARNING: you are running inference on test; if you want to use the baseline evaluation, '
              ' include the flag --canonical')

    print(f'model size is {model_size}')

    model_path = Path(args.model_dir)
    ds_val = datasets.load_from_disk(args.dataset_dir)

    cols = ['input_ids', 'attention_mask']
    if 'val' in args.dataset_dir:
        cols.append('labels')

    ds_val.set_format(type='torch', columns=cols, device='cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_size)

    if args.all:
        ckpts = [x for x in model_path.glob('*/') if x.is_dir()]
    else:
        ckpts = model_path / "model"

    for folder_name in ckpts:
        if not (folder_name / 'config.json').exists():
            continue
        print(f'loading model from {folder_name}...')
        model_ft = T5ForConditionalGeneration.from_pretrained(folder_name)
        model_ft = model_ft.cuda()
        print(f'running inference...')
        pred = run_inference(ds_val, model_ft, tokenizer, args.exp_num, args.batch_size, args.seed)
        out_path = folder_name / f"predictions{suffix}.csv"
        if args.canonical:
            pred = to_canonical(pred)
        pred.to_csv(out_path, index=False)
        print(f'saved to {out_path}')
        torch.cuda.empty_cache()
