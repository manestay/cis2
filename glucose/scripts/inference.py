'''
Runs inference on the trained GLUCOSE models. It will evaluate all checkpoints within the folder
specified by --model_dir.

If you wish to use `evaluation_baseline.py`, pass in the --canonical option. canonical means the
`canonical` CSV format, as given by the original GLUCOSE paper.
'''

import argparse
from pathlib import Path
import os

import datasets
import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.trainer_utils import set_seed

from local_vars import GLUCOSE_DIR, SAVE_DIR
from utils import to_canonical

parser = argparse.ArgumentParser()
parser.add_argument('exp_num', type=str)
parser.add_argument('--model_dir', '-md')
parser.add_argument('--model_size', '-ms')
parser.add_argument('--dataset_dir', '-d', help='the dataset dir, saved by datasets.save_to_disk')

parser.add_argument('--batch_size', '-bs', default=384, type=int)
parser.add_argument('--canonical', '-c', action='store_true',
                    help='also output a CSV in canonical GLUCOSE format for evaluation_baseline.py')
parser.add_argument('--seed', type=int, default=2557)
parser.add_argument('--all_checkpoints', action='store_true', help='evaluate all checkpoints')

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
    if 'labels' in dataset.features:
        labels_decoded = decode_seqs(dataset['labels'], tokenizer, True)
    else:
        labels_decoded = [None] * len(preds_decoded)

    d = {'input': sources_decoded, 'output_true': labels_decoded,
         'output_pred': preds_decoded}

    d['unique_id'] = dataset['experiment_id']
    output = pd.DataFrame(d)
    return output



if __name__ == "__main__":
    args = parser.parse_args()
    exp_name = f'exp{args.exp_num}_{args.model_size}'
    if not args.model_dir:
        args.model_dir = os.path.join(GLUCOSE_DIR, 'outputs', exp_name)
    if not args.dataset_dir:
        args.dataset_dir = os.path.join(SAVE_DIR, exp_name, 'ds_val')

    suffix = '_' + args.dataset_dir.rsplit('_', 1)[-1].strip('/')
    print(f'model size is {args.model_size}')


    model_path = Path(args.model_dir)
    ds_val = datasets.load_from_disk(args.dataset_dir)

    cols = ['input_ids', 'attention_mask']
    if 'val' in args.dataset_dir:
        cols.append('labels')

    ds_val.set_format(type='torch', columns=cols, device='cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.model_size)

    if args.all_checkpoints:
        ckpts = [x for x in model_path.glob('*/') if x.is_dir()]
    else:
        ckpts = [model_path / "model"]

    print(args)

    for folder_name in ckpts:
        if not (folder_name / 'config.json').exists():
            continue
        print(f'loading model from {folder_name}...')
        model_ft = T5ForConditionalGeneration.from_pretrained(folder_name)
        model_ft = model_ft.cuda()
        print(f'running inference...')
        pred = run_inference(ds_val, model_ft, tokenizer, args.exp_num, args.batch_size, args.seed)
        out_path = folder_name / f"predictions{suffix}.csv"
        pred.to_csv(out_path, index=False)
        print(f'saved to {out_path}')
        if args.canonical:
            out_path_canon = folder_name / f"predictions_canonical{suffix}.csv"
            pred = to_canonical(pred)
            pred.to_csv(out_path_canon, index=False)
            print(f'saved canonical to {out_path_canon}')
        torch.cuda.empty_cache()
