import argparse
from pathlib import Path
from os.path import isdir

import datasets
import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer

MODEL_DIR = "/home1/b/bryanli/projects/stories/glucose/outputs/exp0"
DATA_PATH = f"{MODEL_DIR}/ds_val"
parser = argparse.ArgumentParser()
parser.add_argument('exp_num', type=str)
parser.add_argument('--model_dir', '-m', default=MODEL_DIR)
parser.add_argument('--dataset_dir', '-d', default=DATA_PATH,
    help='the dataset, if a dir, assumed to be val, if a .csv, assumed to be test')

parser.add_argument('--all', action='store_true')
parser.add_argument('--batch_size', '-bs', default=384, type=int)

kwargs = dict(
    top_k=15,
    do_sample=True,
    max_length=256)


def generate_from_sentence(model, tokenizer, input):
    inputs = tokenizer.encode(input, return_tensors='pt')
    output_sequences = model.generate(
        inputs.to(model.device),
        attention_mask=batch['attention_mask'],
        pad_token_id=tokenizer.eos_token_id,
        **kwargs
    )

    return [tokenizer.decode(x, skip_special_tokens=True) for x in output_sequences]


def generate_from_dataset(model, tokenizer, dataset, batch_size=128, skip=True):
    output_sequences_all = []
    # for i in tqdm(range(0, len(dataset), batch_size)):
    for i in tqdm(range(0, 1000, batch_size)):
        batch = dataset[i:i+batch_size]
        output_sequences = model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        output_sequences_all.extend(output_sequences)
    print(len(output_sequences_all))
    return output_sequences_all


def decode_seqs(seqs, tokenizer, skip):
    return [tokenizer.decode(x, skip_special_tokens=skip) for x in seqs]


def run_inference(dataset, model, tokenizer, exp_num, batch_size=128):
    preds = generate_from_dataset(model, tokenizer, dataset, batch_size=batch_size)
    preds_decoded = decode_seqs(preds, tokenizer, True)

    dataset_orig = dataset
    dataset = dataset[0:len(preds_decoded)]

    if exp_num == '2b':
        sources_decoded = decode_seqs(dataset['input_ids'], tokenizer, False)
        sources_decoded = [x.split('</s>', 1)[0] for x in sources_decoded]
    else:
        sources_decoded = decode_seqs(dataset['input_ids'], tokenizer, True)
    if 'labels' in dataset:
        labels_decoded = decode_seqs(dataset['labels'], tokenizer, True)
    else:
        labels_decoded = [None] * len(preds_decoded)

    output = pd.DataFrame({'input': sources_decoded, 'output_true': labels_decoded,
                          'output_pred': preds_decoded})

    return output


if __name__ == "__main__":
    args = parser.parse_args()
    model_size = 't5-large' if 'large' in args.model_dir else 't5-base'
    suffix = '_' + args.dataset_dir.rsplit('_', 1)[-1]
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
        pred = run_inference(ds_val, model_ft, tokenizer, args.exp_num, args.batch_size)
        out_path = folder_name.parent / f"predictions{suffix}.csv"
        pred.to_csv(out_path)
        print(f'saved to {out_path}')
        torch.cuda.empty_cache()