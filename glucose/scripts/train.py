"""
Train the various GLUCOSE experiments using the same hyperparameters. The default batch sizes are for
exp0 (the original task); they can be increased since the other experiments have shorter sequences.
"""

import argparse
import os
import logging

import datasets
import numpy as np
import transformers
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorWithPadding
import wandb

from local_vars import GLUCOSE_DIR, SAVE_DIR, EXP_NUMS, SEED, METRICS
from utils import load_tokenizer, get_exp_name

parser = argparse.ArgumentParser()
parser.add_argument('exp_num', choices=EXP_NUMS)
parser.add_argument('--output_dir')
parser.add_argument('--dataset_dir')
parser.add_argument('--seed', type=int, default=SEED)
parser.add_argument('--model_size', '-m', default='t5-base')
parser.add_argument('--sim_metric', '-sm', default='bleu', choices=METRICS)
parser.add_argument('--no_shuffle', dest='shuffle', action='store_false')
parser.add_argument('--eval_bleu', action='store_true')
parser.add_argument('--eval_em', action='store_true')
parser.add_argument('--batch_size_train', '-bst', type=int, default=0)
parser.add_argument('--batch_size_eval', '-bse', type=int, default=0)
parser.add_argument('--specific-only', '-so', action='store_true')

parser.add_argument('--no_logging', dest='logging', action='store_false')

metric = datasets.load_metric('sacrebleu', keep_in_memory=True)


def compute_sacrebleu(eval_pred):
    # This BLEU evaluation is not the same as the one for evaluation, since we do not consider
    # specific and general statements separately. We do 1 BLEU calculation for speed.
    logits, labels = eval_pred
    pred_ids = np.argmax(logits[0], axis=-1)
    preds = [tokenizer.decode(x, skip_special_tokens=True) for x in pred_ids]
    refs = [[tokenizer.decode(x, skip_special_tokens=True)] for x in labels]

    d = metric.compute(predictions=preds, references=refs)
    output_dict = {'bleu': d['score']}
    logging.info(output_dict)
    return output_dict

def compute_exact_match(eval_pred):
    logits, labels = eval_pred
    pred_ids = np.argmax(logits[0], axis=-1)
    preds = [tokenizer.decode(x, skip_special_tokens=True) for x in pred_ids]
    refs = [tokenizer.decode(x, skip_special_tokens=True) for x in labels]
    score = sum([a == b for a, b in zip(preds, refs)]) / len(preds)
    output_dict = {'em': score}
    logging.info(output_dict)
    return output_dict

def main(args, exp_name, tokenizer, ds_train, ds_val, batch_size_train, batch_size_eval, use_fp16,
         eval_bleu=False, eval_em=False):
    transformers.trainer_utils.set_seed(args.seed)
    if eval_bleu:
        eval_metric = 'eval_bleu'
        compute_metrics = compute_sacrebleu
        greater_is_better = True
    elif eval_em:
        eval_metric = 'eval_em'
        compute_metrics = compute_exact_match
        greater_is_better = True
    else:
        eval_metric = "eval_loss"
        compute_metrics = None
        greater_is_better = False

    model = T5ForConditionalGeneration.from_pretrained(
        args.model_size, cache_dir='/nlp/data/bryanli/.cache')

    if args.exp_num == '2b' or args.exp_num == 'A':
        model.resize_token_embeddings(len(tokenizer))

    if args.shuffle:
        ds_train = ds_train.shuffle(seed=args.seed)
        ds_val = ds_val.shuffle(seed=args.seed)

    # taken from GLUCOSE paper and T5 paper when possible
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        # prediction_loss_only=True,
        evaluation_strategy='steps',
        save_steps=2000,
        logging_steps=2000,
        save_total_limit=4,
        remove_unused_columns=True,
        run_name=exp_name,
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
        greater_is_better=greater_is_better,
        seed=args.seed,
        eval_accumulation_steps=20,
        fp16=use_fp16
    )

    # note that we change the lr to 1e-4, since the original 1e-3 converges too fast

    # optimizer = transformers.Adafactor(model.parameters(), lr=0.001,
    optimizer = transformers.Adafactor(model.parameters(), lr=0.0001,
                                       relative_step=False, warmup_init=False, scale_parameter=False,
                                       decay_rate=0.0, clip_threshold=1.0)

    # optimizer = transformers.Adafactor(model.parameters(), lr=0.001, relative_step=False, warmup_init=False,
    #                                    scale_parameter=True, decay_rate=0.0, clip_threshold=1.0)
    scheduler = None

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None

    print("padding to multiple of 8 for fp16" if data_collator else "not fp16")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.args._n_gpu = 2
    trainer.train()
    trainer.save_model(f'{args.output_dir}/model')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.logging:
        logging.basicConfig(level=logging.DEBUG)

    # load tokenizer and datasets from disk
    tokenizer = load_tokenizer(args.model_size, args.exp_num)

    exp_name = get_exp_name(args.exp_num, args.model_size, args.sim_metric, args.specific_only)
    if not args.output_dir:
        args.output_dir = f'{GLUCOSE_DIR}/outputs/{exp_name}'
    if not args.dataset_dir:
        args.dataset_dir = f'{SAVE_DIR}/{exp_name}'

    logging.debug(f'loading datasets from from {args.dataset_dir}...')
    ds_train = datasets.load_from_disk(f'{args.dataset_dir}/ds_train')
    # if args.eval_bleu: # evaluating BLEU takes a long time, use small set
    ds_val = datasets.load_from_disk(f'{args.dataset_dir}/ds_val_small')
    # else:
    #     ds_val = datasets.load_from_disk(f'{args.dataset_dir}/ds_val')
    logging.debug(f'example from train:')
    logging.debug(f'{tokenizer.decode(ds_train[200]["input_ids"])}')
    logging.debug(f'{tokenizer.decode(ds_train[200]["labels"])}')
    if 'output_orig' in ds_train.features:
        logging.debug(ds_train['output_orig'][200])
    if args.model_size == 't5-large':
        batch_size_train = args.batch_size_train or 8
        batch_size_eval = args.batch_size_eval or 12
        use_fp16 = False
    elif args.model_size == 't5-base':
        batch_size_train = args.batch_size_train or 30
        batch_size_eval = args.batch_size_eval or 30
        use_fp16 = True
    else:
        print('invalid model size!')
        os.exit()
    print(args)

    # wandb.login()
    wandb.init(project="glucose_hf", name=exp_name, id=wandb.util.generate_id())
    main(args, exp_name, tokenizer, ds_train, ds_val, batch_size_train, batch_size_eval,
         use_fp16=use_fp16, eval_bleu=args.eval_bleu, eval_em=args.eval_em)
