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
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorWithPadding
import wandb

from local_vars import GLUCOSE_DIR, SAVE_DIR, EXP_NUMS, SEED

parser = argparse.ArgumentParser()
parser.add_argument('exp_num', choices=EXP_NUMS)
parser.add_argument('--output_dir')
parser.add_argument('--dataset_dir')
parser.add_argument('--seed', type=int, default=SEED)
parser.add_argument('--model_size', '-m', default='t5-base')
parser.add_argument('--no_shuffle', dest='shuffle', action='store_false')
parser.add_argument('--eval_bleu', action='store_true') # extremely slow!
parser.add_argument('--batch_size_train', '-bst', type=int, default=0)
parser.add_argument('--batch_size_eval', '-bse', type=int, default=0)

parser.add_argument('--no_logging', dest='logging', action='store_false')

metric = datasets.load_metric('bleu', keep_in_memory=True)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    d = metric.compute(predictions=predictions, references=labels)
    output_dict = {'bleu': d['bleu']}
    return output_dict

def main(args, exp_name, tokenizer, ds_train, ds_val, batch_size_train, batch_size_eval, use_fp16):
    transformers.trainer_utils.set_seed(args.seed)

    print('before model')
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_size, cache_dir='/nlp/data/bryanli/.cache')
    if args.exp_num == '2b':
        model.resize_token_embeddings(len(tokenizer))

    if args.shuffle:
        ds_train = ds_train.shuffle(seed=args.seed)
        ds_val = ds_val.shuffle(seed=args.seed)

    # taken from GLUCOSE paper and T5 paper when possible
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        # prediction_loss_only=True, # If I need co compute only loss and not other metrics, setting this to true will use less RAM
        evaluation_strategy='steps',  # Run evaluation every eval_steps
        save_steps=1000,  # How often to save a checkpoint
        logging_steps=1000,  # How often to log loss to wandb
        save_total_limit=5,  # Number of maximum checkpoints to save
        remove_unused_columns=True,  # Removes useless columns from the dataset
        run_name=exp_name,  # Wandb run name
        load_best_model_at_end=True,  # Whether to load the best model found at each evaluation.
        metric_for_best_model="eval_loss",  # Use loss to evaluate best model.
        greater_is_better=False,  # Best model is the one with the lowest loss, not highest.
        seed=args.seed,
        eval_accumulation_steps=20,
        fp16=use_fp16
    )

    # note that we change the lr to 1e-4, since the original 1e-3 converges too fast

    optimizer = transformers.Adafactor(model.parameters(), lr=0.0001,
                                       relative_step=False, warmup_init=False, scale_parameter=False,
                                       decay_rate=0.0, clip_threshold=1.0)
    scheduler = None


    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None

    print("padding to multiple of 8 for fp16" if data_collator else "not fp16")

    print('before trainer')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.args._n_gpu = 2
    trainer.train()
    trainer.save_model(f'{args.output_dir}/model')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.logging:
        logging.basicConfig(level=logging.DEBUG)

    # load tokenizer and datasets from disk
    tokenizer = AutoTokenizer.from_pretrained(args.model_size)

    exp_name = f'exp{args.exp_num}_{args.model_size}'
    if not args.output_dir:
        args.output_dir = f'{GLUCOSE_DIR}/outputs/{exp_name}'
    if not args.dataset_dir:
        args.dataset_dir = f'{SAVE_DIR}/{exp_name}'

    logging.debug(f'loading datasets from from {args.dataset_dir}...')
    ds_train = datasets.load_from_disk(f'{args.dataset_dir}/ds_train')
    ds_val = datasets.load_from_disk(f'{args.dataset_dir}/ds_val')

    if args.model_size == 't5-large':
        batch_size_train = args.batch_size_train or 4
        batch_size_eval = args.batch_size_eval or (12 if args.eval_bleu else 2)
        use_fp16 = False
    elif args.model_size == 't5-base':
        batch_size_train = args.batch_size_train or 30
        batch_size_eval = args.batch_size_eval or (30 if not args.eval_bleu else 10)
        use_fp16 = True
    else:
        print('invalid model size!')
        os.exit()
    if not args.eval_bleu:
        compute_metrics = None
    print(batch_size_eval, batch_size_train)
    print(args)

    # wandb.login()
    wandb.init(project="glucose_hf", name=exp_name)
    print('before main')
    main(args, exp_name, tokenizer, ds_train, ds_val, batch_size_train, batch_size_eval, use_fp16=use_fp16)
