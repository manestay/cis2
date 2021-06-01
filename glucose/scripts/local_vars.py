import os

GLUCOSE_DIR = '.'
SAVE_DIR = './data'

SEED = 2557
TRAIN_PATH = os.path.join(GLUCOSE_DIR, 'data/GLUCOSE_training_data_final.csv')
TEST_PATH = os.path.join(GLUCOSE_DIR, 'data/nov27_key_final_copy.csv')

EXP_NUMS = ['0', '1', '2a', '2b', '3a', '3b']

COLS_TO_FORMAT = ['input_ids', 'labels', 'attention_mask']

BATCH_SIZE_ENCODE = 512
