import os

GLUCOSE_DIR = '.'
SAVE_DIR = './data'

SEED = 2557
TRAIN_PATH = os.path.join(GLUCOSE_DIR, 'data/GLUCOSE_training_data_final.csv')
TEST_PATH = os.path.join(GLUCOSE_DIR, 'data/nov27_key_final_copy.csv')
ALL_RESULTS_PATH = os.path.join(GLUCOSE_DIR, 'outputs/all_results.tsv')

EXP_NUMS = ['0', '1', '2a', '2b', '3a', 'A']

COLS_TO_FORMAT = ['input_ids', 'labels', 'attention_mask']

BATCH_SIZE_ENCODE = 512

CANONICAL_COLS = [
    'story_id', 'unique_id', 'story', 'selected_sentence',
    '1_specificNL', '1_generalNL', '2_specificNL', '2_generalNL', '3_specificNL', '3_generalNL',
    '4_specificNL', '4_generalNL', '5_specificNL', '5_generalNL', '6_specificNL', '6_generalNL',
    '7_specificNL', '7_generalNL', '8_specificNL', '8_generalNL', '9_specificNL', '9_generalNL',
    '10_specificNL', '10_generalNL']

RESULTS_COLS = ['model', 'split', 'is_baseline', 'specific_avg', 'general_avg'] + \
    [f'specific_dim{i}' for i in range(1, 11)] + [f'general_dim{i}' for i in range(1, 11)]
RESULTS_COLS += ['specific_avg1-5', 'specific_avg6-10', 'general_avg1-5', 'general_avg6-10']
