import pandas as pd

for split in ['test', 'val']:
    # TXT file to reformat
    TXT_NAME = f'outputs/baseline/model/pred_{split}.txt'
    OUT_NAME = f'outputs/baseline/model/predictions_{split}.csv'

    # get template CSV for the other columns
    CSV_NAME = f'outputs/exp0_t5-base/model/predictions_{split}.csv'

    with open(TXT_NAME, 'r') as f:
        lines = [x.strip() for x in f]

    df = pd.read_csv(CSV_NAME)
    # df.rename({'output_pred': 'output_pred_other'}, axis=1, inplace=True)
    df.drop('output_pred', axis=1, inplace=True) # remove the old predictions
    df['output_pred'] = lines
    df.to_csv(OUT_NAME, index=False)
