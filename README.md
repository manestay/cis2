# $\text{CIS}^2$

This repository contains the code used to run the experiments for the paper " $\text{CIS}^2$ : A Simplified Commonsense Inference Evaluation for Story Prose".


## Downloading data
Download the datasets, and unzip them into this folder:
You will need to download the GLUCOSE datasets here: [test](https://drive.google.com/file/d/134C7w3fNvzsUbLvjnhdatYraTwcfeqDw/view?usp=sharing) and [train](https://drive.google.com/file/d/119C50en6LvOBjhyFMBdEX2QRcbKbwiQg/view?usp=sharing).
Unzip these to `glucose/data_final`.

These datasets were provided by the GLUCOSE authors, and are in the original collection format, and not in the preprocessed format of the [original repository](https://github.com/ElementalCognition/glucose).

# Running CIS^2
The main run script is `glucose/scripts/run_pipeline.sh`. Assuming your paths are correct, this script should run all steps directly. However, it is always good practice to initially run commands one at a time. Here we briefly describe the function of the main script, and each of the Python scripts called.

`run_pipeline.sh` will run a full preprocess, training, and evaluation pipeline for each experiment. Key for EXP_NUM values: `0 -> Original`, `1 -> next sentence generation`, `2a -> History`, `2b -> Mask X`, `3a -> History+X`.

* `preprocess.py`: takes in GLUCOSE formatted train and test datasets, and reformats it to datasets for the experiments described in the CIS^2 paper. Splits off part of train for validation if `--val_ids` is passed in.
* `train.py`: runs the training loop to fine-tune a pretrained T5 model.
* `inference.py`: runs inference. By default, it runs inference on validation. Or you can pass in a path to a dataset folder, i.e., for test.
* `evaluation_val.py`: evaluate BLEU scores between a reference set and model predictions when there is only 1 reference.
* `evaluation_test.py`: evaluate BLEU scores when there are 3 references, i.e. for the GLUCOSE test set.
* `evaluation_cis2.py`: evaluate CIS^2 scores.

# Pretrained model
You can download the best model checkpoint for the CIS^2 model [here](https://drive.google.com/file/d/1rxw2r-DzTW_NcAlUGP874plZWQ4yYDwT/view?usp=sharing).

# Citation
```
    @inproceedings{Li2022cis2,
    title={{$CIS^2$: A Simplified Commonsense Inference Evaluation for Story Prose}},
    author={Li, Bryan and Martin, Lara J. and Callison-Burch, Chris},
    archivePrefix = {arXiv},
    eprint = {2202.07880},
    journal={{Workshop on Commonsense Representation and Reasoning (CSRR) at ACL 2022}},
    year={2022},
    url={https://openreview.net/forum?id=Se-xHMYg_bc}
    }
```

# Acknowledgments
We thank Or Biran and Lori Moon of Elemental Cognition for their assistance with working with the GLUCOSE dataset and codebase.

# Contact
Send us an [email](mailto:bryanli@seas.upenn.edu) if you have any questions.
