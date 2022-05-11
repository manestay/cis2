# Careful with Context

This repository contains the code used to run the experiments for the paper ``Careful with Context: A Critique of Commonsense Inference Methods.''

You will need to download the datasets here: [test](https://drive.google.com/file/d/134C7w3fNvzsUbLvjnhdatYraTwcfeqDw/view?usp=sharing) and [train](https://drive.google.com/file/d/119C50en6LvOBjhyFMBdEX2QRcbKbwiQg/view?usp=sharing).
Unzip these to `glucose/data_final`.

The main run script is `glucose/scripts/run_pipeline.sh`. Refer to that for more instructions on running our experiments.

TODO: reorganize the directories -- probably move the notebooks to glucose/notebooks
TODO: rewrite the README to make sure others can run it

# Citation
```
    @inproceedings{Li2022,
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
