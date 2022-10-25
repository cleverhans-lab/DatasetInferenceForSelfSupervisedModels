# Dataset Inference for Self-Supervised Models


Keywords: self-supervised models, model stealing, defenses.

TL;DR: We introduce a new dataset inference defense, which uses the private training set of the victim encoder model to attribute its ownership in the event of stealing.

## Abstract 

Self-supervised models are increasingly prevalent in machine learning (ML) since they reduce the need for expensively labeled data. Because of their versatility in downstream applications, they are increasingly used as a service exposed via public APIs. At the same time, these encoder models are particularly vulnerable to model stealing attacks due to the high dimensionality of vector representations they output. Yet, encoders remain undefended: existing mitigation strategies for stealing attacks focus on supervised learning. We introduce a new dataset inference defense, which uses the private training set of the victim encoder model to attribute its ownership in the event of stealing. The intuition is that the log-likelihood of an encoder's output representations is higher on the victim's training data than on test data if it is stolen from the victim, but not if it is independently trained. We compute this log-likelihood using density estimation models. As part of our evaluation, we also propose to measure the fidelity of stolen encoders without involving downstream tasks and to quantify the effectiveness of the theft detection; instead, we leverage mutual information and distance measurements. Our extensive empirical results in the vision domain demonstrate that dataset inference is a promising direction for defending self-supervised models against model stealing.


## Description of the code

The file `run.py` is used to train victim/independent encoders using the SimCLR training approach. The file `steal.py` is used to steal encoders. `linear_eval.py` is used to run linear evaluation on the victim/stolen/independent models. For the ImageNet encoder, `stealsimsiam.py` is used for stealing and `linsimsiam.py` is used for running linear evaluation on the stolen encoders.

The code for our dataset inference approach can be found at `dataset-inference/gmm.py`. The code for calculating the similarity scores is at `similarity_metrics/dist.py` (cosine/l2 scores) and `similarity_metrics/mutual_information.py` (mutual information score).   

Example scripts to run these files can be found in the folder `scripts`. 

Note: Parts of the code are based off the following Github repositories: https://github.com/sthalles/SimCLR, https://github.com/facebookresearch/simsiam