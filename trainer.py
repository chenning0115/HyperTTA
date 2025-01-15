import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import transformer as transformer
from models import SSRN
import utils
from utils import recorder
from evaluation import HSIEvaluation
from copy import deepcopy
from utils import device

from BNTentTrainer import SSRNTrainer, CNNTrainer
from LNTentTrainer import TransformerTrainer, SSFTTNEWTrainer, SQSTrainer, TransformerOriTrainer
from BNLNtentTrainer import TransformerTrainer as TransformerTrainerBNLN


def get_trainer(params):
    trainer_type = params['net']['trainer']
    if trainer_type == "transformer_origin":
        return TransformerOriTrainer(params)
    if trainer_type == "transformer":
        return TransformerTrainer(params)
    if trainer_type == "SSRN":
        return SSRNTrainer(params)
    if trainer_type == "CNN":
        return CNNTrainer(params)
    if trainer_type == "ssftt":
        return SSFTTNEWTrainer(params)
    if trainer_type == "sqs":
        return SQSTrainer(params)

    if trainer_type == "transformer_bnln":
        return TransformerTrainerBNLN(params)

    assert Exception("Trainer not implemented!")

