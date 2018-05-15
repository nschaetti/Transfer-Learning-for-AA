#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.transforms
import torch


#########################################
# Tokenizer and features
#########################################


# Load dataset
def load_dataset(authors=None):
    """
    Load dataset
    :return:
    """
    # Load from directory
    reutersc50_dataset = torchlanguage.datasets.ReutersC50Dataset(
        authors=authors,
        download=True
    )

    # Reuters C50 dataset training
    reuters_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reutersc50_dataset),
        batch_size=1,
        shuffle=True
    )

    # Reuters C50 dataset test
    reuters_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reutersc50_dataset, train=False),
        batch_size=1,
        shuffle=True
    )
    return reutersc50_dataset, reuters_loader_train, reuters_loader_test
# end load_dataset
