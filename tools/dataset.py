#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import datasets
import torchlanguage.transforms
import torch


#########################################
# Tokenizer and features
#########################################


# Load authorship attribution dataset
def load_AA_dataset(authors=None):
    """
    Load dataset
    :return:
    """
    # Load from directory
    reutersc50_dataset = torchlanguage.datasets.ReutersC50Dataset(
        root='./aa_data',
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


# Load author profiling dataset
def load_AP_dataset(text_transform, batch_size):
    """
    Load author profiling dataset
    :return:
    """
    # Tweet data set 2017 training
    tweet_dataset_17 = datasets.TweetDataset(root='./data/', download=True, lang='en',
                                            text_transform=text_transform)

    # Loader
    pan17loader_training = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(tweet_dataset_17, train=True),
        batch_size=batch_size,
        shuffle=True
    )

    # Loader
    pan17loader_validation = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(tweet_dataset_17, train=False),
        batch_size=batch_size,
        shuffle=True
    )

    return pan17loader_training, pan17loader_validation
# end load_AP_dataset
