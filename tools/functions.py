# -*- coding: utf-8 -*-
#

# Imports
from torchlanguage import transforms as ltransforms
from torchlanguage import models
from torchvision import transforms
from torchvision import models as vmodels
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import argparse
import dataset
import torch
import settings
import os
import codecs

#################
# Arguments
#################


# Tweet argument parser for training model
def argument_parser_training_model():
    """
    Tweet argument parser
    :return:
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="PAN18 Author Profiling challenge")

    # Argument
    parser.add_argument("--output", type=str, help="Model output file", default='.')
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=300)
    parser.add_argument("--n-gram", type=str, help="N-Gram (c1, c2)", default='c1')
    parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
    parser.add_argument("--epoch", type=int, help="Epoch", default=300)
    parser.add_argument("--model", type=str, help="resnet18, alexnet", default='resnet18')
    parser.add_argument("--batch-size", type=int, help="Batch size", default=20)
    parser.add_argument("--val-batch-size", type=int, help="Val. batch size", default=5)
    parser.add_argument("--training-count", type=int, help="Number of samples to train", default=-1)
    parser.add_argument("--test-count", type=int, help="Number of samples to test", default=-1)
    parser.add_argument("--model", type=str, help="Model input file", default='')
    args = parser.parse_args()

    # Use CUDA?
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
# end argument_parser_training_model


#################
# Transformers
#################


# Get AP transformer
def AP_transformer(n_gram, voc=None):
    """
    Get tweet transformer
    :param n_gram:
    :return:
    """
    if voc is None:
        token_to_ix = dict()
    else:
        token_to_ix = voc
    # end if
    if n_gram == 'c1':
        return transforms.Compose([
            ltransforms.RemoveRegex(
                regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            ltransforms.ToLower(),
            ltransforms.Character(),
            ltransforms.ToIndex(start_ix=1, token_to_ix=token_to_ix),
            ltransforms.ToLength(length=settings.min_length),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram]['en'] - 1)
        ])
    else:
        return transforms.Compose([
            ltransforms.RemoveRegex(
                regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            ltransforms.ToLower(),
            ltransforms.Character2Gram(),
            ltransforms.ToIndex(start_ix=1, token_to_ix=token_to_ix),
            ltransforms.ToLength(length=settings.min_length),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram]['en'] - 1)
        ])
    # end if
# end tweet_transformer
