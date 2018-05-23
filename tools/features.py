#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys
import torchlanguage.transforms
import os
import torch
from torchlanguage import transforms as ltransforms
from tools import settings


# Get tweet transformer
def tweet_transformer(lang, n_gram, voc=None):
    """
    Get tweet transformer
    :param lang:
    :param n_gram:
    :return:
    """
    if voc is None:
        token_to_ix = dict()
    else:
        token_to_ix = voc
    # end if
    if n_gram == 'c1':
        return ltransforms.Compose([
            ltransforms.RemoveRegex(
                regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            ltransforms.ToLower(),
            ltransforms.Character(),
            ltransforms.ToIndex(start_ix=1, token_to_ix=token_to_ix),
            ltransforms.ToLength(length=settings.min_length),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram][lang] - 1)
        ])
    else:
        return ltransforms.Compose([
            ltransforms.RemoveRegex(
                regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            ltransforms.ToLower(),
            ltransforms.Character2Gram(),
            ltransforms.ToIndex(start_ix=1, token_to_ix=token_to_ix),
            ltransforms.ToLength(length=settings.min_length),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram][lang] - 1)
        ])
    # end if
# end tweet_transformer
