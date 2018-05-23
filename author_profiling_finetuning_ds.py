#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : single_model_tweet.py
# Description : Train CNN on Tweet.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Neuchâtel, Suisse
#
# This file is part of the PAN18 author profiling challenge code.
# The PAN18 author profiling challenge code is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

# Imports
import torch
from torchlanguage import models
import torch.nn as nn
from torch import optim
import copy
import numpy as np
from tools import settings, functions, dataset
import author_profiling_test_model

# Parse argument
args = functions.argument_parser_training_model()

# Transformer
transformer = functions.AP_transformer(args.n_gram)

# Load data sets
pan17loader_training, pan17loader_validation = dataset.load_AP_dataset(transformer, args.batch_size)

# Loss function
loss_function = nn.CrossEntropyLoss()

# For each size
for size in np.arange(0.1, 1.1, 1.0):
    # 10-CV
    for k in np.arange(0, 10):
        # Log
        print(u"Starting fold {}".format(k))

        # Set training size
        pan17loader_training.dataset.set_size(size)

        # Set fold
        pan17loader_training.dataset.set_fold(k)
        pan17loader_validation.dataset.set_fold(k)

        # Model
        model = models.CNNCTweet(text_length=settings.min_length, vocab_size=settings.voc_sizes[args.n_gram]['en'],
                                 embedding_dim=args.dim, n_classes=2)
        model.load_state_dict(torch.load(open(args.model, 'rb')))

        # Replace last linear layer
        model.linear = nn.Linear(model.linear.in_features, 6)

        # To GPU
        if args.cuda:
            model.cuda()
        # end if

        # Best model
        best_model = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Test model
        author_profiling_test_model.test_model(
            model,
            args.epoch,
            pan17loader_training,
            pan17loader_validation,
            loss_function,
            optimizer,
            transformer,
            args.output, args.cuda
        )
    # end for
# end for
