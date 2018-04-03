#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
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
import argparse
import torch.utils.data
from echotorch import datasets
from echotorch.transforms import text
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from authors import authors as aa
import os

# Settings
n_epoch = 600
embedding_dim = 300
n_authors = 15
use_cuda = False

# Argument parser
parser = argparse.ArgumentParser(description="CNN transfer learning")

# Argument
parser.add_argument("--input", type=str, help="Pre-trained classifier file", default='.')
parser.add_argument("--character-gram", type=int, help="Character gram", default=2)
parser.add_argument("--n-authors", type=int, help="Number of authors to pre-train", default=20)
parser.add_argument("--batch-size", type=int, help="Batch size", default=64)
parser.add_argument("--epoch", type=int, help="Epoch", default=500)
parser.add_argument("--pretraining", type=str, help="Pre-training type (finetuning, extractor)", default="finetuning")
args = parser.parse_args()

# Authors
authors = aa[:args.n_authors]

# Load pre-trained model
model, gram_to_ix = torch.load(open(args.input, 'rb'))

# If feature extract, then block params
if args.pretraining == "extractor":
    for param in model.parameters():
        param.requires_grad = False
    # end for
# end if

# Word embedding
if args.character_gram == 1:
    transform = text.Character(fixed_length=7978, gram_to_ix=gram_to_ix)
    window_size = 7978
    voc_size = 60
elif args.character_gram == 2:
    transform = text.Character2Gram(fixed_length=7978, gram_to_ix=gram_to_ix)
    window_size = 7978
    voc_size = 1653
elif args.character_gram == 3:
    transform = text.Character3Gram(fixed_length=7977, gram_to_ix=gram_to_ix)
    window_size = 7977
    voc_size = 16889
# end if

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(
    datasets.ReutersC50Dataset(download=True, authors=authors, transform=transform),
    batch_size=args.batch_size, shuffle=False)

# Model
model.linear2 = nn.Linear(model.linear_size, args.n_authors)
if use_cuda:
    model.cuda()
# end if

# Loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# First fold and training mode
reutersloader.dataset.set_fold(0)

# Epoch
for epoch in range(args.epoch):
    # Total losses
    training_loss = 0.0
    test_loss = 0.0

    # Get test data for this fold
    for i, data in enumerate(reutersloader):
        # Inputs and labels
        inputs, labels, time_labels = data

        # To variable
        inputs, outputs = Variable(inputs), Variable(torch.LongTensor(labels))
        if use_cuda:
            inputs, outputs = inputs.cuda(), outputs.cuda()
        # end if

        # Zero grad
        model.zero_grad()

        # Compute output
        log_probs = model(inputs)

        # Loss
        loss = loss_function(log_probs, outputs)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Add
        training_loss += loss.data[0]
    # end for

    # Set test mode
    reutersloader.dataset.set_train(False)

    # Counters
    total = 0.0
    success = 0.0

    # For each test sample
    for i, data in enumerate(reutersloader):
        # Inputs and labels
        inputs, labels, time_labels = data

        # To variable
        inputs, outputs = Variable(inputs), Variable(torch.LongTensor(labels))
        if use_cuda:
            inputs, outputs = inputs.cuda(), outputs.cuda()
        # end if

        # Forward
        model_outputs = model(inputs)
        loss = loss_function(model_outputs, outputs)

        # Take the max as predicted
        _, predicted = torch.max(model_outputs.data, 1)

        # Add to correctly classified word
        success += (predicted == outputs.data).sum()
        total += predicted.size(0)

        # Add loss
        test_loss += loss.data[0]
    # end for

    # Print and save loss
    print(u"Epoch {}, training loss {}, test loss {}, accuracy {}".format(epoch, training_loss, test_loss,
                                                                          success / total * 100.0))
# end for
