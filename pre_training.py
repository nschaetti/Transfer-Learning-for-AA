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
from modules import CNNClassifier
from echotorch import datasets
from echotorch.transforms import text
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import os
from authors import authors as aa
import copy

# Settings
embedding_dim = 50
use_cuda = False
voc_size = 0

# Argument parser
parser = argparse.ArgumentParser(description="CNN pre-training")

# Argument
parser.add_argument("--output", type=str, help="Pre-trained classifier output file", default='.')
parser.add_argument("--n-features", type=int, help="Number of features", default=60)
parser.add_argument("--n-authors", type=int, help="Number of authors to pre-train", default=30)
parser.add_argument("--character-gram", type=int, help="Character gram", default=2)
parser.add_argument("--batch-size", type=int, help="Batch size", default=64)
parser.add_argument("--epoch", type=int, help="Epoch", default=500)
args = parser.parse_args()

# Authors
authors = aa[-args.n_authors:]

# Word embedding
if args.character_gram == 1:
    transform = text.Character(fixed_length=7978, start_ix=1)
    window_size = 7978
    voc_size = 60
elif args.character_gram == 2:
    transform = text.Character2Gram(fixed_length=7978, start_ix=1)
    window_size = 7978
    voc_size = 1653
elif args.character_gram == 3:
    transform = text.Character3Gram(fixed_length=7977, start_ix=1)
    window_size = 7977
    voc_size = 16889
# end if

# Reuters C50 dataset
reutersloader = torch.utils.data.DataLoader(
    datasets.ReutersC50Dataset(download=True, authors=authors, transform=transform),
    batch_size=1, shuffle=False)

# Model
model = CNNClassifier(voc_size=voc_size, embedding_dim=embedding_dim, n_authors=args.n_authors, window_size=window_size, n_features=args.n_features)
if use_cuda:
    model.cuda()
# end if

# Best model
best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0

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

    # Test accuracy
    accuracy = success / total * 100.0

    # Print and save loss
    print(u"Epoch {}, training loss {}, test loss {}, accuracy {}".format(epoch, training_loss, test_loss,
                                                                          accuracy))

    # Best model?
    if accuracy > best_acc:
        best_acc = accuracy
        best_model = copy.deepcopy(model.state_dict())
    # end if
# end for

# Show best model
print(u"Best accuracy : {}".format(best_acc))

# Load best model
model.load_state_dict(best_model)

# Dict
gram_to_ix = reutersloader.dataset.transform.gram_to_ix

# Save model
torch.save((model, gram_to_ix), open(os.path.join(args.output, u"cnn_pretrained." + str(args.n_features) + u"." + u".p"), 'wb'))
