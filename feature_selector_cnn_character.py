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
import os
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torchlanguage.models
from torchlanguage import transforms
from tools import dataset, settings
import echotorch.utils

# Argument parser
parser = argparse.ArgumentParser(description="CNN Character Feature Selector for AA (CCSAA)")

# Argument
parser.add_argument("--output", type=str, help="Embedding output file", default='.')
parser.add_argument("--results", type=str, help="Embedding output file", default='.')
parser.add_argument("--start-fold", type=int, help="Starting fold", default=0)
parser.add_argument("--end-fold", type=int, help="Ending fold", default=9)
parser.add_argument("--text-length", type=int, help="Text length", default=20)
parser.add_argument("--n-filters", type=int, help="Number of filters", default=50)
parser.add_argument("--batch-size", type=int, help="Batch-size", default=64)
parser.add_argument("--n-gram", type=str, help="Character n-gram", default='c1')
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Transforms
if args.n_gram == 'c1':
    transform = transforms.Compose([
        transforms.Character(),
        transforms.ToIndex(start_ix=1),
        transforms.ToLength(length=args.text_length, min=True),
        transforms.ToNGram(n=args.text_length, overlapse=True),
        transforms.Reshape((-1, args.text_length))
    ])
else:
    transform = transforms.Compose([
        transforms.Character2Gram(),
        transforms.ToIndex(start_ix=1),
        transforms.ToLength(length=args.text_length, min=True),
        transforms.ToNGram(n=args.text_length, overlapse=True),
        transforms.Reshape((-1, args.text_length))
    ])
# end if

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_AA_dataset(1, settings.training_authors)
reutersc50_dataset.transform = transform
print(u"{} authors".format(reutersc50_dataset.n_authors))
n_authors = reutersc50_dataset.n_authors

# Loss function
loss_function = nn.CrossEntropyLoss()

# Save results
iteration_results = np.zeros(settings.ccsaa_epoch)

# 10-CV
# for k in np.arange(args.start_fold, args.end_fold+1):
# Log
#     print(u"Starting fold {}".format(k))

# Set fold
# reuters_loader_train.dataset.set_fold(k)
# reuters_loader_test.dataset.set_fold(k)

# Model
model = torchlanguage.models.CCSAA(
    text_length=args.text_length,
    vocab_size=settings.voc_sizes['c2']['en'],
    embedding_dim=settings.ccsaa_embedding_dim,
    out_channels=(args.n_filters, args.n_filters, args.n_filters),
    n_classes=settings.n_training_authors
)
if args.cuda:
    model.cuda()
# end if

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=settings.ccsaa_lr, momentum=settings.ccsaa_momentum)

# Best model
best_acc = 0.0

# Epoch
for epoch in range(settings.ccsaa_epoch):
    # Total losses
    training_loss = 0.0
    test_loss = 0.0

    # Train
    model.train()

    # Get test data for this fold
    for i, data in enumerate(reuters_loader_train):
        # Inputs and labels
        input_samples, sample_labels, _ = data

        # Reshape
        input_samples = input_samples.view(-1, args.text_length)

        # Outputs
        output_samples = torch.LongTensor(input_samples.size(0)).fill_(sample_labels[0])

        # For each batch
        for j in np.arange(0, input_samples.size(0) - args.batch_size, args.batch_size):
            # To variable
            inputs, outputs = Variable(input_samples[j:j+args.batch_size]), Variable(output_samples[j:j+args.batch_size])
            if args.cuda:
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
    # end for

    # Counters
    token_total = 0.0
    token_success = 0.0
    total = 0.0
    success = 0.0

    # Eval
    model.eval()

    # For each test sample
    for i, data in enumerate(reuters_loader_test):
        # Inputs and labels
        input_samples, sample_labels, _ = data

        # Reshape
        input_samples = input_samples.view(-1, args.text_length)

        # Outputs
        output_samples = torch.LongTensor(input_samples.size(0)).fill_(sample_labels[0])

        # Label
        label = torch.LongTensor(1).fill_(sample_labels[0])
        author_prob = torch.zeros(1, n_authors)
        if args.cuda:
            author_prob, label = author_prob.cuda(), label.cuda()
        # end if

        # For each batch
        prob_count = 0.0
        for j in np.arange(0, input_samples.size(0) - args.batch_size, args.batch_size):
            # To variable
            inputs, outputs = Variable(input_samples[j:j + args.batch_size]), Variable(
                output_samples[j:j + args.batch_size])
            if args.cuda:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            # end if

            # Forward
            model_outputs = model(inputs)
            loss = loss_function(model_outputs, outputs)

            # Add
            author_prob += torch.sum(model_outputs.data, dim=0)

            # Token success
            _, token_predicted = torch.max(model_outputs, dim=1)
            token_success += int((token_predicted == outputs).sum())
            token_total += inputs.size(0)

            # Add loss
            test_loss += loss.data[0]

            # Prob count
            prob_count += inputs.size(0)
        # end for

        # Prob over time
        author_prob /= prob_count

        # Max over time
        _, predicted = torch.max(author_prob, dim=1)

        # Add to correctly classified word
        success += (predicted == label).sum()
        total += 1.0
    # end for

    # Accuracy
    accuracy = success / total * 100.0
    token_accuracy = token_success / token_total * 100.0

    # Print and save loss
    print(u"Epoch {}, training loss {}, test loss {}, token accuracy {}, accuracy {}".format(
        epoch,
        training_loss,
        test_loss,
        token_accuracy,
        accuracy)
    )

    # Save
    iteration_results[epoch] = token_accuracy

    # Save if best
    if token_accuracy > best_acc:
        best_acc = token_accuracy
        # Save model
        print(u"Saving model with best accuracy {}".format(best_acc))
        torch.save(model.state_dict(), open(
            os.path.join(args.output, u"ccsaa.pth"),
            'wb'))
        torch.save(transform.transforms[1].token_to_ix, open(
            os.path.join(args.output, u"ccsaa.voc.pth"),
            'wb'))
    # end if
# end for

# Log best accuracy
print(u"Best accuracy {}".format(best_acc))

# Save results
torch.save(iteration_results, open(args.results, 'wb'))

# Make average for each iterations
# iteration_results_average = np.average(iteration_results, axis=1)

# Show
print(iteration_results)

# Print as latex
for i in range(iteration_results.shape[0]):
    print(u"({}, {})".format(i+1, iteration_results[i]))
# end for
