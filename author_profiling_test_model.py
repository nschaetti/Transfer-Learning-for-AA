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
from tools import settings
from torch.autograd import Variable


# Test model
def test_model(model, epoch, train_loader, valid_loader, loss_function, optimizer, transformer, output, cuda=False):
    # Best model
    best_acc = 0.0

    # Epoch
    for epoch in range(epoch):
        # Total losses
        training_loss = 0.0
        training_total = 0.0
        test_loss = 0.0
        test_total = 0.0

        # For each training set
        for data in train_loader:
            # Inputs and labels
            inputs, labels = data

            # Gender
            country = labels[:, 1]

            # Batch size
            data_batch_size = inputs.size(0)

            # Merge batch and authors
            inputs = inputs.view(-1, settings.min_length)
            country = country.view(data_batch_size * 100)

            # Variable and CUDA
            inputs, country = Variable(inputs), Variable(country)
            if cuda:
                inputs, country = inputs.cuda(), country.cuda()
            # end if

            # Zero grad
            model.zero_grad()

            # Compute output
            log_probs = model(inputs)

            # Loss
            loss = loss_function(log_probs, country)

            # Backward and step
            loss.backward()
            optimizer.step()

            # Add
            training_loss += loss.data[0]
            training_total += 1.0
        # end for

        # Counters
        total = 0.0
        success = 0.0

        # For validation set
        for data in valid_loader:
            # Inputs and labels
            inputs, labels = data

            # Gender
            country = labels[:, 1]

            # Batch size
            data_batch_size = inputs.size(0)

            # Merge batch and authors
            inputs = inputs.view(-1, settings.min_length)
            country = country.view(data_batch_size * 100)

            # Variable and CUDA
            inputs, country = Variable(inputs), Variable(country)
            if cuda:
                inputs, country = inputs.cuda(), country.cuda()
            # end if

            # Forward
            model_outputs = model(inputs)

            # Compute loss
            loss = loss_function(model_outputs, country)

            # Take the max as predicted
            _, predicted = torch.max(model_outputs.data, 1)

            # Add to correctly classified word
            success += (predicted == country.data).sum()
            total += predicted.size(0)

            # Add loss
            test_loss += loss.data[0]
            test_total += 1.0
        # end for

        # Accuracy
        accuracy = success / total * 100.0

        # Print and save loss
        print(u"Epoch {}, training loss {}, test loss {}, accuracy {}".format(epoch, training_loss / training_total,
                                                                              test_loss / test_total, accuracy))

        # Save if better
        if accuracy > best_acc:
            best_acc = accuracy
            print(u"Saving model with best accuracy {}".format(best_acc))
            torch.save(
                transformer.transforms[3].token_to_ix,
                open(output +"voc.p", 'wb')
            )
            torch.save(
                model.state_dict(),
                open(output + ".p", 'wb')
            )
        # end if
    # end for
# end test_model
