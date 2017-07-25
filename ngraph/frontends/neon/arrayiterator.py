# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np
import ngraph as ng
from future.utils import viewitems
import six
from ngraph.frontends.neon import ax
import collections


class ArrayIterator(object):

    def __init__(self, data_arrays, batch_size, total_iterations=None):
        """
        During initialization, the input data will be converted to backend tensor objects
        (e.g. CPUTensor or GPUTensor). If the backend uses the GPU, the data is copied over to the
        device.

        Args:
            data_arrays (ndarray, shape: [# examples, feature size]): Input features of the
                dataset.
            batch_size (int): number of examples in each minibatch
            total_iterations (int): number of minibatches to cycle through on this iterator.
                                    If not provided, it will cycle through all of the data once.
        """
        # Treat singletons like list so that iteration follows same syntax
        self.batch_size = batch_size
        self.axis_names = None
        if isinstance(data_arrays, dict):
            self.data_arrays = {k: v['data'] for k, v in data_arrays.items()}
            self.axis_names = {k: v['axes'] for k, v in data_arrays.items()}
        elif isinstance(data_arrays, collections.Sequence):
            self.data_arrays = {k: x for k, x in enumerate(data_arrays)}
        else:
            self.data_arrays = {0: data_arrays}

        self.keys = list(self.data_arrays.keys())

        if not self.axis_names:
            self.axis_names = {k: None for k in self.keys}

        # just get an arbitrary element for len
        self.ndata = len(self.data_arrays[self.keys[0]])

        if self.ndata < self.batch_size:
            raise ValueError('Number of examples is smaller than the batch size')

        self.start = 0
        self.index = 0

        self.total_iterations = self.nbatches if total_iterations is None else total_iterations

    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return -((self.start - self.ndata) // self.batch_size)

    def make_placeholders(self, include_iteration=False):
        placeholders = {}
        ax.N.length = self.batch_size
        for k, axnm in self.axis_names.items():
            p_axes = ng.make_axes([ax.N])
            for i, sz in enumerate(self.data_arrays[k].shape[1:], 1):
                name = axnm[i] if axnm else None
                if name == ax.REC.name:
                    ax.REC.length = sz
                    _axis = ax.REC
                else:
                    _axis = ng.make_axis(length=sz, name=name)
                p_axes += _axis
            placeholders[k] = ng.placeholder(p_axes)
        if include_iteration:
            placeholders['iteration'] = ng.placeholder(axes=())
        return placeholders

    def reset(self):
        """
        Resets the starting index of this dataset to zero. Useful for calling
        repeated evaluations on the dataset without having to wrap around
        the last uneven minibatch. Not necessary when data is divisible by batch size
        """
        self.start = 0
        self.index = 0

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.

        Yields:
            tuple: The next minibatch which includes both features and labels.
        """
        i1 = self.start
        while self.index < self.total_iterations:
            i1 = (self.start + self.index * self.batch_size) % self.ndata
            bsz = min(self.batch_size, self.ndata - i1)
            oslice1 = slice(i1, i1 + bsz)
            self.index += 1

            if self.batch_size > bsz:
                batch_bufs = {k: np.concatenate([src[oslice1], src[:self.batch_size - bsz]])
                              for k, src in self.data_arrays.items()}
            else:
                batch_bufs = {k: src[oslice1] for k, src in self.data_arrays.items()}

            batch_bufs['iteration'] = self.index
            yield batch_bufs

        self.start = (self.start + self.total_iterations * self.batch_size) % self.ndata


class SequentialArrayIterator(object):

    def __init__(self, data_arrays, time_steps, batch_size,
                 total_iterations=None, reverse_target=False, get_prev_target=False):
        self.get_prev_target = get_prev_target
        self.reverse_target = reverse_target

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.index = 0

        if isinstance(data_arrays, dict):
            self.data_arrays = {k: v for k, v in viewitems(data_arrays)}
        else:
            raise ValueError("Must provide dict as input")

        # just get an arbitrary element for len
        self.ndata = len(six.next(six.itervalues(self.data_arrays)))
        self.ndata = self.ndata // (self.batch_size * self.time_steps) * self.batch_size
        self.ntokens = self.ndata * self.time_steps
        self.nbatches = self.ndata // self.batch_size

        if self.ndata < self.batch_size:
            raise ValueError('Number of examples is smaller than the batch size')

        self.total_iterations = self.nbatches if total_iterations is None else total_iterations

        self.data_arrays = {k: x[:self.ntokens].reshape(
            self.batch_size,
            self.nbatches,
            self.time_steps
        ) for k, x in viewitems(self.data_arrays)}

        if self.reverse_target:
            self.data_arrays['tgt_txt'][:] = self.data_arrays['tgt_txt'][:, :, ::-1]

        if self.get_prev_target:
            self.data_arrays['prev_tgt'] = np.roll(self.data_arrays['tgt_txt'], shift=1, axis=2)

    def make_placeholders(self):
        ax.N.length = self.batch_size
        ax.REC.length = self.time_steps

        p_axes = ng.make_axes([ax.N, ax.REC])
        return {k: ng.placeholder(p_axes) for k in self.data_arrays.keys()}

    def reset(self):
        self.index = 0

    def __iter__(self):
        while self.index < self.total_iterations:
            idx = self.index % self.nbatches
            self.index += 1

            yield {k: np.squeeze(x[:, idx:(idx + 1), :]) for k, x in viewitems(self.data_arrays)}


class RollingWindowIterator(object):

    def __init__(self, data_array, batch_size, seq_len,
                 return_sequences=False, total_iterations=None, stride=1):
        """
        Given an input sequence, generates overlapping windows of samples

        Input: numpy array
            data_array : Numpy array of shape (S, D).
                                S is length of sequence
                                D is input feature dimension
        Output of each iteration: Dictionary of input and output samples
            samples['X'] has size (batch_size, seq_len, D)
            samples['y'] has size (batch_size, 1, D) (if return_sequences is False)
            samples['y'] has size (batch_size, seq_len, D) (if return_sequences is True)

        Example:
            data_array is a numpy array with shape (S, 1): [a1, a2, ..., aS]
            If return_sequences is False:
                Each generated sample will be an input sequence / output single point pairs:
                    Sample0: Input:  [a1, a2, ..., a(seq_len)]
                             Output: a(seq_len + 1)
                    Sample1: Input:  [a(stride +1), a(stride+2), ..., a(stride+seq_len)]
                             Output: a(stride+seq_len +1)
                            ...
                Each iteration will return batch_size number of samples

                If stride = 1, the window will shift by one
                    Hence Sample0 and Sample1 will have (seq_len - 1) elements that are the same
                If stride = seq_len, Sample0 and Sample1 will have no overlapping elements

            If return_sequences is True:
                Each generated sample will be an input sequence / output sequence pairs such as:
                    Sample0: Input:  [a1, a2, ..., a(seq_len)]
                             Output: [a2, a3, ..., a(seq_len + 1)
                    Sample1: Input:  [a(stride +1), a(stride+2), ..., a(stride+seq_len)]
                             Output: [a(stride + 2), a(stride + 3), ..., a(stride+seq_len +1)]
                            ...
                Each iteration will return batch_size number of samples

                If stride = 1, the window will shift by one
                    Hence Sample0 and Sample1 will have (seq_len - 1) elements that are the same
                If stride = seq_len, Sample0 and Sample1 will have no overlapping elements
        seq_len: Width of the rolling window requested
        batch_size: how many samples to return for each iteration
        total_iterations: number of batches to retrieve from the sequence (roll over if necessary)
                         If set to None, will rotate through the whole sequence only once
        """
        self.return_seq = return_sequences
        self.data_array = data_array
        self.seq_len = seq_len

        if (stride is None):
            self.stride = 1
        else:
            self.stride = stride
        if (len(data_array.shape) == 1):
            self.feature_dim = 1
        else:
            self.feature_dim = data_array.shape[1]

        # Treat singletons like list so that iteration follows same syntax
        self.batch_size = batch_size

        # Get the total length of the sequence
        self.ndata = len(self.data_array)

        if self.ndata < self.batch_size:
            raise ValueError('Number of examples is smaller than the batch size')

        self.start = 0
        self.index = 0

        self.total_iterations = self.nbatches if total_iterations is None else total_iterations

    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return -((self.start - self.ndata) // self.batch_size // self.stride)

    def reset(self):
        """
        Resets the starting index of this dataset to zero. Useful for calling
        repeated evaluations on the dataset without having to wrap around
        the last uneven minibatch. Not necessary when data is divisible by batch size
        """
        self.start = 0
        self.index = 0

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.

        Yields:
            dictionary: The next minibatch which includes both features and labels.
                samples['X']: Features, with shape (batch_size, seq_len, feature_dim)
                samples['y']: Labels, with shape (batch_size, 1, feature_dim)
        """
        if (self.return_seq is False):
            out_seq_len = 1
        else:
            out_seq_len = self.seq_len

        samples = {'X': np.squeeze(np.zeros((self.batch_size, self.seq_len, self.feature_dim))),
                   'y': np.squeeze(np.zeros((self.batch_size, out_seq_len, self.feature_dim)))}
        stride = self.stride
        while self.index < self.total_iterations:
            strt_idx = (self.start + self.index * self.batch_size * stride)
            end_idx = strt_idx + self.seq_len
            self.index += 1
            sample_id = 0
            for batch_idx in range(self.batch_size):
                idcs = np.arange(strt_idx + (batch_idx * stride),
                                 end_idx + (batch_idx * stride)) % self.ndata
                samples['X'][sample_id] = self.data_array[idcs]
                if (self.return_seq is False):
                    samples['y'][sample_id] = self.data_array[(idcs[-1] + 1) % self.ndata]
                else:
                    samples['y'][sample_id] = self.data_array[(idcs + 1) % self.ndata]
                sample_id += 1
            samples['iteration'] = self.index
            yield samples
