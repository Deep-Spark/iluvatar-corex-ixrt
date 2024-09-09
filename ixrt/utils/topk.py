# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#

import numpy as np


def topk(a, k, axis=-1, largest=True, sorted=True):
    """
    Finds values and indices of the `k` largest/smallest
    elements along the given `axis`.
    Parameters
    ----------
    a: array_like
        Array with given axis at least k.
    k: int
        Number of top elements to look for along the given axis.
    axis: int or None, optional
        Axis along which to find topk. If None, the array is flattened
        before sorting. The default is -1 (the last axis).
    largest: bool, optional
        Controls whether to return largest or smallest elements.
    sorted: bool, optional
        If true the resulting k elements will be sorted by the values.
    Returns
    -------
    topk_values : ndarray
        Array of values of `k` largest/smallest elements
        along the specified `axis`.
    topk_indices: ndarray, int
        Array of indices of `k` largest/smallest elements
        along the specified `axis`.
    See Also
    --------
    sort : Describes sorting algorithms used.
    argsort : Indirect sort.
    partition : Describes partition algorithms used.
    argpartition : Indirect partial sort.
    take_along_axis : Take values from the input array by
                      matching 1d index and data slices.
    Examples
    --------
    One dimensional array:
    >>> x = np.array([3, 1, 2])
    >>> np.topk(x, 2)
    (array([3, 2]), array([0, 2], dtype=int64))
    Two-dimensional array:
    >>> x = np.array([[0, 3, 4], [3, 2, 1], [5, 1, 2]])
    >>> val, ind = np.topk(x, 2, axis=1) # along the last axis
    >>> val
    array([[4, 3],
           [3, 2],
           [5, 2]])
    >>> ind
    array([[2, 1],
           [0, 1],
           [0, 2]])
    >>> val, ind = np.topk(x, 2, axis=None) # along the flattened array
    >>> val
    array([5, 4])
    >>> ind
    array([6, 2])
    >>> val, ind = np.topk(x, 2, axis=0) # along the first axis
    >>> val
    array([[5, 3, 4],
           [3, 2, 2]])
    >>> ind
    array([[2, 0, 0],
           [1, 1, 2]])
    """
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size - k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
    else:
        index_array = np.argpartition(a, k - 1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis
        )
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis
        )
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices
