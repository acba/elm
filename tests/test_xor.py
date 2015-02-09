#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    test_xor
    ----------------------------------

"""

from elm import ELMKernel, ELMRandom
import numpy as np

# output | input
DATABASE_XOR = np.array([[-1, -1, -1],
                         [1, -1, 1],
                         [1, 1, -1],
                         [-1, 1, 1]
                        ])


def test_xor_elmk():

    elmk = ELMKernel()

    elmk.train(DATABASE_XOR)
    te_result = elmk.test(DATABASE_XOR)
    predicted = te_result.predicted_targets

    predicted[predicted < 0] = -1
    predicted[predicted > 0] = 1

    te_result.predicted_targets = predicted

    assert (te_result.get_accuracy() == 1)


def test_xor_elmr():

    elmr = ELMRandom()

    elmr.train(DATABASE_XOR)
    te_result = elmr.test(DATABASE_XOR)
    predicted = te_result.predicted_targets

    predicted[predicted < 0] = -1
    predicted[predicted > 0] = 1

    te_result.predicted_targets = predicted
    assert (te_result.get_accuracy() == 1)