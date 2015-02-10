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

    try:
        elmk.train(DATABASE_XOR)
        te_result = elmk.test(DATABASE_XOR)
        predicted = te_result.predicted_targets

        predicted[predicted < 0] = -1
        predicted[predicted > 0] = 1

        te_result.predicted_targets = predicted
        
    except:
        ERROR = 1
    else:
        ERROR = 0

    assert (ERROR == 0)


def test_xor_elmr():

    elmr = ELMRandom()

    try:
        elmr.train(DATABASE_XOR)
        te_result = elmr.test(DATABASE_XOR)
        predicted = te_result.predicted_targets

        predicted[predicted < 0] = -1
        predicted[predicted > 0] = 1

        te_result.predicted_targets = predicted

    except:
        ERROR = 1
    else:
        ERROR = 0

    assert (ERROR == 0)