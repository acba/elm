#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    test_classification
    ----------------------------------

    Datasets used were from sklearn.datasets

    import numpy as np
    from sklearn.datasets import load_boston, load_diabetes, load_iris

    data = load_iris()
    data = np.hstack((data["target"].reshape(-1, 1), data["data"]))
    np.savetxt("iris.data", data)

"""

import elm
import numpy as np


def test_elmk_iris():

    # load dataset
    data = elm.read("tests/data/iris.data")

    # create a regressor
    elmk = elm.ELMKernel()

    try:
        # search for best parameter for this dataset
        elmk.search_param(data, cv="kfold", of="accuracy", eval=10)

        # split data in training and testing sets
        tr_set, te_set = elm.split_sets(data, training_percent=.8, perm=True)

        #train and test
        tr_result = elmk.train(tr_set)
        te_result = elmk.test(te_set)
    except:
        ERROR = 1
    else:
        ERROR = 0

    assert (ERROR == 0)

    # te_result.predicted_targets = np.round(te_result.predicted_targets)
    # assert (te_result.get_accuracy() <= 20)


def test_elmr_iris():

    # load dataset
    data = elm.read("tests/data/iris.data")

    # create a regressor
    elmr = elm.ELMRandom()

    try:
        # search for best parameter for this dataset
        elmr.search_param(data, cv="kfold", of="accuracy", eval=10)

        # split data in training and testing sets
        tr_set, te_set = elm.split_sets(data, training_percent=.8, perm=True)

        #train and test
        tr_result = elmr.train(tr_set)
        te_result = elmr.test(te_set)
    except:
        ERROR = 1
    else:
        ERROR = 0

    assert (ERROR == 0)

    # assert (te_result.get_rmse() <= 20)
