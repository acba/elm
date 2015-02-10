#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    test_regression
    ----------------------------------

    Datasets used were from sklearn.datasets

    import numpy as np
    from sklearn.datasets import load_boston, load_diabetes

    data = load_boston()
    data = np.hstack((data["target"].reshape(-1, 1), data["data"]))
    np.savetxt("boston.data", data)

    data = load_diabetes()
    data = np.hstack((data["target"].reshape(-1, 1), data["data"]))
    np.savetxt("diabetes.data", data)

"""

import elm


def test_elmk_boston():

    # load dataset
    data = elm.read("tests/data/boston.data")

    # create a regressor
    elmk = elm.ELMKernel()

    try:
        # search for best parameter for this dataset
        # elmk.search_param(data, cv="kfold", of="rmse")

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
    # te_result.get_rmse()
    # assert (te_result.get_rmse() <= 20)


def test_elmk_diabetes():

    # load dataset
    data = elm.read("tests/data/diabetes.data")

    # create a regressor
    elmk = elm.ELMKernel()

    try:
        # search for best parameter for this dataset
        # elmk.search_param(data, cv="kfold", of="rmse")

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
    # assert (te_result.get_rmse() <= 70)


def test_elmr_boston():

    # load dataset
    data = elm.read("tests/data/boston.data")

    # create a regressor
    elmr = elm.ELMRandom()

    try:
        # search for best parameter for this dataset
        # elmr.search_param(data, cv="kfold", of="rmse")

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


def test_elmr_diabetes():

    # load dataset
    data = elm.read("tests/data/diabetes.data")

    # create a regressor
    elmr = elm.ELMRandom()

    try:
        # search for best parameter for this dataset
        # elmr.search_param(data, cv="kfold", of="rmse")

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
    # assert (te_result.get_rmse() <= 70)
