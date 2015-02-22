# -*- coding: utf-8 -*-

"""
    This file contains CVError class and all developed methods.
"""

import numpy as np


class CVError(object):
    """
        CVError is a class that saves :class:`Error` objects from all folds
        of a cross-validation method.

        Attributes:
            fold_errors (list of :class:`Error`): a list of all Error objects
                created through cross-validation process.
            all_fold_errors (dict): a dictionary containing lists of error
                values of all folds.
            all_fold_mean_errors (dict): a dictionary containing the mean of
                *all_fold_errors* lists.
    """

    def __init__(self, fold_errors):
        self.fold_errors = fold_errors

        self.all_fold_errors = {}
        self.all_fold_mean_errors = {}

        for error in self.fold_errors[0].available_error_metrics:
            self.all_fold_errors[error] = []
            self.all_fold_mean_errors[error] = -99

        self.calc_metrics()

    def calc_metrics(self):
        """
            Calculate a folds mean of all error metrics.

            Available error metrics are "rmse", "mse", "mae", "me", "mpe",
            "mape", "std", "hr", "hr+", "hr-" and "accuracy".
        """

        for fold in self.fold_errors:
            for error in fold.dict_errors:
                if fold.dict_errors[error] == "Not calculated":
                    fold.dict_errors[error] = fold.get(error)

                self.all_fold_errors[error].append(fold.dict_errors[error])

        for error in sorted(self.all_fold_errors.keys()):
            self.all_fold_mean_errors[error] = \
                np.mean(self.all_fold_errors[error])

    def print_errors(self):
        """
            Print a mean of all error through all folds.
        """

        for error in sorted(self.all_fold_errors.keys()):
            print(error, " mean:", self.all_fold_mean_errors[error])
            print(self.all_fold_errors[error], "\n")

        print()

    def get(self, error):
        return self.all_fold_mean_errors[error]

    def get_rmse(self):
        return self.all_fold_mean_errors["rmse"]

    def get_accuracy(self):
        return self.all_fold_mean_errors["accuracy"]
