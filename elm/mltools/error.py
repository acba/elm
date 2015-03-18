# -*- coding: utf-8 -*-

"""
    This file contains Error class and all developed methods.
"""

import numpy as np

try:
    from prettytable import PrettyTable
except ImportError:
    _PRETTYTABLE_AVAILABLE = 0
else:
    _PRETTYTABLE_AVAILABLE = 1

try:
    from scipy import stats
except ImportError:
    _SCIPY_STATS_AVAILABLE = 0
else:
    _SCIPY_STATS_AVAILABLE = 1


class Error(object):
    """
        Error is a class that saves expected and predicted values to calculate
        error metrics.

        Attributes:
            regressor_name (str): Deprecated.
            expected_targets (numpy.ndarray): array of expected values.
            predicted_targets (numpy.ndarray): array of predicted values.
            dict_errors (dict): a dictionary containing all calculated errors
                and their values.

    """

    available_error_metrics = ["rmse", "mse", "mae", "me", "mpe", "mape",
                               "std", "hr", "hr+", "hr-", "accuracy"]

    def __init__(self, expected, predicted, regressor_name=""):

        if type(expected) is list:
            expected = np.array(expected)
        if type(predicted) is list:
            predicted = np.array(predicted)

        expected = expected.flatten()
        predicted = predicted.flatten()

        self.regressor_name = regressor_name
        self.expected_targets = expected
        self.predicted_targets = predicted

        self.dict_errors = {}
        for error in self.available_error_metrics:
            self.dict_errors[error] = "Not calculated"

    def _calc(self, name, expected, predicted, force=False):
        """
            a
        """

        if self.dict_errors[name] == "Not calculated" or force:
            if name == "mae":
                error = expected - predicted
                self.dict_errors[name] = np.mean(np.fabs(error))

            elif name == "me":
                error = expected - predicted
                self.dict_errors[name] = error.mean()

            elif name == "mse":
                error = expected - predicted
                self.dict_errors[name] = (error ** 2).mean()

            elif name == "rmse":
                error = expected - predicted
                self.dict_errors[name] = np.sqrt((error ** 2).mean())

            elif name == "mpe":

                if np.count_nonzero(expected != 0) == 0:
                    self.dict_errors[name] = np.nan
                else:
                    # Remove all indexes that have 0, so I can calculate
                    # relative error
                    find_zero = expected != 0
                    _et = np.extract(find_zero, expected)
                    _pt = np.extract(find_zero, predicted)

                    relative_error = (_et - _pt) / _et

                    self.dict_errors[name] = 100 * relative_error.mean()

            elif name == "mape":

                if np.count_nonzero(expected != 0) == 0:
                    self.dict_errors[name] = np.nan
                else:
                    # Remove all indexes that have 0, so I can calculate
                    # relative error
                    find_zero = expected != 0
                    _et = np.extract(find_zero, expected)
                    _pt = np.extract(find_zero, predicted)

                    relative_error = (_et - _pt) / _et

                    self.dict_errors[name] = \
                        100 * np.fabs(relative_error).mean()

            elif name == "std":
                error = expected - predicted
                self.dict_errors[name] = np.std(error)

            elif name == "hr":
                _c = expected * predicted

                if np.count_nonzero(_c != 0) == 0:
                    self.dict_errors[name] = np.nan
                else:
                    self.dict_errors[name] = np.count_nonzero(_c > 0) / \
                                             np.count_nonzero(_c != 0)

            elif name == "hr+":
                _a = expected
                _b = predicted

                if np.count_nonzero(_b > 0) == 0:
                    self.dict_errors[name] = np.nan
                else:
                    self.dict_errors[name] = \
                        np.count_nonzero((_a > 0) * (_b > 0)) / \
                        np.count_nonzero(_b > 0)

            elif name == "hr-":
                _a = expected
                _b = predicted

                if np.count_nonzero(_b < 0) == 0:
                    self.dict_errors[name] = np.nan
                else:
                    self.dict_errors[name] = \
                        np.count_nonzero((_a < 0) * (_b < 0)) / \
                        np.count_nonzero(_b < 0)

            elif name == "accuracy":
                _a = expected.astype(int)
                _b = np.round(predicted).astype(int)

                self.dict_errors[name] = np.count_nonzero(_a == _b) / _b.size

            else:
                print("Error:", name,
                      "- Invalid error or not available to calculate.")
                return

    def calc_metrics(self, expected_targets, predicted_targets, force=False):
        """
            Calculate all error metrics.

            Available error metrics are "rmse", "mse", "mae", "me", "mpe",
            "mape", "std", "hr", "hr+", "hr-" and "accuracy".

        """

        for error in sorted(self.dict_errors.keys()):
            self._calc(error, expected_targets, predicted_targets, force)

    def print_errors(self, dataprocess=None):
        """
            Print all errors metrics.

            Note:
                For better printing format, install :mod:`prettytable`.

        """

        # if dataprocess is not None:
        #     exp = dataprocess.reverse_scale_target(self.expected_targets)
        #     pre = dataprocess.reverse_scale_target(self.predicted_targets)
        #
        #     self.calc_metrics(exp, pre, force=True)
        # else:
        #     self.calc_metrics(self.expected_targets, self.predicted_targets,
        #                       force=True)

        self.calc_metrics(self.expected_targets,
                          self.predicted_targets,
                          force=True)

        if _PRETTYTABLE_AVAILABLE:

            table = PrettyTable(["Error", "Value"])
            table.align["Error"] = "l"
            table.align["Value"] = "l"

            for error in sorted(self.dict_errors.keys()):
                table.add_row([error, np.around(self.dict_errors[error], decimals=8)])

            print()
            print(table.get_string(sortby="Error"))
            print()

        else:
            print("For better table format install 'prettytable' package.")

            print()
            print("Error | ", "Value")
            for error in sorted(self.dict_errors.keys()):
                print(error, np.around(self.dict_errors[error], decimals=8))
            print()

    def print_values(self):
        """
            Print expected and predicted values.
        """

        print("Expected: ", self.expected_targets.reshape(1, -1), "\n",
              "Predicted: ", self.predicted_targets.reshape(1, -1), "\n")

    def get(self, error):
        """
            Calculate and return value of an error.

            Arguments:
                error (str): Error to be calculated.

            Returns:
                float: value of desired error.
        """
        self._calc(error, self.expected_targets, self.predicted_targets)
        return self.dict_errors[error]

    def get_std(self):
        self._calc("std", self.expected_targets, self.predicted_targets)
        return self.dict_errors["std"]

    def get_mae(self):
        self._calc("mae", self.expected_targets, self.predicted_targets)
        return self.dict_errors["mae"]

    def get_mse(self):
        self._calc("mse", self.expected_targets, self.predicted_targets)
        return self.dict_errors["mse"]

    def get_rmse(self):
        self._calc("rmse", self.expected_targets, self.predicted_targets)
        return self.dict_errors["rmse"]

    def get_mpe(self):
        self._calc("mpe", self.expected_targets, self.predicted_targets)
        return self.dict_errors["mpe"]

    def get_mape(self):
        self._calc("mape", self.expected_targets, self.predicted_targets)
        return self.dict_errors["mape"]

    def get_me(self):
        self._calc("me", self.expected_targets, self.predicted_targets)
        return self.dict_errors["me"]

    def get_hr(self):
        self._calc("hr", self.expected_targets, self.predicted_targets)
        return self.dict_errors["hr"]

    def get_hrm(self):
        self._calc("hr-", self.expected_targets, self.predicted_targets)
        return self.dict_errors["hr-"]

    def get_hrp(self):
        self._calc("hr+", self.expected_targets, self.predicted_targets)
        return self.dict_errors["hr+"]

    def get_accuracy(self):
        self._calc("accuracy", self.expected_targets, self.predicted_targets)
        return self.dict_errors["accuracy"]

    def get_error(self):
        return (self.expected_targets - self.predicted_targets).flatten()

    def get_anderson(self):
        """
            Anderson-Darling test for data coming from a particular
            distribution.

            Returns:
                tuple: statistic value, critical values and significance values.

            Note:
                Need scipy.stats module to perform Anderson-Darling test.
        """

        if not _SCIPY_STATS_AVAILABLE:
            raise ImportError("Need 'scipy' module to calculate "
                              "anderson-darling test.")

        error = (self.expected_targets - self.predicted_targets).flatten()

        # from matplotlib import pyplot as plt
        # import matplotlib.mlab as mlab
        #
        # plt.figure(figsize=(24.0, 12.0))
        # _, bins, _ = plt.hist(error, 50, normed=1)
        # _mu = np.mean(error)
        # _sigma = np.std(error)
        # plt.plot(bins, mlab.normpdf(bins, _mu, _sigma))
        # plt.show()
        # plt.close()

        # Calculate Anderson-Darling normality test index
        ad_statistic, ad_c, ad_s = stats.anderson(error, "norm")

        return ad_statistic, ad_c, ad_s

    def get_shapiro(self):
        """
            Perform the Shapiro-Wilk test for normality.

            Returns:
                tuple: statistic value and p-value.

            Note:
                Need scipy.stats module to perform Shapiro-Wilk test.
        """

        if not _SCIPY_STATS_AVAILABLE:
            raise ImportError("Need 'scipy' module to calculate "
                              "shapiro-wilk test.")

        error = (self.expected_targets - self.predicted_targets).flatten()

        # Calculate Shapiro-Wilk normality index
        sw_statistic, sw_p_value = stats.shapiro(error)

        return sw_statistic, sw_p_value

    def scale_back(self, dataprocess):

        self.expected_targets = \
            dataprocess.reverse_scale_target(self.expected_targets)
        self.predicted_targets = \
            dataprocess.reverse_scale_target(self.predicted_targets)

        self.calc_metrics(self.expected_targets, self.predicted_targets,
                          force=True)