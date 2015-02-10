# -*- coding: utf-8 -*-

"""
    This file contains MLTools class and all developed methods.
"""

# Python2 support
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import pickle


class MLTools(object):
    """
        A Python implementation of several methods needed for machine learning
        classification/regression.

        Attributes:
            last_training_pattern (numpy.ndarray): Full path to the package
                to  test.
            has_trained (boolean): package_name str
            cv_best_rmse (float): package_name str

    """

    def __init__(self):
        self.last_training_pattern = []

        self.has_trained = False

        self.cv_best_rmse = "Not cross-validated"

    #################################################
    ########### Methods to be overridden ############
    #################################################

    def _local_train(self, training_patterns, training_expected_targets,
                     params):
        """
            Should be overridden.
        """
        return None

    def _local_test(self, testing_patterns, testing_expected_targets,
                    predicting):
        """
            Should be overridden.
        """
        return None

    # ########################
    # Public Methods
    # ########################

    def _ml_search_param(self, database, dataprocess, path_filename, save,
                         cv, min_f):
        """
            Should be overridden.
        """
        return None

    def _ml_print_parameters(self):
        """
            Should be overridden.
        """
        return None

    def _ml_predict(self, horizon=1):
        """
            Predict next targets based on previous training.

            Arguments:
                horizon (int): number of predictions.

            Returns:
                numpy.ndarray: a column vector containing all predicted targets.
        """

        if not self.has_trained:
            print("Error: Train before predict.")
            return

        # Create first new pattern
        new_pattern = np.hstack([self.last_training_pattern[2:],
                                 self.last_training_pattern[0]])

        # Create a fake target (1)
        new_pattern = np.insert(new_pattern, 0, 1).reshape(1, -1)

        predicted_targets = np.zeros((horizon, 1))

        for t_counter in range(horizon):
            te_errors = self.test(new_pattern, predicting=True)

            predicted_value = te_errors.predicted_targets
            predicted_targets[t_counter] = predicted_value

            # Create a new pattern including the actual predicted value
            new_pattern = np.hstack([new_pattern[0, 2:],
                                     np.squeeze(predicted_value)])

            # Create a fake target
            new_pattern = np.insert(new_pattern, 0, 1).reshape(1, -1)

        return predicted_targets

    def _ml_train(self, training_matrix, params):
        """
            wr

        """

        training_patterns = training_matrix[:, 1:]
        training_expected_targets = training_matrix[:, 0]

        training_predicted_targets = \
            self._local_train(training_patterns,
                              training_expected_targets,
                              params)

        training_errors = Error(training_expected_targets,
                                training_predicted_targets,
                                regressor_name=self.regressor_name)

        # Save last pattern for posterior predictions
        self.last_training_pattern = training_matrix[-1, :]
        self.has_trained = True

        return training_errors

    def _ml_test(self, testing_matrix, predicting=False):
        """ wr

        """

        testing_patterns = testing_matrix[:, 1:]
        testing_expected_targets = testing_matrix[:, 0].reshape(-1, 1)

        testing_predicted_targets = self._local_test(testing_patterns,
                                                     testing_expected_targets,
                                                     predicting)

        testing_errors = Error(testing_expected_targets,
                               testing_predicted_targets,
                               regressor_name=self.regressor_name)

        return testing_errors

    def _ml_train_iterative(self, database_matrix, params=[],
                            sliding_window=168, k=1):
        """
            Training method used by Fred 09 paper.
        """

        # Number of dimension/lags/order
        p = database_matrix.shape[1] - 1

        # Amount of training/testing procedures
        number_iterations = database_matrix.shape[0] + p - k - sliding_window + 1
        print("Number of iterations: ", number_iterations)

        # Training set size
        tr_size = sliding_window - p - 1

        # Sum -z_i value to every input pattern, Z = r_t-(p-1)-k
        z = database_matrix[0:-k, 1].reshape(-1, 1) * np.ones((1, p))
        database_matrix[k:, 1:] = database_matrix[k:, 1:] - z

        pr_target = []
        ex_target = []
        for i in range(number_iterations):
            # Train with sliding window training dataset
            self._ml_train(database_matrix[k+i:k+i+tr_size-1, :], params)

            # Predicted target with training_data - z_i ( r_t+1  )
            pr_t = self._ml_predict(horizon=1)

            # Sum z_i value to get r'_t+1 = r_t+1 + z_i
            pr_t = pr_t[0][0] + z[i, 0]
            pr_target.append(pr_t)

            # Expected target
            ex_target.append(database_matrix[k+i+tr_size, 0])

        pr_result = Error(expected=ex_target, predicted=pr_target)

        return pr_result

    def save_regressor(self, file_name):
        """
            Save current classifier/regressor to file_name file.
        """

        try:
            # First save all class attributes

            file = file_name
            with open(file, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        except:
            print("Error while saving ", file_name)
            return
        else:
            print("Saved model as: ", file_name)

    def load_regressor(self, file_name):
        """
            Load classifier/regressor to memory.
        """

        try:
            # First load all class attributes

            file = file_name
            with open(file, 'rb') as f:
                self = pickle.load(f)

        except:
            print("Error while loading ", file_name)
            return

        return self


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

    def _calc(self, name, expected, predicted):
        """
            a
        """

        if self.dict_errors[name] == "Not calculated":
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

    def calc_metrics(self):
        """
            Calculate all error metrics.

            Available error metrics are "rmse", "mse", "mae", "me", "mpe",
            "mape", "std", "hr", "hr+", "hr-" and "accuracy".

        """

        for error in sorted(self.dict_errors.keys()):
            self._calc(error, self.expected_targets, self.predicted_targets)

    def print_errors(self):
        """
            Print all errors metrics.

            Note:
                For better printing format, install :mod:`prettytable`.

        """

        self.calc_metrics()

        try:
            from prettytable import PrettyTable

            table = PrettyTable(["Error", "Value"])
            table.align["Error"] = "l"
            table.align["Value"] = "l"

            for error in sorted(self.dict_errors.keys()):
                table.add_row([error, np.around(self.dict_errors[error], decimals=8)])

            print()
            print(table.get_string(sortby="Error"))
            print()

        except ImportError:
            print("For better table format install 'prettytable' module.")

            print()
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

        try:
            from scipy import stats
        except ImportError:
             raise ImportError("Need 'scipy.stats' module to calculate "
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

        try:
            from scipy import stats
        except ImportError:
            raise ImportError("Need 'scipy.stats' module to calculate "
                              "shapiro-wilk test.")

        error = (self.expected_targets - self.predicted_targets).flatten()

         # Calculate Shapiro-Wilk normality index
        sw_statistic, sw_p_value = stats.shapiro(error)

        return sw_statistic, sw_p_value


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


def read(file_name):
    """
        Read data from txt file.

        Arguments:
            file_name (str): path and file name.

        Returns:
            numpy.ndarray: a matrix containing all read data.
    """

    data = np.loadtxt(file_name)

    return data


def write(file_name, data):
    """
        Write data to txt file.

        Arguments:
            file_name (str): path and file name.
            data (numpy.ndarray): data to be written.

    """

    np.savetxt(file_name, data)


def split_sets(data, training_percent=None, n_test_samples=None, perm=False):
    """
        Split data matrix into training and test matrices.

        Training matrix size will be set using the training_percent
        parameter, so its samples are the firsts samples found at
        data matrix, the rest of samples will be testing matrix.

        If neither training_percent or number_test_samples are set, an error
        will happen, only one of the parameters can be set at a time.

        Arguments:
            data (numpy.ndarray): A matrix containing nxf patterns features.
            training_percent (float): An optional parameter used to
                calculate the number of patterns of training matrix.
            n_test_samples (int): An optional parameter used to set the
                number of patterns of testing matrix.
            perm (bool): A flag to choose if should permute(shuffle) database
                before splitting sets.

        Returns:
            tuple: Both training and test matrices.

    """

    number_of_samples = data.shape[0]

    # Permute data
    if perm:
        np.random.shuffle(data)

    if n_test_samples is not None:
        training_samples = number_of_samples - n_test_samples
    elif training_percent is not None:
        training_samples = round(number_of_samples * training_percent)
    else:
        raise Exception("Error: Missing \"training_percent\" or \"numberTestSamples\""
                        "parameter.")

    training_matrix = data[0:training_samples, :]
    testing_matrix = data[training_samples:, :]

    return training_matrix, testing_matrix


def time_series_cross_validation(ml, database, params, number_folds=10,
                                 dataprocess=None):
    """
        Performs a k-fold cross-validation on a Time Series as described by
        Rob Hyndman.

        See Also:
            http://robjhyndman.com/hyndsight/crossvalidation/

        Arguments:
            ml (:class:`ELMKernel` or :class:`ELMRandom`):
            database (numpy.ndarray): uses 'data' matrix to perform
                cross-validation.
            params (list): list of parameters from *ml* to train/test.
            number_folds (int): number of folds to be created from training and
                testing matrices.
            dataprocess (:class:`DataProcess`): an object that will pre-process
                database before training. Defaults to None.

        Returns:
            tuple: tuple of :class:`CVError` from training and testing.
    """

    if number_folds < 2:
        print("Error: Must have at least 2-folds.")
        return

    number_patterns = database.shape[0]
    fold_size = round(number_patterns / number_folds)

    folds = []
    for k in range(number_folds):
        folds.append(database[k * fold_size:(k + 1) * fold_size, :])

    training_errors = []
    testing_errors = []

    training_matrix = folds[0]
    testing_matrix = []
    for k in range(number_folds - 1):
        if k > 0:
            training_matrix = \
                np.concatenate((training_matrix, testing_matrix), axis=0)

        testing_matrix = folds[k + 1]

        # If dataprocess is available applies defined processes
        if dataprocess is not None:
            training_matrix, testing_matrix = \
                dataprocess.auto(training_matrix, testing_matrix)

        tr_error = ml.train(training_matrix, params)
        te_error = ml.test(testing_matrix)

        training_errors.append(tr_error)
        testing_errors.append(te_error)

    cv_training_error = CVError(training_errors)
    cv_testing_error = CVError(testing_errors)

    return cv_training_error, cv_testing_error


def kfold_cross_validation(ml, database, params, number_folds=10,
                           dataprocess=None):
    """
        Performs a k-fold cross-validation.

        Arguments:
            ml (:class:`ELMKernel` or :class:`ELMRandom`):
            database (numpy.ndarray): uses 'data' matrix to perform
                cross-validation.
            params (list): list of parameters from *ml* to train/test.
            number_folds (int): number of folds to be created from training and
                testing matrices.
            dataprocess (:class:`DataProcess`): an object that will pre-process
                database before training. Defaults to None.

        Returns:
            tuple: tuple of :class:`CVError` from training and testing.

    """

    if number_folds < 2:
        print("Error: Must have at least 2-folds.")
        return

    # Permute patterns
    np.random.shuffle(database)

    # Number of dimensions considering only 1 output
    n_dim = database.shape[1] - 1
    number_patterns = database.shape[0]
    fold_size = np.ceil(number_patterns / number_folds)

    folds = []
    for k in range(number_folds):
        folds.append(database[k * fold_size: (k + 1) * fold_size, :])

    training_errors = []
    testing_errors = []

    for k in range(number_folds):

        # Training matrix is all folds except "k"
        training_matrix = \
            np.array(folds[:k] + folds[k+1:-1]).reshape(-1, n_dim + 1)
        if k < number_folds - 1:
            training_matrix = np.vstack((training_matrix, folds[-1]))

        testing_matrix = folds[k]

        # If dataprocess is available applies defined processes
        if dataprocess is not None:
            training_matrix, testing_matrix = \
                dataprocess.auto(training_matrix, testing_matrix)

        training_errors.append(ml.train(training_matrix, params))
        testing_errors.append(ml.test(testing_matrix))

    cv_training_error = CVError(training_errors)
    cv_testing_error = CVError(testing_errors)

    return cv_training_error, cv_testing_error


def copy_doc_of(fun):
    def decorator(f):
        f.__doc__ = fun.__doc__
        return f

    return decorator
