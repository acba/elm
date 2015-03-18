# -*- coding: utf-8 -*-

"""
    This file contains MLTools class and all developed methods.
"""

# Python2 support
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .cverror import CVError
from .error import Error

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

        # Training parameters
        self.has_trained = False
        self.last_training_pattern = []

        # Iterative training parameters
        self.has_trained_it = False
        self.trained_it_sw = 0
        self.trained_it_k = 0
        self.last_it_window = []
        self.last_z = []

        # Cross-validation parameters
        self.has_cv = False
        self.cv_name = "Not cross-validated"
        self.cv_error_name = "Not cross-validated"
        self.cv_best_error = "Not cross-validated"
        self.cv_best_params = "Not cross-validated"

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

    # def _ml_search_param(self, database, dataprocess, path_filename, save,
    #                      cv, min_f):
    #     """
    #         Should be overridden.
    #     """
    #     return None
    #
    # def _ml_print_parameters(self):
    #     """
    #         Should be overridden.
    #     """
    #     return None

    def _ml_train(self, training_matrix, params):
        """
            Calculate output_weight values needed to test/predict data.

            If params is provided, this method will use at training phase.
            Else, it will use the default value provided at object
            initialization.

            Arguments:
                training_matrix (numpy.ndarray): a matrix containing all
                    patterns that will be used for training.
                params (list): a list of parameters defined at
                    :func:`ELMKernel.__init__`

            Returns:
                :class:`Error`: training error object containing expected,
                    predicted targets and all error metrics.

            Note:
                Training matrix must have target variables as the first column.
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
        """
            Calculate test predicted values based on previous training.

            Args:
                testing_matrix (numpy.ndarray): a matrix containing all
                    patterns that will be used for testing.
                predicting (bool): Don't set.

            Returns:
                :class:`Error`: testing error object containing expected,
                    predicted targets and all error metrics.

            Note:
                Testing matrix must have target variables as the first column.
        """

        if not self.has_trained:
            raise Exception("Need perform train before test/predict.")

        testing_patterns = testing_matrix[:, 1:]
        testing_expected_targets = testing_matrix[:, 0].reshape(-1, 1)

        testing_predicted_targets = self._local_test(testing_patterns,
                                                     testing_expected_targets,
                                                     predicting)

        testing_errors = Error(testing_expected_targets,
                               testing_predicted_targets,
                               regressor_name=self.regressor_name)

        return testing_errors

    def _ml_predict(self, horizon=1):
        """
            Predict next targets based on previous training.

            Arguments:
                horizon (int): number of predictions.

            Returns:
                numpy.ndarray: a column vector containing all predicted targets.
        """

        if not self.has_trained:
            raise Exception("Error: Train before predict.")

        # Create first new pattern
        new_pattern = np.hstack([self.last_training_pattern[2:],
                                 self.last_training_pattern[0]])

        # Create a fake target (1)
        new_pattern = np.insert(new_pattern, 0, 1).reshape(1, -1)

        predicted_targets = np.zeros((horizon, 1))

        for t_counter in range(horizon):
            te_errors = self._ml_test(new_pattern, predicting=True)

            predicted_value = te_errors.predicted_targets
            predicted_targets[t_counter] = predicted_value

            # Create a new pattern including the actual predicted value
            new_pattern = np.hstack([new_pattern[0, 2:],
                                     np.squeeze(predicted_value)])

            # Create a fake target
            new_pattern = np.insert(new_pattern, 0, 1).reshape(1, -1)

        return predicted_targets

    def _ml_train_it(self, database_matrix, params=[], dataprocess=None,
                            sliding_window=168, k=1, search=False):
        """
            Training method used by Fred 09 paper.
        """

        # Number of dimension/lags/order
        p = database_matrix.shape[1] - 1

        # Number of training/testing procedures
        number_iterations = database_matrix.shape[0] + p - k - sliding_window +1

        # Training set size
        tr_size = sliding_window - p - 1

        new_database_matrix = np.copy(database_matrix)

        # Sum -z_i value to every input pattern, Z = r_t-(p-1)-k
        z = database_matrix[0:-k, 1].reshape(-1, 1) * np.ones((1, p+1))
        new_database_matrix[k:, :] = database_matrix[k:, :] - z

        pr_target = []
        ex_target = []
        for i in range(number_iterations):
            tr_matrix = new_database_matrix[k+i:k+i+tr_size-1, :]
            te_matrix = new_database_matrix[k+i+tr_size, :].reshape(1, -1)

            if dataprocess is not None:
                # Pre-process data
                tr_matrix, te_matrix = dataprocess.auto(tr_matrix, te_matrix)

            # Search best parameters for each window:
            if search:
                data = np.vstack((tr_matrix, te_matrix))
                params = self.search_param(database=data,
                                           dataprocess=dataprocess,
                                           cv="ts", cv_nfolds=3, of="rmse",
                                           eval=50, print_log=False)

            # Train sliding window dataset
            self._ml_train(tr_matrix, params)

            # Predicted target with training_data - z_i ( r_t+1  )
            pr_t = self._ml_test(te_matrix)
            pr_t = pr_t.predicted_targets

            if dataprocess is not None:
                # Pos-process data
                pr_t = dataprocess.reverse_scale_target(pr_t)

            # Sum z_i value to get r'_t+1 = r_t+1 + z_i
            pr_t = pr_t[0] + z[i+tr_size, 0]
            pr_target.append(pr_t)

            # Expected target
            ex_target.append(database_matrix[k+i+tr_size, 0])

            # Last iteration, save window
            if i == number_iterations-1:
                self.has_trained_it = True
                self.trained_it_sw = sliding_window
                self.trained_it_k = k
                self.last_it_window = database_matrix[k+i:k+i+tr_size+1, :]
                self.last_z = database_matrix[-(k+tr_size+1):, 1]

        pr_result = Error(expected=ex_target, predicted=pr_target)

        return pr_result

    def _ml_predict_it(self, horizon=1, dataprocess=None):
        """
            Predict using train_iterative method.
        """

        if not self.has_trained_it:
            raise Exception("Need perform train_it before predict_it.")

        # Number of dimension/lags/order
        p = self.last_it_window.shape[1] - 1
        k = self.trained_it_k
        sliding_window = self.trained_it_sw

        # Training set size
        tr_size = sliding_window - p - 1

        # Fill z with all possible values
        z = self.last_z.flatten()
        if p > 1:
            z = np.append(z, self.last_it_window[-1, 2:].flatten())

        z = np.append(z, self.last_it_window[-1, 0])

        # Sum -z_i value to every input pattern, Z = r_t-(p-1)-k
        database = self.last_it_window
        new_database = database - \
                       z[:database.shape[0]].reshape(-1, 1) * np.ones((1, p+1))

        real_pr_target = []
        ex_target = []
        for i in range(horizon):
            tr_matrix = new_database

            if dataprocess is not None:
                # Pre-process data
                tr_matrix, _ = dataprocess.auto(tr_matrix)

            # Train sliding window dataset, use last trained parameter
            self._ml_train(tr_matrix, params=[])

            # Predicted target with training_data - z_i ( r_t+1  )
            pr_t = self._ml_predict(horizon=1)
            pr_t = pr_t.flatten()

            if dataprocess is not None:
                # Pos-process data
                pr_t = dataprocess.reverse_scale_target(pr_t)

            # Insert predicted value to training dataset (shift the window)
            # Create a new pattern including the actual predicted value
            if p > 1:
                new_pattern = np.hstack([new_database[-1, 2:],
                                         new_database[-1, 0]])
            else:
                new_pattern = new_database[-1, 0]
            new_pattern = np.hstack([pr_t, new_pattern])

            # Update training matrix (with new pattern)
            new_database = np.vstack((new_database[1:, :], new_pattern))

            # Append new value to z
            z = np.append(z, pr_t)

            # Sum z_i value to get r'_t+1 = r_t+1 + z_i
            pr_t = pr_t[0] + z[i+(tr_size+1)]
            real_pr_target.append(pr_t)

        return real_pr_target

    def save_model(self, file_name):
        """
            Save current classifier/regressor to file_name file.
        """

        try:
            with open(file_name, 'wb') as f:
                pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

        except:
            raise Exception("Error while saving ", file_name)

        else:
            print("Saved model as: ", file_name, "\n\n")

    def load_model(self, file_name):
        """
            Load classifier/regressor to memory.
        """

        try:
            with open(file_name, 'rb') as f:
                ml_model = pickle.load(f)
                self.__dict__.update(ml_model)

        except:
            raise Exception("Error while loading ", file_name)

        return self

    def print_cv_log(self):

        print()
        print("Cross-validation: ", self.cv_name)
        print("Error: ", self.cv_error_name)
        print("Error value: ", self.cv_best_error)
        print("Best parameters: ", self.cv_best_params)
        print()

    def get_cv_flag(self):
        return self.has_cv

    def get_cv_params(self):
        return self.cv_best_params


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


def split_sets(data, dataprocess=None, training_percent=None,
               n_test_samples=None, perm=False):
    """
        Split data matrix into training and test matrices.

        Training matrix size will be set using the training_percent
        parameter, so its samples are the firsts samples found at
        data matrix, the rest of samples will be testing matrix.

        If neither training_percent or number_test_samples are set, an error
        will happen, only one of the parameters can be set at a time.

        Arguments:
            data (numpy.ndarray): A matrix containing nxf patterns features.
            dataprocess (:class:`DataProcess`): an object that will pre-process
                database before training. Defaults to None.
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

    # If dataprocess is available applies defined processes
    if dataprocess is not None:
        training_matrix, testing_matrix = dataprocess.auto(training_matrix,
                                                           testing_matrix)

    return training_matrix, testing_matrix


def time_series_cross_validation(ml, database, params, number_folds=10,
                                 dataprocess=None):
    """
        Performs a k-fold cross-validation on a Time Series as described by
        Rob Hyndman.

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

        See Also:
            http://robjhyndman.com/hyndsight/crossvalidation/
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
            # Pre-process data
            training_matrix, testing_matrix = \
                dataprocess.auto(training_matrix, testing_matrix)

        tr_error = ml.train(training_matrix, params)
        te_error = ml.test(testing_matrix)

        if dataprocess is not None:
            # Pos-process data
            tr_error.scale_back(dataprocess)
            te_error.scale_back(dataprocess)

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
            # Pre-process data
            training_matrix, testing_matrix = \
                dataprocess.auto(training_matrix, testing_matrix)

        tr_error = ml.train(training_matrix, params)
        te_error = ml.test(testing_matrix)

        if dataprocess is not None:
            # Pos-process data
            tr_error.scale_back(dataprocess)
            te_error.scale_back(dataprocess)

        training_errors.append(tr_error)
        testing_errors.append(te_error)

    cv_training_error = CVError(training_errors)
    cv_testing_error = CVError(testing_errors)

    return cv_training_error, cv_testing_error


def copy_doc_of(fun):
    def decorator(f):
        f.__doc__ = fun.__doc__
        return f

    return decorator
