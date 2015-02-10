# -*- coding: utf-8 -*-

"""
    This file contains ELMKernel classes and all developed methods.
"""

# Python2 support
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .mltools import *

import numpy as np
import optunity
import ast

import sys
if sys.version_info < (3, 0):
    import ConfigParser as configparser
else:
    import configparser

try:
    from scipy.special import expit
except ImportError:
    _SCIPY = 0
else:
    _SCIPY = 1

# Find configuration file
from pkg_resources import Requirement, resource_filename
_ELMR_CONFIG = resource_filename(Requirement.parse("elm"), "elm/elmr.cfg")


class ELMRandom(MLTools):
    """
        A Python implementation of ELM Random Neurons defined by Huang[1].

        An ELM is a single-hidden layer feedforward network (SLFN) proposed by
        Huang  back in 2006, in 2012 the author revised and introduced a new
        concept of using kernel functions to his previous work.

        This implementation currently accepts both methods proposed at 2012,
        random neurons and kernel functions to estimate classifier/regression
        functions.

        Let the dimensionality "d" of the problem be the sum of "t" size (number of
        targets per pattern) and "f" size (number of features per pattern).
        So, d = t + f

        The data will be set as Pattern = (Target | Features).

        If database has *N* patterns, its size follows *Nxd*.


        Note:
            [1] Paper reference: Huang, 2012, "Extreme Learning Machine for
            Regression and  Multiclass Classification"

        Attributes:
            input_weight (numpy.ndarray): a random matrix (*Lxd-1*) needed
                to calculate H(**x**).
            output_weight (numpy.ndarray): a column vector (*Nx1*) calculated
                after training, represent :math:\\beta.
            bias_of_hidden_neurons (numpy.ndarray): a random column vector
                (*Lx1*) needed to calculate H(**x**).
            param_function (str): function that will be used for training.
            param_c (float): regularization coefficient (*C*) used for training.
            param_l (list of float): number of neurons that will be used for
                training.
            param_opt (bool): a boolean used to calculate an optimization
                when number of training patterns are much larger than neurons
                (N >> L).

        Other Parameters:
            regressor_name (str): The name of classifier/regressor.
            available_functions (list of str): List with all available
                functions.
            default_param_function (str): Default function if not set at
                class constructor.
            default_param_c (float): Default parameter c value if not set at
                class constructor.
            default_param_l (integer): Default number of neurons if not set at
                class constructor.
            default_param_opt (bool): Default boolean optimization flag.

        Note:
            * **regressor_name**: defaults to "elmr".
            * **default_param_function**: defaults to "sigmoid".
            * **default_param_c**: defaults to 2 ** -6.
            * **default_param_l**: defaults to 500.
            * **default_param_opt**: defaults to False.

    """

    def __init__(self, params=[]):
        """
            Class constructor.

            Arguments:
                params (list): first argument (*str*) is an available function,
                    second argument (*float*) is the coefficient *C* of
                    regularization, the third is the number of hidden neurons
                    and the last argument is an optimization boolean.

            Example:

                >>> import elm
                >>> params = ["sigmoid", 1, 500, False]
                >>> elmr = elm.ELMRandom(params)

        """
        super(self.__class__, self).__init__()

        self.available_functions = ["sigmoid", "multiquadric"]

        self.regressor_name = "elmr"

        self.default_param_function = "sigmoid"
        self.default_param_c = 2 ** -6
        self.default_param_l = 500
        self.default_param_opt = False

        self.input_weight = []
        self.output_weight = []
        self.bias_of_hidden_neurons = []

        # Initialized parameters values
        if not params:
            self.param_function = self.default_param_function
            self.param_c = self.default_param_c
            self.param_l = self.default_param_l
            self.param_opt = self.default_param_opt
        else:
            self.param_function = params[0]
            self.param_c = params[1]
            self.param_l = params[2]
            self.param_opt = params[3]

    # ########################
    # Private Methods
    # ########################

    def __set_random_weights(self, number_of_hidden_nodes,
                             number_of_attributes):
        """
            Initialize random values to calculate function

            Arguments:
                number_hidden_nodes (int): number of neurons.
                number_of_attributes (int): number of features.

        """

        self.input_weight = np.random.rand(number_of_hidden_nodes,
                                           number_of_attributes) * 2 - 1

        self.bias_of_hidden_neurons = np.random.rand(number_of_hidden_nodes, 1)

    def __map_hidden_layer(self, function_type, number_hidden_nodes, data):
        """
            Map argument "data" to the hidden layer feature space.

            Arguments:
                function_type (str): function to map input data to feature
                    space.
                number_hidden_nodes (int): number of hidden neurons.
                data (numpy.ndarray): data to be mapped to feature space.

            Returns:
                numpy.ndarray: mapped data.

        """

        number_of_data = data.shape[0]

        if function_type == "sigmoid" or function_type == "sig" or \
            function_type == "sin" or function_type == "sine" or \
            function_type == "hardlim" or \
                function_type == "tribas":

            temp = np.dot(self.input_weight, data.conj().T)
            bias_matrix = np.tile(self.bias_of_hidden_neurons,
                                  number_of_data)
            temp = temp + bias_matrix

        elif function_type == "mtquadric" or function_type == "multiquadric":
            temph1 = np.tile(np.sum(data ** 2, axis=1).reshape(-1, 1),
                             number_hidden_nodes)

            temph2 = \
                np.tile(np.sum(self.input_weight ** 2, axis=1).reshape(-1, 1),
                        number_of_data)

            temp = temph1 + temph2.conj().T \
                   - 2 * np.dot(data, self.input_weight.conj().T)

            temp = temp.conj().T + \
                   np.tile(self.bias_of_hidden_neurons ** 2, number_of_data)

        elif function_type == "gaussian" or function_type == "rbf":
            temph1 = np.tile(np.sum(data ** 2, axis=1).reshape(-1, 1),
                             number_hidden_nodes)

            temph2 = \
                np.tile(np.sum(self.input_weight ** 2, axis=1).reshape(-1, 1),
                        number_of_data)

            temp = temph1 + temph2.conj().T \
                - 2 * np.dot(data, self.input_weight.conj().T)

            temp = \
                np.multiply(temp.conj().T, np.tile(self.bias_of_hidden_neurons,
                                                   number_of_data))
        else:
            print("Error: Invalid function type")
            return

        if function_type == "sigmoid" or function_type == "sig":
            if _SCIPY:
                h_matrix = expit(temp)
            else:
                h_matrix = 1 / (1 + np.exp(-temp))
        elif function_type == "sine" or function_type == "sin":
            h_matrix = np.sin(temp)
        elif function_type == "mtquadric" or function_type == "multiquadric":
            h_matrix = np.sqrt(temp)
        elif function_type == "gaussian" or function_type == "rbf":
            h_matrix = np.exp(temp)
        else:
            print("Error: Invalid function type")
            return

        return h_matrix

    def _local_train(self, training_patterns, training_expected_targets,
                     params):

        # If params not provided, uses initialized parameters values
        if not params:
            pass
        else:
            self.param_function = params[0]
            self.param_c = params[1]
            self.param_l = params[2]
            self.param_opt = params[3]

        number_of_attributes = training_patterns.shape[1]

        self.__set_random_weights(self.param_l, number_of_attributes)

        h_train = self.__map_hidden_layer(self.param_function, self.param_l,
                                          training_patterns)

        # If N >>> L, param_opt should be True
        if self.param_opt:
            self.output_weight = np.linalg.solve(
                (np.eye(h_train.shape[0]) / self.param_c) +
                np.dot(h_train, h_train.conj().T),
                np.dot(h_train, training_expected_targets))

        else:
            self.output_weight = np.dot(h_train, np.linalg.solve(
                ((np.eye(h_train.shape[1]) / self.param_c) + np.dot(
                    h_train.conj().T, h_train)),
                training_expected_targets))

        training_predicted_targets = np.dot(h_train.conj().T,
                                            self.output_weight)

        return training_predicted_targets

    def _local_test(self, testing_patterns, testing_expected_targets,
                    predicting):

        h_test = self.__map_hidden_layer(self.param_function, self.param_l,
                                         testing_patterns)

        testing_predicted_targets = np.dot(h_test.conj().T, self.output_weight)

        return testing_predicted_targets

    # ########################
    # Public Methods
    # ########################

    def search_param(self, database, dataprocess=None, path_filename=("", ""),
                     save=False, cv="ts", of="rmse", f=None, eval=50):
        """
            Search best hyperparameters for classifier/regressor based on
            optunity algorithms.

            Arguments:
                database (numpy.ndarray): a matrix containing all patterns
                    that will be used for training/testing at some
                    cross-validation method.
                dataprocess (DataProcess): an object that will pre-process
                    database before training. Defaults to None.
                path_filename (tuple): *TODO*.
                save (bool): *TODO*.
                cv (str): Cross-validation method. Defaults to "ts".
                of (str): Objective function to be minimized at
                    optunity.minimize. Defaults to "rmse".
                f (list of str): a list of functions to be used by the
                    search. Defaults to None, this set all available
                    functions.
                eval (int): Number of steps (evaluations) to optunity algorithm.

            Each set of hyperparameters will perform a cross-validation
            method chosen by param cv.

            Available *cv* methods:
                - "ts" :func:`mltools.time_series_cross_validation()`
                    Perform a time-series cross-validation suggested by Hydman.

                - "kfold" :func:`mltools.kfold_cross_validation()`
                    Perform a k-fold cross-validation.

            Available *of* function:
                - "accuracy", "rmse", "mape", "me".


            See Also:
                http://optunity.readthedocs.org/en/latest/user/index.html
        """

        if f is None:
            search_functions = self.available_functions
        elif type(f) is list:
            search_functions = f
        else:
            raise Exception("Invalid format for argument 'f'.")

        print(self.regressor_name)
        print("##### Start search #####")

        config = configparser.ConfigParser()
        if sys.version_info < (3, 0):
            config.readfp(open(_ELMR_CONFIG))
        else:
            config.read_file(open(_ELMR_CONFIG))

        best_function_error = 99999.9
        temp_error = best_function_error
        best_param_function = ""
        best_param_c = 0
        best_param_l = 0
        for function in search_functions:

            if sys.version_info < (3, 0):
                elmr_c_range = ast.literal_eval(config.get("DEFAULT",
                                                           "elmr_c_range"))

                neurons = config.getint("DEFAULT", "elmr_neurons")

            else:
                function_config = config["DEFAULT"]
                elmr_c_range = ast.literal_eval(function_config["elmr_c_range"])
                neurons = ast.literal_eval(function_config["elmr_neurons"])

            param_ranges = [[elmr_c_range[0][0], elmr_c_range[0][1]]]

            def wrapper_opt(param_c):
                """
                    Wrapper for optunity.
                """

                if cv == "ts":
                    cv_tr_error, cv_te_error = \
                        time_series_cross_validation(self, database,
                                                     params=[function,
                                                             2 ** param_c,
                                                             neurons,
                                                             False],
                                                     number_folds=10,
                                                     dataprocess=dataprocess)

                elif cv == "kfold":
                    cv_tr_error, cv_te_error = \
                        kfold_cross_validation(self, database,
                                               params=[function,
                                                       2 ** param_c,
                                                       neurons,
                                                       False],
                                               number_folds=10,
                                               dataprocess=dataprocess)

                else:
                    raise Exception("Invalid type of cross-validation.")

                if of == "accuracy":
                    util = 1 / cv_te_error.get_accuracy()
                else:
                    util = cv_te_error.get(of)

                # print("c:", param_c, "util: ", util)
                return util

            optimal_pars, details, _ =  \
                optunity.minimize(wrapper_opt,
                                  solver_name="cma-es",
                                  num_evals=eval,
                                  param_c=param_ranges[0])

            # Save best function result
            if details[0] < temp_error:
                temp_error = details[0]

                if of == "accuracy":
                    best_function_error = 1 / temp_error
                else:
                    best_function_error = temp_error

                best_param_function = function
                best_param_c = optimal_pars["param_c"]
                best_param_l = neurons

            if of == "accuracy":
                print("Function: ", function,
                      " best cv value: ", 1/details[0])
            else:
                print("Function: ", function,
                      " best cv value: ", details[0])

        # MLTools Attribute
        self.cv_best_rmse = best_function_error

        # elmr Attribute
        self.param_function = best_param_function
        self.param_c = best_param_c
        self.param_l = best_param_l

        print("##### Search complete #####")
        self.print_parameters()

        return None

    def print_parameters(self):
        """
            Print current parameters.
        """

        print()
        print("Regressor Parameters")
        print()
        print("Regularization coefficient: ", self.param_c)
        print("Function: ", self.param_function)
        print("Hidden Neurons: ", self.param_l)
        print()
        print("CV error: ", self.cv_best_rmse)
        print("")
        print()

    def get_available_functions(self):
        """
            Return available functions.
        """

        return self.available_functions

    def train(self, training_matrix, params=[]):
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

        return self._ml_train(training_matrix, params)

    def test(self, testing_matrix, predicting=False):
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

        return self._ml_test(testing_matrix, predicting)


    @copy_doc_of(MLTools._ml_predict)
    def predict(self, horizon=1):

        return self._ml_predict(horizon)

    @copy_doc_of(MLTools._ml_train_iterative)
    def train_iterative(self, database_matrix, params=[], sliding_window=168,
                        k=1):

        return self._ml_train_iterative(database_matrix, params,
                                        sliding_window, k)


