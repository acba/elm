# -*- coding: utf-8 -*-

"""
    This file contains DataProcess class and all developed methods.
"""

import numpy as np

try:
    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.decomposition import KernelPCA
    from sklearn import preprocessing as pre

except ImportError:
    _SKLEARN_AVAILABLE = 0
else:
    _SKLEARN_AVAILABLE = 1


class DataProcess:
    """
        Class responsible for pre processing and pos processing data to be
        used for training of regressors/classifiers.


        Attributes:
            default_minmax_param (tuple): asd
            default_pca_n_components (int): asd
            default_kpca_n_components (int): sad
            default_kpca_gamma_param (float): asd

    """

    def __init__(self, scale_method=None, scale_params=None, scale_output=False,
                 transform_method=None, transform_params=None):

        self.default_minmax_param = (-1, 1)

        self.default_ica_param = None

        self.default_pca_n_components = None

        self.default_kpca_n_components = None
        self.default_kpca_gamma_param = 1

        # Check for sklearn libraries
        if (scale_method is not None or transform_method is not None)\
                and not _SKLEARN_AVAILABLE:

            raise Exception("Please install 'scikit-learn' package to perform "
                            "data transformations.")

        # Adjust strings
        if scale_method is not None:
            scale_method = str.lower(scale_method)
        if transform_method is not None:
            transform_method = str.lower(transform_method)

        # check and set input
        self._set_all(scale_method, scale_params, scale_output,
                      transform_method, transform_params)

        # Used for PCA
        self.pca = []

        # Used for kPCA
        self.kpca =[]

        # Used for ICA
        self.ica =[]

        # Used for scaling data
        self.scaler = []

        # Number of features + number of target variables
        self.n_dimension = 0

    def _set_all(self, scale_method, scale_params, scale_output,
                transform_method, transform_params):

        # Check scale arguments
        if scale_method == "minmax":
            if scale_params is None:
                scale_params = self.default_minmax_param
            else:
                if type(scale_params) is not tuple:
                    raise Exception("Error: Minmax parameter must be a 2-tuple.")

        elif scale_method == "standardization":
            if scale_params is not None:
                print("Warning: Standardization don't use any parameter.")

        elif scale_method is not None:
            raise Exception("Error: Scale methods available are 'minmax' and "
                            "'standardization'.")

        # Check transform arguments
        if transform_method == "pca":
            if type(transform_params) is list:
                transform_params = transform_params[0]
            elif transform_params is None:
                transform_params = self.default_pca_n_components
            else:
                raise Exception("Error: PCA parameter is the number of "
                                "components of the new space, can be a integer "
                                "or a list , eg.([2])")

        elif transform_method == "kpca":
            if type(transform_params) is list:
                if len(transform_params) is not 2:
                    raise Exception("Error: kPCA have 2 parameters, eg. "
                                    "[2, 0.1].")

            elif transform_params is None:
                transform_params = [self.default_kpca_n_components,
                                    self.default_kpca_gamma_param]
            else:
                raise Exception("Error: kPCA parameters are the number of "
                                "components of the feature new space, must be "
                                "an integer, and the Kernel RBF parameter "
                                "(gamma).")

        elif transform_method == "ica":
            pass

        elif transform_method is not None:
            raise Exception("Error: Transform methods available are PCA, "
                            "KPCA and ICA.")

        self.scale_method = scale_method
        self.scale_params = scale_params
        self.scale_output = scale_output

        self.transform_method = transform_method
        self.tranform_params = transform_params

    def auto(self, training_matrix, testing_matrix=None):
        """

        """

        tr_matrix, te_matrix = self.scale(training_matrix, testing_matrix)

        tr_matrix, te_matrix = self.transform(tr_matrix, te_matrix)

        return tr_matrix, te_matrix

    def transform(self, training_data, testing_data):

        if self.transform_method == "pca":
            tr_matrix, te_matrix = self.apply_pca(training_data,
                                                  testing_data,
                                                  self.tranform_params)
        elif self.transform_method == "kpca":
            tr_matrix, te_matrix = self.apply_kpca(training_data,
                                                   testing_data,
                                                   self.tranform_params[0],
                                                   self.tranform_params[1])
        elif self.transform_method == "ica":
            pass

        # None case
        else:
            tr_matrix = training_data
            te_matrix = testing_data

        return tr_matrix, te_matrix

    def apply_pca(self, training_data, testing_data, n_components=None):
        """
            Performs a PCA on data, calculate W transformation matrix,
            keep it to later transformations and returns data transformed to
            new space.

            Data should be NxD format, where N is the number of samples and D
            the number of features (dimensions).

            Arguments:

            Returns:

        """

        training_targets = training_data[:, 0].reshape(-1, 1)
        training_patterns = training_data[:, 1:]

        if testing_data is not None:
            testing_targets = testing_data[:, 0].reshape(-1, 1)
            testing_patterns = testing_data[:, 1:]

        self.pca = sklearnPCA(n_components=n_components)
        tr_patterns = self.pca.fit_transform(training_patterns)
        if testing_data is not None:
            te_patterns = self.pca.transform(testing_patterns)

        tr_matrix = np.concatenate([training_targets, tr_patterns], axis=1)
        if testing_data is not None:
            te_matrix = np.concatenate([testing_targets, te_patterns], axis=1)
        else:
            te_matrix = testing_data

        return tr_matrix, te_matrix

    def reverse_pca(self, data):
        """
            Performs a transformation on data to the original feature space.
        """

        if self.pca is not []:
            original_space_data = self.pca.inverse_transform(data)
        else:
            print("Error: Can't reverse if PCA wasn't performed yet.")
            return

        return original_space_data

    def apply_kpca(self, training_data, testing_data, n_components=None,
                   gamma=15):
        """
            Performs a kernel PCA on data, calculate W transformation matrix,
            keep it to later transformations and returns data transformed to
            new space.

            Data should be NxD format, where N is the number of samples and D
            the number of features (dimensions).

            Arguments:

            Returns:

        """

        training_targets = training_data[:, 0].reshape(-1, 1)
        training_patterns = training_data[:, 1:]

        if testing_data is not None:
            testing_targets = testing_data[:, 0].reshape(-1, 1)
            testing_patterns = testing_data[:, 1:]

        # If number of components is None, will set it to be the number of
        # features, won't be a reduction of feature dimension
        if n_components is None:
            n_components = training_patterns.shape[1]

        # print("Kpca number of components: ", n_components)
        self.kpca = KernelPCA(n_components=n_components,
                              kernel='rbf',
                              gamma=gamma)

        tr_patterns = self.kpca.fit_transform(training_patterns)
        if testing_data is not None:
            te_patterns = self.kpca.transform(testing_patterns)

        tr_matrix = np.concatenate([training_targets, tr_patterns], axis=1)
        if testing_data is not None:
            te_matrix = np.concatenate([testing_targets, te_patterns], axis=1)
        else:
            te_matrix = testing_data

        return tr_matrix, te_matrix

    def reverse_kpca(self, data):
        """
            Performs a transformation on data to the original feature space.

            Arguments:
                data -
            Returns:
                transformed data
        """

        if self.kpca is not []:
            original_space_data = self.kpca.inverse_transform(data)
        else:
            print("Error: Can't reverse if kPCA wasn't performed yet.")
            return

        return original_space_data

    def apply_ica(self, data):
        pass

    def reverse_ica(self, data):
        pass

    def scale(self, training_matrix, testing_matrix):
        """

            Arguments:
                asd (asd): asd
            Returns:
                s

        """

        self.n_dimension = training_matrix.shape[1]

        if self.scale_output is True:
            tr_data = training_matrix
            te_data = testing_matrix

        else:
            tr_data = training_matrix[:, 1:]
            training_targets = training_matrix[:, 0].reshape(-1, 1)

            if testing_matrix is not None:
                te_data = testing_matrix[:, 1:]
                testing_targets = testing_matrix[:, 0].reshape(-1, 1)

        if self.scale_method == "minmax":
            self.scaler = pre.\
                MinMaxScaler(feature_range=self.scale_params).fit(tr_data)

            tr_data = self.scaler.transform(tr_data)
            if testing_matrix is not None:
                te_data = self.scaler.transform(te_data)

        elif self.scale_method == "standardization":
            self.scaler = pre.StandardScaler().fit(tr_data)

            tr_data = self.scaler.transform(tr_data)
            if testing_matrix is not None:
                te_data = self.scaler.transform(te_data)

        elif self.scale_method is None:
            tr_matrix = training_matrix
            te_matrix = testing_matrix

        else:
            print("Error: Unavailable scale method.")
            return

        if self.scale_method is not None:
            if self.scale_output is True:
                tr_matrix = tr_data
                if testing_matrix is not None:
                    te_matrix = te_data
            else:
                tr_matrix = np.concatenate([training_targets, tr_data], axis=1)
                if testing_matrix is not None:
                    te_matrix = np.concatenate([testing_targets, te_data],
                                               axis=1)

        if testing_matrix is None:
            te_matrix = testing_matrix

        return tr_matrix, te_matrix

    def reverse_scale_target(self, data):
        """"""

        n_targets = data.size
        tmp = np.hstack((data.reshape(-1, 1),
                         np.empty((n_targets, self.n_dimension-1))))

        if self.scale_method == "minmax" or \
                self.scale_method == "standardization":
            tmp = self.scaler.inverse_transform(tmp)
            rev_data = tmp[:, 0].reshape(data.shape)
        else:
            rev_data = data

        return rev_data