=====
Usage
=====

To use Python Extreme Learning Machine (ELM) in a project::

    import elm

    # download an example dataset from
    # https://github.com/acba/elm/tree/develop/tests/data


    # load dataset
    data = elm.read("iris.data")

    # create a classifier
    elmk = elm.ELMKernel()

    # search for best parameter for this dataset
    # define "kfold" cross-validation method, "accuracy" as a objective function
    # to be optimized and perform 10 searching steps.
    # best parameters will be saved inside 'elmk' object
    elmk.search_param(data, cv="kfold", of="accuracy", eval=10)

    # split data in training and testing sets
    # use 80% of dataset to training and shuffle data before splitting
    tr_set, te_set = elm.split_sets(data, training_percent=.8, perm=True)

    #train and test
    # results are Error objects
    tr_result = elmk.train(tr_set)
    te_result = elmk.test(te_set)

    print(te_result.get_accuracy)
