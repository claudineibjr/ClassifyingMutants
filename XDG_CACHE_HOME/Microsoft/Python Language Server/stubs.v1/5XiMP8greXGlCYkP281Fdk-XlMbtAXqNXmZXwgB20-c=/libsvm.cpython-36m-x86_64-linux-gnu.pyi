import builtins as _mod_builtins

LIBSVM_KERNEL_TYPES = _mod_builtins.list()
__builtins__ = {}
__doc__ = "\nBinding for libsvm_skl\n----------------------\n\nThese are the bindings for libsvm_skl, which is a fork of libsvm[1]\nthat adds to libsvm some capabilities, like index of support vectors\nand efficient representation of dense matrices.\n\nThese are low-level routines, but can be used for flexibility or\nperformance reasons. See sklearn.svm for a higher-level API.\n\nLow-level memory management is done in libsvm_helper.c. If we happen\nto run out of memory a MemoryError will be raised. In practice this is\nnot very helpful since hight changes are malloc fails inside svm.cpp,\nwhere no sort of memory checks are done.\n\n[1] https://www.csie.ntu.edu.tw/~cjlin/libsvm/\n\nNotes\n-----\nMaybe we could speed it a bit further by decorating functions with\n@cython.boundscheck(False), but probably it is not worth since all\nwork is done in lisvm_helper.c\nAlso, the signature mode='c' is somewhat superficial, since we already\ncheck that arrays are C-contiguous in svm.py\n\nAuthors\n-------\n2010: Fabian Pedregosa <fabian.pedregosa@inria.fr>\n      Gael Varoquaux <gael.varoquaux@normalesup.org>\n"
__file__ = '/home/claudinei/.local/lib/python3.6/site-packages/sklearn/svm/libsvm.cpython-36m-x86_64-linux-gnu.so'
__name__ = 'sklearn.svm.libsvm'
__package__ = 'sklearn.svm'
__test__ = _mod_builtins.dict()
def cross_validation():
    "\n    Binding of the cross-validation routine (low-level routine)\n\n    Parameters\n    ----------\n\n    X : array-like, dtype=float, size=[n_samples, n_features]\n\n    Y : array, dtype=float, size=[n_samples]\n        target vector\n\n    svm_type : {0, 1, 2, 3, 4}\n        Type of SVM: C SVC, nu SVC, one class, epsilon SVR, nu SVR\n\n    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}\n        Kernel to use in the model: linear, polynomial, RBF, sigmoid\n        or precomputed.\n\n    degree : int\n        Degree of the polynomial kernel (only relevant if kernel is\n        set to polynomial)\n\n    gamma : float\n        Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other\n        kernels. 0.1 by default.\n\n    coef0 : float\n        Independent parameter in poly/sigmoid kernel.\n\n    tol : float\n        Stopping criteria.\n\n    C : float\n        C parameter in C-Support Vector Classification\n\n    nu : float\n\n    cache_size : float\n\n    random_seed : int, optional\n        Seed for the random number generator used for probability estimates.\n        0 by default.\n\n    Returns\n    -------\n    target : array, float\n\n    "
    pass

def decision_function():
    '\n    Predict margin (libsvm name for this is predict_values)\n\n    We have to reconstruct model and parameters to make sure we stay\n    in sync with the python object.\n    '
    pass

def fit():
    "\n    Train the model using libsvm (low-level method)\n\n    Parameters\n    ----------\n    X : array-like, dtype=float64, size=[n_samples, n_features]\n\n    Y : array, dtype=float64, size=[n_samples]\n        target vector\n\n    svm_type : {0, 1, 2, 3, 4}, optional\n        Type of SVM: C_SVC, NuSVC, OneClassSVM, EpsilonSVR or NuSVR\n        respectively. 0 by default.\n\n    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, optional\n        Kernel to use in the model: linear, polynomial, RBF, sigmoid\n        or precomputed. 'rbf' by default.\n\n    degree : int32, optional\n        Degree of the polynomial kernel (only relevant if kernel is\n        set to polynomial), 3 by default.\n\n    gamma : float64, optional\n        Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other\n        kernels. 0.1 by default.\n\n    coef0 : float64, optional\n        Independent parameter in poly/sigmoid kernel. 0 by default.\n\n    tol : float64, optional\n        Numeric stopping criterion (WRITEME). 1e-3 by default.\n\n    C : float64, optional\n        C parameter in C-Support Vector Classification. 1 by default.\n\n    nu : float64, optional\n        0.5 by default.\n\n    epsilon : double, optional\n        0.1 by default.\n\n    class_weight : array, dtype float64, shape (n_classes,), optional\n        np.empty(0) by default.\n\n    sample_weight : array, dtype float64, shape (n_samples,), optional\n        np.empty(0) by default.\n\n    shrinking : int, optional\n        1 by default.\n\n    probability : int, optional\n        0 by default.\n\n    cache_size : float64, optional\n        Cache size for gram matrix columns (in megabytes). 100 by default.\n\n    max_iter : int (-1 for no limit), optional.\n        Stop solver after this many iterations regardless of accuracy\n        (XXX Currently there is no API to know whether this kicked in.)\n        -1 by default.\n\n    random_seed : int, optional\n        Seed for the random number generator used for probability estimates.\n        0 by default.\n\n    Returns\n    -------\n    support : array, shape=[n_support]\n        index of support vectors\n\n    support_vectors : array, shape=[n_support, n_features]\n        support vectors (equivalent to X[support]). Will return an\n        empty array in the case of precomputed kernel.\n\n    n_class_SV : array\n        number of support vectors in each class.\n\n    sv_coef : array\n        coefficients of support vectors in decision function.\n\n    intercept : array\n        intercept in decision function\n\n    probA, probB : array\n        probability estimates, empty array for probability=False\n    "
    pass

def predict():
    "\n    Predict target values of X given a model (low-level method)\n\n    Parameters\n    ----------\n    X : array-like, dtype=float, size=[n_samples, n_features]\n    svm_type : {0, 1, 2, 3, 4}\n        Type of SVM: C SVC, nu SVC, one class, epsilon SVR, nu SVR\n    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}\n        Type of kernel.\n    degree : int\n        Degree of the polynomial kernel.\n    gamma : float\n        Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other\n        kernels. 0.1 by default.\n    coef0 : float\n        Independent parameter in poly/sigmoid kernel.\n\n    Returns\n    -------\n    dec_values : array\n        predicted values.\n    "
    pass

def predict_proba():
    "\n    Predict probabilities\n\n    svm_model stores all parameters needed to predict a given value.\n\n    For speed, all real work is done at the C level in function\n    copy_predict (libsvm_helper.c).\n\n    We have to reconstruct model and parameters to make sure we stay\n    in sync with the python object.\n\n    See sklearn.svm.predict for a complete list of parameters.\n\n    Parameters\n    ----------\n    X : array-like, dtype=float\n    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}\n\n    Returns\n    -------\n    dec_values : array\n        predicted values.\n    "
    pass

def set_verbosity_wrap():
    '\n    Control verbosity of libsvm library\n    '
    pass

