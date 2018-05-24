#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:24:10 2018

@author: Soeren Becker
"""

import numpy as np
import scipy.spatial.distance
import scipy.optimize as opt
import matplotlib.pyplot as plt
import argparse
import timeit
from sklearn import linear_model
import sklearn as skl


def load_toy_data(exms=100, feats=10, non_zeros=5, sigma=0.0, corr=0.0, seed = None):
    # data generation, code taken from Adascreen implementation.
    if seed != None:
        np.random.seed(seed)
    # Generate data similar as done in the Sasvi paper
    X = np.random.uniform(low=0.,high=+1., size=(exms, feats))
    for i in range(1,feats):
        X[:,i] = (1.0-corr)*X[:,i] + corr*X[:,i-1]
    # (exms x features)
    beta_star = np.random.uniform(low=-1., high=+1., size=feats)
    cut = feats-non_zeros
    inds = np.random.permutation(range(feats))
    beta_star[inds[:cut]] = 0.0
    y = X.dot(beta_star) + sigma*np.random.rand(exms)
    return X, y, beta_star


def screen_DPP(X, y, beta, l0, l, normX, normy, existing_bounds, nzIdcs):
    """
    Screening according to Strong rule.
    Args:
    -----
        X: features, should have shape [samples, features]
        y: targets, should have shape [samples, 1]
        beta: vector of Lasso coefficients, should have shape [features, 1]
        l0: previous value of lambda (=lmax in the first screening iteration)
        l: current value of lambda
        normX: array, containing the norm for each feature
        normy: scalar, norm of target vector
        existing_bounds: array, current estimates for which value of lambda a feature's coefficient may become non-zero
        nzIdcs: boolean array of shape (feature,) indicating whether a feature is in the active set according to previous screening

    Returns:
    --------
        nzIdcs: boolean array, indicating whether a feature is in the active set
        existing_bounds: array, updated estimates for which value of lambda a feature's coefficient may become non-zero
    """
    # loop over features

    X_beta = X[:,nzIdcs].dot(beta[nzIdcs])

    for idx, feat in enumerate(X.T):
        x = np.atleast_2d(feat)
        xy = normX[idx] * normy
        # compute lambda for which lhs and rhs of DPP rule are equal
        bound = (l0 * xy) / (l0 - np.abs(x.dot(y - X_beta)) + xy)
        # if bound is smaller than current value of l, update bound
        existing_bounds[idx] = bound
        
    # get idcs of features that cannot be discarded
    nzIdcs = l < existing_bounds 
    return (nzIdcs, existing_bounds)



def screen_EDPP(X, y, beta, l0, l, normX, lmax_x, existing_bounds, nzIdcs):
    """
    Screening according to EDPP.
    Args:
    -----
        X: features, should have shape [samples, features]
        y: targets, should have shape [samples, 1]
        beta: vector of Lasso coefficients, should have shape [features, 1]
        l0: previous value of lambda (=lmax in the first screening iteration)
        l: current value of lambda
        normX: array, containing the norm for each feature
        lmax_x: needed for computation of v1. Should be np.array as returned by get_lambda_max() for first screening iteration, else should be None.
        existing_bounds: array, previous estimates for which value of lambda a feature's coefficient may become non-zero
        nzIdcs: boolean array of shape (features,) indicating whether a feature is in the active set based on previous screening
    Returns:
    --------
        nzIdcs: boolean array, indicating whether a feature is in the active set
        existing_bounds: array, updated estimates for which value of lambda a feature's coefficient may become non-zero
    """
    #----------------------------------------------------------------------------------------------#
    def EDPP_gap(lmda, xT_theta, xT_k1, xT_k2, sqnorm_k1, sqnorm_k2, k1T_k2, norm_x, k1, k2, lhs_rhs = False):
        """
        Cost function for minimization
        """
        LHS = np.squeeze(xT_theta + ((0.5/float(lmda)) * xT_k1) + 0.5*xT_k2)
        RHS = np.squeeze(1 - (0.5 * norm_x * np.sqrt(sqnorm_k1/(lmda**2) +k1T_k2/lmda + sqnorm_k2)))
        gap = RHS - np.abs(LHS)
        if not lhs_rhs:
            if np.abs(LHS) > RHS:
                return np.finfo(np.float32).max
            else:
                return RHS - LHS
        else:
            if np.abs(LHS) > RHS:
                return np.finfo(np.float32).max, LHS, RHS
            else:
                return RHS - LHS, LHS, RHS
    #----------------------------------------------------------------------------------------------#

    # pre-compute feature independent terms
    X_beta = X[:,nzIdcs].dot(beta[nzIdcs])
    theta = (y - X_beta) / l0

    # definition of v1 depends on whether lambda == LAMBDA_MAX, see paper
    if type(lmax_x) == np.ndarray:
        v1 = np.atleast_2d(lmax_x).T
    else:
        v1 = y/l0 - theta

    sqnorm_v1 = np.linalg.norm(v1, ord=2)**2
    # terms k1 and k2 summarizes v1, v2 and v3 with lambda_k factored out
    k1 = y - v1.T.dot(y) * v1/(sqnorm_v1)
    k2 = - theta + (v1.T.dot(theta)/sqnorm_v1) * v1
    sqnorm_k1 = np.linalg.norm(k1, ord=2)**2
    sqnorm_k2 = np.linalg.norm(k2, ord=2)**2
    k1T_k2 = 2*k1.T.dot(k2)
    # compute feature dependent terms that are independent of lambda
    XT_theta = X.T.dot(theta)
    XT_k1 = X.T.dot(k1)
    XT_k2 = X.T.dot(k2)

    # loop over features
    for idx in range(len(existing_bounds)):
        # optimize with respect to lambda
        opt_result = opt.minimize_scalar(fun=EDPP_gap, method = "Bounded", bounds=(0, l0),
                                         args=(XT_theta[idx], XT_k1[idx], XT_k2[idx], sqnorm_k1, sqnorm_k2, k1T_k2, normX[idx], k1, k2, False),
                                         options={'disp': 0, 'maxiter': 500, 'xatol': 1e-09})

        _, lhs, rhs = EDPP_gap(opt_result.x, XT_theta[idx], XT_k1[idx], XT_k2[idx], sqnorm_k1, sqnorm_k2, k1T_k2, normX[idx], k1, k2, True)
        gap = rhs - np.abs(lhs)
        # update feature
        if gap > 0:
            # if result is valid, update to new lambda boundary
            existing_bounds[idx] = opt_result.x
        else:
            # If gap < 0, the result is not valid. An invalid result implies 
            # the feature cannot be discarded for lambda in [0, l]. However, 
            # the previous estimate, i.e., the one based on l0, of the boundary 
            # for this feature may be within that interval and may this be 
            # invalid, too. Therefore the existing bound must be updated to
            # the max of l0 and the previously computed boundary.
            # Hmm, I am actually not sure if this case ever occurs??
            existing_bounds[idx] = np.max([l0, existing_bounds[idx]])

    nzIdcs = l < existing_bounds
    return (nzIdcs, existing_bounds)



def get_lambda_max(X, y):
    """
    Get minimal lambda for which optimal coefficient vector beta of LASSO problem will be zero
    Args:
    ----
        X: predictors. X.shape is [observations x features]
        y: targets.

    Returns:
    -------
        lmax: minimal lambda
        lmax_ind: index of feature corresponding to lmax
        lmax_x: (elementwise absolute value of) feature vector corresponding to lmax
    """
    vals = np.abs(X.T.dot(y))
    lmax_ind = int(vals.argsort(axis = 0)[-1][0])
    lmax = float(vals[lmax_ind])
    lmax_x = X[:, lmax_ind]

    if lmax_x.dot(y)<0.0:
        lmax_x = -lmax_x
    return lmax, lmax_ind, lmax_x


def get_plain_path(scale, lb, ub, steps):
    """
    Helper function to compute lambda_path
    """
    if scale=='linear':
        path = np.linspace(ub, lb, steps, endpoint=True)
    if scale=='log':
        path = np.logspace(np.log10(ub), np.log10(lb), steps, endpoint=True)
    return path


def mse(y_pred, y_true):
    """
    Helper function to compute mean squared error between two np.arrays
    """
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)
    return np.mean((y_pred - y_true)**2)



def assert_boundary_validity(nonZero_path, coef_path):
    """
    Function to check whether all screening bounds are valid, i.e., that there is no coefficient nonzero when screening rule says it is zero.

    Args: results of follow_lambda_path()
    ----
        nonZero_path: 2d-array of shape len(lambda_path) x features, boolean indications whether a feature is nonZero according to screening rule, per lambda
        coef_path: 2d-array of shape len(lambda_path) x features, values of fitted coefficient vector, per lambda

    Returns:
    -------
        returnFlag: boolean flag indicating whether all boundaries are valid or not.
    """
    returnFlag = True
    for idx, (nonZero_indicator, beta) in enumerate(zip(nonZero_path, coef_path)):
        # which indices are zero according to screening?
        idcsShouldBeZero = set(np.where(nonZero_indicator == 0)[0])
        # which coefficients are nonzero?
        idcsAreNotZero = set(np.where(beta != 0)[0])
        # there should be no overlap
        if len(list(idcsAreNotZero.intersection(idcsShouldBeZero))) != 0:
            print("boundary for feature {} is not valid".format(idx))
            returnFlag = False
    return returnFlag


def follow_lambda_path(X, y, Xval, yval, lambda_path, lmax_x, screen, screenRule_name, no_screening=False, screen_always=False):

    """
    Function that performs calls screening and model fitting along a given lambda path.
    
    Args:
    ----
        X: training data (observations x features)
        y: training targets
        Xval: validation data (observations x features)
        yval: validation targets
        lambda_path: array holding the values of lambda at which model shell be fitted
        lmax_x: needed for v1 computation in screen_edpp and is computed in get_lambda_max()
        screen: screening function (currently only screen_dpp() and screen_edpp() are implemented)
        screenRule_name: name of screening algorithm
        no_screening: If True, no screening will be performed
        screen_always: If True, screening will be performed at every step of lambda path. (This has no effect if no_screening==True.)
    
    Returns:
    -------
        nonZero_path: 2d-array of shape len(lambda_path) x features, boolean indications whether a feature is nonZero according to screening rule, per lambda
        coef_path: 2d-array of shape len(lambda_path) x features, values of fitted coefficient vector, per lambda
        screening_positions: position along lambda path at which screening algorithm ran
        err_train: training error
        err_val: validation error
        duration_screening: time needed for screening computations only (i.e. without model fitting)
    """

    # measure time needed for screening alone
    duration_screening = 0
    # arrays to store training and validation error
    err_train = -1 * np.ones_like(lambda_path)
    err_val = -1 * np.ones_like(lambda_path)
    # list to store screening positions
    screening_positions = []
    # precompute norm of feature vectors
    if not no_screening: 
        t = timeit.default_timer()
        normX = np.linalg.norm(X, ord=2, axis=0)
        duration_screening += timeit.default_timer() - t
    # compensation factor because mathematical definition of lasso problem differs between screening literature and scikit learn
    sqrt_n = np.sqrt(X.shape[0])
    # norm of response vector (not needed for edpp)
    if not no_screening and not screenRule_name == "edpp":
        t = timeit.default_timer()
        normy = np.linalg.norm(y, ord=2)
        duration_screening += timeit.default_timer() - t
    # array to store boolean masks indicating which coefficients are non-zero according to screening rule
    nonZero_path = np.zeros([len(lambda_path), X.shape[1]])
    # array to store true coefficient vectors
    coef_path = np.zeros([len(lambda_path), X.shape[1]])

    # walk along given lambda path
    for idx, lmda in enumerate(lambda_path):

        # first iteration, lambda == LAMBDA_MAX
        if idx == 0:
            # set up model
            model = linear_model.Lasso(alpha=lmda, fit_intercept=False)
            model.fit(X*sqrt_n, y*sqrt_n)
            # check that all coefs are zero for lambda == lambda_max
            assert(np.isclose(np.sum(np.abs(model.coef_)), 0)), "beta for lambda_max should be zero but np.sum(np.abs(beta)) = {}".format(np.sum(np.abs(model.coef_)))
            coefs_full = np.zeros([X.shape[1]])
            existing_bounds = lambda_path[0] * np.ones(X.shape[1])
            # boolean mask indicating which coefficients are non zero
            if no_screening:
                nzIdcs = np.array([True]*X.shape[1])
            else:
                nzIdcs = np.array([False]*X.shape[1])
            # next smaller value of lambda for which a new feature will be in the active set
            next_lmda_bound = lmda
            # last value of lambda used for screening
            l0 = lmda
            # model prediction
            y_pred_train = model.predict(X*np.sqrt(X.shape[0]))
            y_pred_val = model.predict(Xval*np.sqrt(Xval.shape[0]))


        else:
            # subsequent iterations
            if not no_screening and (screen_always or lmda < next_lmda_bound):
                # store position where screening was performed
                screening_positions.append(lmda)
                # screen
                t = timeit.default_timer()
                if screenRule_name == "edpp":
                    # special case because edpp needs different arguments
                    if idx == 1:
                        v1 = lmax_x
                    else:
                        v1 = None
                    nzIdcs, existing_bounds = screen(X, y, np.atleast_2d(coefs_full).T, l0, lmda, normX, v1, existing_bounds, nzIdcs)
                else:
                    nzIdcs, existing_bounds = screen(X, y, np.atleast_2d(coefs_full).T, l0, lmda, normX, normy, existing_bounds, nzIdcs)
                duration_screening += timeit.default_timer() - t
                # compute next smaller value of lambda for which a new feature will be in the active set
                bound_candidates = existing_bounds[np.where(existing_bounds<lmda)[0]]
                if len(bound_candidates) > 0:
                    next_lmda_bound = np.max(bound_candidates)
                
            # solve lasso problem with current value of lambda
            model.alpha = lmda
            # fit model using active set only
            model.fit(X[:,nzIdcs]*sqrt_n, y*sqrt_n)
            # keep record of entire coefficient vector
            coefs_full = np.zeros([X.shape[1]])
            # store coefficients
            coefs_full[nzIdcs] = model.coef_
            # what are the weights of each feature?
            coef_path[idx] = coefs_full
            # which features are active? (boolean vector)
            nonZero_path[idx] = existing_bounds > lmda
            # update lambda_0 as used by screening rules
            l0 = lmda

            # compute training
            y_pred_train = model.predict(X[:,nzIdcs]*np.sqrt(X.shape[0]))
            y_pred_val = model.predict(Xval[:,nzIdcs]*np.sqrt(Xval.shape[0]))

        # compute training error
        err_train[idx] = mse(y_pred_train, y*np.sqrt(y.shape[0]))
        # compute validation error
        err_val[idx] = mse(y_pred_val, yval*np.sqrt(yval.shape[0]))

    return nonZero_path, coef_path, screening_positions, err_train, err_val, duration_screening




def run(nSamples, mFeatures, n_nonZeros, sigma, corr, valRatio, screenRule_name, gridSize, seed, save, no_screening, screen_always):

    """
    Main routine: generates data, calls function for screening and model fitting, visualizes results

    Args:
    ----
        nSamples: number of observations in generated training data
        mFeatures: number of features in generated training data
        n_nonZeros: number of coefficients which are truely non-zero
        sigma: std. of noise during data generation
        corr: correlation of features in generated data
        valRatio: number of validation data observations relative to training data. For valRatio == 1, both have same number of observations
        screenRule_name: name of screening rule to use
        gridSize: number of steps in lambda path
        seed: seed for random data generation
        save: If True, visualization is saved, elso visualization is shown
        no_screening: If True, no screening will be performed
        screen_always: If True, screening will be performed at every step of lambda path. (This has no effect if no_screening==True.) 
    """

    print("Screening with {}".format(screenRule_name))
    # generate data
    Xall, yall, beta_star = load_toy_data(exms=nSamples+int(np.ceil(valRatio*nSamples)), feats=mFeatures, non_zeros=n_nonZeros, seed=seed, corr=corr, sigma=sigma)
    yall = np.atleast_2d(yall).T
    # split into training and validation data
    order = np.random.permutation(Xall.shape[0])
    X = Xall[order < nSamples]
    y = yall[order < nSamples]
    Xval = Xall[order >= nSamples]
    yval = yall[order >= nSamples]


    print("Training data: {}x{} [observations x features]".format(X.shape[0], X.shape[1]))

    # compute maximum meaningful value of lambda
    LAMBDA_MAX, lmax_ind, lmax_x = get_lambda_max(X, y)
    print("LAMBDA_MAX = {}".format(LAMBDA_MAX))
    # by def, the solution for lambda_max is beta = 0
    beta = np.atleast_2d(np.zeros(X.shape[1])).T
    # compute lambda path at which to fit model
    lambda_path = get_plain_path(scale="linear", lb=0, ub=LAMBDA_MAX, steps=gridSize)
    # select screening rule
    if screenRule_name == "dpp":
        screen = screen_DPP
    elif screenRule_name == "edpp":
        screen = screen_EDPP
    else:
        raise NotImplementedError("{} is not implemented.".format(screenRule_name))

    start_time_complete = timeit.default_timer()
    nonZero_path, coef_path, screening_positions, err_train, err_val, duration_screening = follow_lambda_path(X, y, Xval, yval, lambda_path, lmax_x, screen, screenRule_name, no_screening, screen_always)
    duration_complete = timeit.default_timer() - start_time_complete

    print("Duration (overall): {}".format(duration_complete))
    print("Duration (screening): {}".format(duration_screening))
    print("number of screenings: {}".format(len(screening_positions)))

    if assert_boundary_validity(nonZero_path, coef_path):
        print("All boundaries are valid.")


    # visualize
    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))
    # resetting the color cycle to default ensures that corresponding coefficient path and lambda boundaries have the same color
    ax1.set_prop_cycle(None)
    # plot coefficients
    ax1.plot(lambda_path, coef_path, label=r"$\beta_k$")
    # resetting the color cycle to default ensures that corresponding coefficient path and lambda boundaries have the same color
    ax1.set_prop_cycle(None)
    if not no_screening:
        # plot boundaries
        ax1.plot(lambda_path, np.max(np.abs(coef_path)) * nonZero_path, "--", label=r"$\lambda$ boundary")
        # plot screening positions
        ax1.plot(screening_positions, np.zeros_like(screening_positions), "ro", ms=3.0, label="screening position")
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(r"$\beta^{(i)}$")
    ylim = 1.05 * np.max(np.abs(coef_path))
    xlims = [-lambda_path[0]*0.05, lambda_path[0]*1.05]
    ax1.set_xlim(xlims)
    ax1.set_ylim([-ylim, ylim])
    # remove duplicates in legend
    handles, labels = ax1.get_legend_handles_labels()
    labels, idcs = np.unique(labels, return_index=True)
    handles = [handles[i] for i in idcs]
    ax1.legend(handles, labels, loc="best")
    ax1.set_title(screenRule_name)

    ax2.plot(lambda_path, err_train, ".--", label="training error")
    ax2.plot(lambda_path, err_val, ".--", label="validation error")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("Error")
    ax2.set_xlim(xlims)
    ax2.set_ylim([0, np.max([np.max(err_train), np.max(err_val)])])
    ax2.legend()
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of training samples", default=20, type=int)
    parser.add_argument("-m", help="number of features", default=10, type=int)
    parser.add_argument("-z", help=r"number of features whose coefficient $\beta != 0$", default=5, type=int)
    parser.add_argument("-s", help="std. of noise in generated data", default=0.0, type=float)
    parser.add_argument("-c", help="correlation of features in generated data", default=0.0, type=float)
    parser.add_argument("-r", help="screening rule", default="edpp", choices=["dpp", "edpp"], type=str)
    parser.add_argument("-g", help="number of lambdas for which the model is fit to obtain the \
                                    ground truth solution of (non-)zero betas", default=100, type=int)
    parser.add_argument("-v", help="number of validation samples relative to training samples", default=1.0, type=float)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save", help="If True figure is saved, else figure is shown", default=False, action="store_true")
    parser.add_argument("--screen_always", default=False, action="store_true", help="If True, screening is performed \
                                                                                in every iteration, if False, screening is skipped \
                                                                                as long as no new feature will be in the active set.")
    parser.add_argument("--no_screening", default = False, action="store_true", help="If true, no screening will be done.")
    args = parser.parse_args()

    # Call main routine
    run(args.n, args.m, args.z, args.s, args.c, args.v, args.r, args.g, args.seed, args.save, args.no_screening, args.screen_always)



"""
def follow_lambda_path(X, y, lambda_path, normy, lmax_x, screen, debug):
    # start timer
    start_time = timeit.default_timer()
    # precompute feature norms
    normX = np.linalg.norm(X, ord=2, axis=0)
    # pre allocate some variables
    existing_bounds = lambda_path[0] * np.ones(X.shape[1])
    # beta_path stores the coefficient vector for each lambda step
    beta_path = np.zeros([X.shape[1], len(lambda_path)])
    # nonZero_path stores the number of non-zero coeffiients
    nonZero_path = np.zeros([len(lambda_path)])
    # correction factor to compensate difference in sklearn lasso objective and lasso objective in screening literature.
    sqrt_n = np.sqrt(X.shape[0])
    # initial model fit, result should be a zero vector.
    # No screening is required as by definition all features will be discarted for lambda_max
    model = linear_model.Lasso(alpha=lambda_path[0], fit_intercept=False)
    model.fit(X*sqrt_n, y*sqrt_n)
    assert(np.sum(np.abs(model.coef_)) == 0), "beta for lambda_max should be zero but np.sum(np.abs(beta)) = {}".format(np.sum(np.abs(model.coef_)))
    # get current coefficient vector
    beta = np.atleast_2d(model.coef_).T
    # store current coefficient vector
    beta_path[:, 0] = np.squeeze(beta)
    # store current number of nonZero coefficients
    nonZero_path[0] = 0
    # get indices of nonZero coefficnents --> no coefficient is non-zero (i.e. all are zero)
    nzIdcs = np.array([])

    beta_full = np.atleast_2d(np.zeros([X.shape[1], 1]))

    print("bounds", existing_bounds)

    # move along lambda path
    for idx, lmda in enumerate(lambda_path[1:]):
        #coefs_warm_idcs = nzIdcs
        #coefs_warm = np.zeros(X.shape[1])
        #if idx > 0:
        #    coefs_warm[coefs_warm_idcs] = model.coef_
        # screen

        nzIdcs, existing_bounds = screen(X, y, beta_full, lambda_path[0], lmda, existing_bounds, nzIdcs, normX, normy, lmax_x if idx == 0 else None)
        
        print("bounds", existing_bounds)
        print("nzIdcs", nzIdcs)

        nonZero_path[1+idx] = len(nzIdcs)
        # update lambda
        model.alpha = lmda
        #model.coef_ = coefs_warm[nzIdcs]
        # solve lasso using only non-discarted features
        model.fit(X[:, nzIdcs]*sqrt_n, y*sqrt_n)

        beta_full = np.atleast_2d(np.zeros([X.shape[1], 1]))
        beta_full[nzIdcs] = model.coef_

        beta = np.atleast_2d(model.coef_).T
        beta_path[nzIdcs, 1+idx] = np.squeeze(beta)
        print("iter {}: lambda = {},  nonZeros {}, predicted {}".format(1+idx, lmda, len(np.where(model.coef_ != 0)[0]), len(nzIdcs)))

        if debug:
            model.fit(X*sqrt_n, y*sqrt_n)
            assert set(np.where(model.coef_ != 0)[0]) - set(nzIdcs) == set(), "discarted but nonzero {}".format(set(np.where(model.coef_ != 0)[0]) - set(nzIdcs))

    t = timeit.default_timer() - start_time

    return t, beta_path, nonZero_path
"""


"""
def get_lambda_boundaries_EDPP(X, y, beta, lambda_old, existing_bounds, lmax_x=None):
    '''Attempt to obtain EDPP boundaries analytically...but this currently does not work'''
    theta = (y - X.dot(beta)) / lambda_old

    if type(lmax_x) != np.ndarray:
        v1 = y/lambda_old - theta
    else:
        v1 = np.atleast_2d(lmax_x).T
        assert np.all(theta == y/lambda_old)

    # helper variables for intermediate results
    print("v1={}".format(v1))
    sqnorm_v1 = np.linalg.norm(v1, ord = 2)**2
    v1_y_normed_v1 = (v1.T.dot(y)/sqnorm_v1) * v1
    v1_theta_normed_v1 = (v1.T.dot(theta)/sqnorm_v1) * v1

    a = np.linalg.norm(y - v1_y_normed_v1, ord=2)**2
    b = (y - v1_y_normed_v1).T.dot(theta + v1_theta_normed_v1)
    c = np.linalg.norm(theta + v1_theta_normed_v1, ord=2)**2

    lambda_bound = []
    for idx, x in enumerate(X.T):

        # x dependent helper variables
        x_theta = x.T.dot(theta)
        sqnorm_x = np.linalg.norm(x, ord=2)**2
        d = (x.T.dot(y - v1_y_normed_v1))**2
        # LHS >= 0 
        f_pos = 4*(1-x_theta) * x.T.dot(y - v1_y_normed_v1) + 2*(x.T.dot(y - v1_y_normed_v1) * x.T.dot(theta + v1_theta_normed_v1))
        g_pos = 4*(1-x_theta)**2 - 4*(1-x_theta) * x.T.dot(theta - v1_theta_normed_v1) + (x.T.dot(theta + v1_theta_normed_v1))**2
        # LHS < 0
        f_neg = 4*(1+x_theta) * x.T.dot(y - v1_y_normed_v1) + 2*(x.T.dot(y - v1_y_normed_v1) * x.T.dot(theta + v1_theta_normed_v1))
        g_neg = 4*(1+x_theta)**2 - 4*(1+x_theta) * x.T.dot(theta - v1_theta_normed_v1) + (x.T.dot(theta + v1_theta_normed_v1))**2


        s = sqnorm_x * a - d
        t_pos = f_pos - 2*sqnorm_x * b
        u_pos = g_pos - sqnorm_x * c

        t_neg = f_neg - 2*sqnorm_x * b
        u_neg = g_neg - sqnorm_x * c

        print("a", a)
        print("b", b)
        print("c", c)
        print("d", d)
        print("f+", f_pos)
        print("f-", f_neg)
        print("g+", g_pos)
        print("g-", g_neg)
        print("s", s)
        print("t+", t_pos)
        print("t-", t_neg)
        print("u+", u_pos)
        print("u-", u_neg)

        _lambda_pos = np.abs((t_pos - np.sqrt(4*s*u_pos+t_pos**2)) / (2*u_pos))
        _lambda_neg = np.abs((t_neg - np.sqrt(4*s*u_neg+t_neg**2)) / (2*u_neg))

        print("{} _lambda+ = {}".format(idx, _lambda_pos))
        print("{} _lambda- = {}".format(idx, _lambda_neg))

        v2t_pos = (y/_lambda_pos - theta) - (v1.T.dot(y/_lambda_pos - theta) / sqnorm_v1) * v1
        v2t_neg = (y/_lambda_neg - theta) - (v1.T.dot(y/_lambda_neg - theta) / sqnorm_v1) * v1

        LHS_pos = np.abs(x.T.dot(theta) + 0.5* x.T.dot(v2t_pos))
        RHS_pos = 1 - 0.5 * np.linalg.norm(v2t_pos, ord=2) * np.sqrt(sqnorm_x)

        LHS_neg = np.abs(x.T.dot(theta) + 0.5* x.T.dot(v2t_neg))
        RHS_neg = 1 - 0.5 * np.linalg.norm(v2t_neg, ord=2) * np.sqrt(sqnorm_x)

        print("LHS+ = {}".format(LHS_pos))
        print("RHS+ = {}\n".format(RHS_pos))
        print("LHS- = {}".format(LHS_neg))
        print("RHS- = {}\n".format(RHS_neg))


        if existing_bounds[idx] == -1.0:
            if x_theta + 0.5*x.T.dot(v2t_pos) >= 0:
                lambda_bound.append(_lambda_pos)
            else:
                lambda_bound.append(_lambda_pos)
        else:
            lambda_bound.append(existing_bounds[idx])

    lambda_bound = np.squeeze(lambda_bound)

    return lambda_bound
"""

