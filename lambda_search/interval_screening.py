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

def normalize_data(X=None, y=None, mean_free=True):
    # expect X \in M(EXMS x DIMS)
    # (a) normalize y to have unit norm
    # (b) normalize X such that each feature has unit norm
    print('Normalizing data. X0={0}, X1={1}.'.format(X.shape[0], X.shape[1]))
    if X is not None:
        print('Calculate mean:')
        mX= np.mean(X, axis=0)
        print('Normalize using sklearn:')
        #Y = skl.preprocessing.normalize(X.T, norm='l2').T
        skl.preprocessing.normalize(X, norm='l2', axis=0, copy=False)
        #print np.diag(X.T.dot(X))
        #print np.diag(Y.T.dot(Y))
        if not mean_free:
            X += mX
    #X = skl.preprocessing.normalize(X, norm='l2')
    if y is not None:
        my = np.mean(y)
        y -= my
        #y /= np.sqrt(y.dot(y.T))
        y /= np.linalg.norm(y, ord=2)
        if not mean_free:
            y += my
    # return X \in M(EXMS x DIMS)
    print('Done.')
    return (X, y)

def load_toy_data(exms=100, feats=10000, non_zeros=100, sigma=0.0, corr=0.0, seed = None):
    # data generation, code taken from Adascreen implementation.
    sigma = 0.1
    corr = 0
    non_zeros = 50
    exms = 100
    feats=100
    np.random.seed(13)
    #if seed != None:
    #    np.random.seed(seed)
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


def screen_DPP(X, y, beta, l0, l, existing_bounds, nzIdcs, normX, normy, *args, **kwargs):
    """
    Screening according to Strong rule.
    Args:
    -----
        X: features, should have shape [samples, features]
        y: targets, should have shape [samples, 1]
        beta: vector of Lasso coefficients, should have shape [features, 1]
        normX: array, containing the norm for each feature
        normy: scalar, norm of target vector
        l0: previous value of lambda (=lmax in the first screening iteration)
        l: current value of lambda
        existing_bounds: array, current estimates for which value of lambda a feature's coefficient may become non-zero
        
    Returns:
    --------
        nzIdcs: array, idcs of features that may have non-zero coefficient
        existing_bounds: array, updated estimates for which value of lambda a feature's coefficient may become non-zero
    """
    # loop over features

    for idx, feat in enumerate(X.T):
        x = np.atleast_2d(feat)
        xy = normX[idx] * normy
        # compute lambda for which lhs and rhs of Strong rule are equal
        bound = (l0 * xy) / (l0 - np.abs(x.dot(y - X[:,nzIdcs].dot(beta))) + xy)
        # if bound is smaller than current value of l, update bound
        #print("bound={}, l={}".format(bound, l))
        existing_bounds[idx] = bound
        
    # get idcs of features that cannot be discarded
    nzIdcs = np.where(existing_bounds > l)[0]
    return nzIdcs, existing_bounds


def get_lambda_boundaries_DPP(X, y, beta, lambda_old, existing_bounds):
    """
    X is a assumed to have dimensions [samples, features].
    """

    normY = np.linalg.norm(y, ord = 2)

    lambda_bound = []
    for idx, feature in enumerate(X.T):

        if existing_bounds[idx] == -1.0:

            xy = np.linalg.norm(feature, ord = 2) * normY

            num = lambda_old * xy
            inner = feature.dot(y - np.dot(X, beta))
            denom = lambda_old - np.abs(inner) + xy
            _lambda = num/denom
            lambda_bound.append(_lambda[0])
        else:
            lambda_bound.append(existing_bounds[idx])

    lambda_bound = np.squeeze(lambda_bound)

    return lambda_bound




def screen_EDPP(X, y, beta, l0, l, existing_bounds, normX, *args, **kwargs):
    """
    Screening according to EDPP.
    Args:
    -----
        X: features, should have shape [samples, features]
        y: targets, should have shape [samples, 1]
        beta: vector of Lasso coefficients, should have shape [features, 1]
        normX: array, containing the norm for each feature
        l0: previous value of lambda (=lmax in the first screening iteration)
        l: current value of lambda
        existing_bounds: array, current estimates for which value of lambda a feature's coefficient may become non-zero
        
    Returns:
    --------
        nzIdcs: array, idcs of features that may have non-zero coefficient
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
    theta = (y - X.dot(beta)) / l0
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

        _, lhs, rhs = EDPP_gap(opt_result.x, xT_theta, xT_k1, xT_k2, sqnorm_k1, sqnorm_k2, k1T_k2, norm_x, k1, k2, True)
        gap = rhs - np.abs(lhs)
        # update feature
        if gap > 0:
            # if result is valid, update to new lambda boundary
            existing_bounds[idx] = opt_result
        else:
            # If gap < 0, the result is not valid. An invalid result implies 
            # the feature cannot be discarded for lambda in [0, l0]. However, 
            # the previous estimate, i.e., the one based on l0, of the boundary 
            # for this feature may be within that interval and may this be 
            # invalid, too. Therefore the existing bound must be updated to
            # the max of l0 and the previously computed boundary.
            existing_bounds[idx] = np.max([l0, existing_bounds[idx]])

    nzIdcs = np.where(existing_bounds > l)[0]
    return nzIdcs, existing_bounds



def get_lambda_max(X, y):
    """
    Get minimal lambda for which optimal coefficient vector beta of LASSO problem will be zero
    Args:
    ----
        X: predictors. X.shape is [samples x features]
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


def get_plain_path(path_scale, ub, lb, steps):
    if path_scale=='linear':
        path = np.linspace(ub, lb, steps, endpoint=True)[::-1]
    if path_scale=='log':
        path = np.logspace(np.log10(ub), np.log10(lb), steps, endpoint=True)[::-1]
    return path


def mse(y_pred, y_true):
    """Mean squared error between two np.arrays"""
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)
    return np.mean((y_pred - y_true)**2)



def assert_boundary_validity(screenBounds, lambda_paths, lambda_grid):
    """
    Function to assert that no screening boundaries exceeds the value of lambda for which a coefficient becomes non-zero.
    Args:
    ----
        screenBounds: array containing the screening bound for each feature.
        lambda_paths: 2d-array containing the coefficient vector per lambda. Expeced shape: [mFeature, len_lambda_path]
        lambda_grid: array containing lambda values for which LASSO model was solved.
    """
    for idx, (scrB, path) in enumerate(zip(screenBounds, lambda_paths)):
        empBs = lambda_grid[np.where(path>0)[0]]
        if len(empBs) > 0:
            empB = empBs[-1]
            assert scrB >= empB, "Screening Boundary is smaller than empirical Boundary"+\
                                    " for feature {}:\n{} vs {}".format(idx, scrB, empB)
    print("Screening Boundaries are valid.")
    return



def follow_lambda_path(X, y, lambda_path, screen, debug, *args, **kwargs):
    # start timer
    start_time = timeit.default_timer()
    # precompute feature norms
    normX = np.linalg.norm(X, ord=2, axis=0)
    # pre allocate some variables
    existing_bounds = lambda_path[-1] * np.ones(X.shape[1])
    beta_path = np.zeros([X.shape[1], len(lambda_path)])
    nonZero_path = np.zeros([len(lambda_path)])
    # correction factor to compensate difference in sklearn lasso objective and lasso objective in screening literature.
    sqrt_n = np.sqrt(X.shape[0])
    # initial model fit, result should be a zero vector.
    # No screening is required as by definition all features will be discarted for lambda_max
    model = linear_model.Lasso(alpha=lambda_path[0], fit_intercept=False)
    model.fit(X*sqrt_n, y*sqrt_n)
    assert(np.sum(np.abs(model.coef_)) == 0), "beta for lambda_max should be zero but np.sum(np.abs(beta)) = {}".format(np.sum(np.abs(model.coef_)))
    beta = np.atleast_2d(model.coef_).T
    beta_path[:, 0] = np.squeeze(beta)
    nzIdcs = np.arange(X.shape[1])
    # move along lambda path
    for idx, lmda in enumerate(lambda_path[1:]):
        # screen
        nzIdcs, existing_bounds = screen(X, y, beta, lambda_path[0], lmda, existing_bounds, nzIdcs, normX, *args, **kwargs)
        nonZero_path[idx] = len(nzIdcs)
        # update lambda
        model.alpha = lmda
        # solve lasso using only non-discarted features
        model.fit(X[:, nzIdcs]*sqrt_n, y*sqrt_n)
        beta = np.atleast_2d(model.coef_).T
        beta_path[nzIdcs, 1+idx] = np.squeeze(beta)
        print("iter {}: lambda = {},  nonZeros {}, predicted {}".format(1+idx, lmda, len(np.where(model.coef_ != 0)[0]), len(nzIdcs)))

        if debug:
            model.fit(X*sqrt_n, y*sqrt_n)
            assert set(np.where(model.coef_ != 0)[0]) - set(nzIdcs) == set(), "discarted but nonzero {}".format(set(np.where(model.coef_ != 0)[0]) - set(nzIdcs))

    t = timeit.default_timer() - start_time

    return t, beta_path, nonZero_path





def run(nSamples, mFeatures, nZeros, sigma, corr, valRatio, screenRule_name, gridSize, seed, save, steps=100):

    print("Screening with {}".format(screenRule_name))

    # create data
    X, y, beta_star = load_toy_data(exms=nSamples, feats=mFeatures, non_zeros=mFeatures-nZeros, seed=seed, corr=corr, sigma=sigma)
    (X, y) = normalize_data(X, y, mean_free=True)
    y = np.atleast_2d(y).T

    print("X = {}".format(X))

    # compute maximum meaningful value of lambda
    LAMBDA_MAX, lmax_ind, lmax_x = get_lambda_max(X, y)

    print("X.shape: {}, y.shape: {}".format(X.shape, y.shape))
    print("LAMBDA_MAX = {}, lmax_x = {}".format(LAMBDA_MAX, lmax_x))

    # by def, the solution for lambda_max is beta = 0
    beta = np.atleast_2d(np.zeros(X.shape[1])).T

    lambda_path = get_plain_path("linear", 0, LAMBDA_MAX+1e-9, 100)

    running_time = 0
    if screenRule_name == "dpp":
        screen = screen_DPP
        start_time = timeit.default_timer()
        normy = np.linalg.norm(y, ord=2)
        print(normy)
        t = timeit.default_timer() - start_time
        running_time += t
    elif screenRule_name == "edpp":
        screen = screen_EDPP
        normy = None
    else:
        raise NotImplementedError("{} is not implemented.".format(screenRule_name))

    t, beta_path, nonZero_path = follow_lambda_path(X, y, lambda_path, screen, normy=normy, debug=True)

    #plt.plot(lambda_path, nonZero_path)
    #plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of training samples", default="20", type=int)
    parser.add_argument("-m", help="number of features", default="10", type=int)
    parser.add_argument("-z", help=r"number of features whose coefficient $\beta = 0$", default=0, type=int)
    parser.add_argument("-s", help="std. of noise in generated data", default=0.0, type=float)
    parser.add_argument("-c", help="correlation of features in generated data", default=0.0, type=float)
    parser.add_argument("-r", help="screening rule", default="dpp", choices=["dpp", "edpp"], type=str)
    parser.add_argument("-g", help="number of lambdas for which the model is fit to obtain the \
                                    ground truth solution of (non-)zero betas", default=100, type=int)
    parser.add_argument("-v", help="number of validation samples in relation to training samples", default=1.0, type=float)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save", help="IF True figure is saved, else figure is shown", default=False, action="store_true")
    args = parser.parse_args()

    # Call main routine
    run(args.n, args.m, args.z, args.s, args.c, args.v, args.r, args.g, args.seed, args.save)



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

