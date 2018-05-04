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
#from mpltools import color as mplcolor
import argparse
from sklearn import linear_model



def load_toy_data(exms=100, feats=10000, non_zeros=100, sigma=0.0, corr=0.0, seed = None):
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


def get_lambda_boundaries_EDPP2(X, y, beta, lambda_old, existing_bounds, lmax_x):
    """
    Obtain lambda boundaries according to EDPP.
    This is the non-optimized and thus slower reference implementation.
    """

    def EDPP_gap2(lmda, feat, X, y, beta, lambda_old, lmax_x=None):
        """
        Cost function for minimization
        """
        theta = (y - X.dot(beta)) / lambda_old

        if type(lmax_x) != np.ndarray:
            v1 = y/lambda_old - theta
        else:
            v1 = np.atleast_2d(lmax_x).T
            assert np.all(theta == y/lambda_old)

        sqnorm_v1 = np.linalg.norm(v1, ord=2)**2
        v2 = y/lmda - theta
        v2t = v2 - (v1.T.dot(v2)/sqnorm_v1) * v1

        LHS = np.squeeze(feat.T.dot(theta + 0.5*v2t))
        RHS = np.squeeze(1 - 0.5 * np.linalg.norm(v2t, ord=2) * np.linalg.norm(feat, ord=2))

        gap = RHS - np.abs(LHS)

        if np.abs(LHS) > RHS:
            return np.inf
        else:
            return RHS - LHS
    
    bounds = []
    gaps = []
    for idx, feat in enumerate(X.T):
        if existing_bounds[idx] == -1:
            opt_result = opt.minimize_scalar(EDPP_gap2, method = "Bounded",
                                            args=(np.atleast_2d(feat).T, X, y, beta, lambda_old, lmax_x), 
                                            bounds=(0, lambda_old))
            bounds.append(opt_result.x)
            gaps.append(opt_result.fun)
        else:
            bounds.append(existing_bounds[idx])

    return np.squeeze(bounds), np.squeeze(gaps)


def get_lambda_boundaries_EDPP(X, y, beta, lambda_old, existing_bounds, lmax_x):
    """
    Obtain lambda boundaries according to EDPP.
    """

    #----------------------------------------------------------------------------------------------#
    def EDPP_gap(lmda, xT_theta, xT_k1, xT_k2, sqnorm_k1, sqnorm_k2, k1T_k2, norm_x, k1, k2, debug = False):
        """
        Cost function for minimization
        """
        LHS = np.squeeze(xT_theta + ((0.5/float(lmda)) * xT_k1) + 0.5*xT_k2)
        RHS = np.squeeze(1 - (0.5 * norm_x * np.sqrt(sqnorm_k1/(lmda**2) +k1T_k2/lmda + sqnorm_k2)))
        gap = RHS - np.abs(LHS)
        if not debug:
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
    theta = (y - X.dot(beta)) / lambda_old
    if type(lmax_x) == np.ndarray:
        v1 = np.atleast_2d(lmax_x).T
    else:
        v1 = y/lambda_old - theta
    sqnorm_v1 = np.linalg.norm(v1, ord=2)**2
    # terms k1 and k2 summarizes v1, v2 and v3 with lambda_k factored out
    k1 = y - v1.T.dot(y) * v1/(sqnorm_v1)
    k2 = - theta + (v1.T.dot(theta)/sqnorm_v1) * v1
    sqnorm_k1 = np.linalg.norm(k1, ord=2)**2
    sqnorm_k2 = np.linalg.norm(k2, ord=2)**2
    k1T_k2 = 2*k1.T.dot(k2)
    bounds = np.zeros(X.shape[1])
    gaps = np.zeros(X.shape[1])
    # loop over features
    for idx, x in enumerate(X.T):

        # compute feature dependent terms that are independent of lambda
        xT_theta = np.dot(x, theta)
        xT_k1 = np.dot(x, k1)
        xT_k2 = np.dot(x, k2)
        norm_x = np.linalg.norm(x, ord=2)
        # optimize with respect to lambda
        opt_result = opt.minimize_scalar(fun=EDPP_gap, method = "Bounded", bounds=(0, lambda_old),
                                         args=(xT_theta, xT_k1, xT_k2, sqnorm_k1, sqnorm_k2, k1T_k2, norm_x, k1, k2, False),
                                         options={'disp': 0, 'maxiter': 500, 'xatol': 1e-09})

        _, lhs, rhs = EDPP_gap(opt_result.x, xT_theta, xT_k1, xT_k2, sqnorm_k1, sqnorm_k2, k1T_k2, norm_x, k1, k2, True)
        gap = rhs - np.abs(lhs)
        gaps[idx] = gap
        if gap < 0:
            bounds[idx] = existing_bounds[idx]
        else:
            bounds[idx] = np.squeeze(opt_result.x)

    return bounds, gaps

    
def get_lambda_boundaries_SAVE(X, y, beta, lambda_old, existing_bounds):
    """
    Obtain lambda boundaries accoring to SAVE rule
    """
    y = y - X.dot(beta)
    normY = np.linalg.norm(y, ord = 2)
    lambda_bound = []
    for idx, feature in enumerate(X.T):

        if existing_bounds[idx] == -1.0:

            xy = np.linalg.norm(feature, ord = 2) * normY
            num = lambda_old * (np.abs(feature.dot(y))+xy)
            denom = lambda_old + xy
            _lambda = num / denom
            lambda_bound.append(_lambda)

        else:
            lambda_bound.append(existing_bounds[idx])

    lambda_bound = np.squeeze(lambda_bound)

    return lambda_bound



def get_lambda_boundaries_STRONG(X, y, beta, lambda_old, existing_bounds):
    """
    Obtain lambda boundaries according to STRONG rule
    """
    y = y - X.dot(beta)
    lambda_bound = []
    for idx, feature in enumerate(X.T):

        if existing_bounds[idx] == -1.0:

            _lambda = 0.5 * (np.abs(feature.dot(y)) + lambda_old)
            lambda_bound.append(_lambda)

        else:
            lambda_bound.append(existing_bounds[idx])

    lambda_bound = np.squeeze(lambda_bound)

    return lambda_bound


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


def mse(y_pred, y_true):
    """Mean squared error between two np.arrays"""
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)
    return np.mean((y_pred - y_true)**2)


def createFigure(lambda_paths, lambda_grid, screenBounds, beta_star, errTrain, errVal, save=True):
    """
    Creates a figure that show lambda paths and screening boundaries.
    Args:
    ----
        lambda_paths: 2d-array containing optimal LASSO coefficients per lambda. lambda_path.shape is assumed to be [mfeatures x len_lambdaPath].
        lambda_grid: array of lambda values that LASSO was solved for.
        screenBounds: array with mFeature entries, containing the screening boundary per feature.
        beta_star: ground truth coefficient vector.
        errTrain: training error per lambda of lambda_grid.
        errVal: validation error per lambda of lambda_grid.
        save: if True figure is saved, if False figure is shown.
    """
    beta_star = np.atleast_2d(beta_star)
    if beta_star.shape[0] != 1:
        beta_star = beta_star.T
    # helper variable for path plotting
    lambda_Grid = np.atleast_2d(lambda_grid)
    if not lambda_Grid.shape[0] == 1:
        lambda_Grid = lambda_Grid.T
    lambda_Grid = lambda_Grid.repeat(lambda_paths.shape[0], axis = 0)
    # show coefficient path only where it is non-zero
    lambda_Grid = np.ma.masked_where(np.abs(lambda_paths) == 0, lambda_Grid)
    lambda_paths = np.ma.masked_where(np.abs(lambda_paths) == 0, lambda_paths)

    ylim = np.max([np.max(np.abs(lambda_paths)), np.max(np.abs(beta_star))])
    xlims = [-lambda_grid[-1]*0.05, lambda_grid[-1]*1.05]
    fig = plt.figure(figsize=(20,10))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0)) 
    ax1.grid()
    # lambda paths
    ax1.plot(lambda_Grid.T, lambda_paths.T, ".-", label=r"$\beta_k$ path")
    # screening boundaries
    ax1.plot(np.vstack([screenBounds, screenBounds]), np.array([-ylim, ylim]), "--", label=r"$\lambda_k$")
    # ground truth coefficients beta star
    ax1.plot(-0.025*lambda_grid[-1]*np.ones_like(beta_star), beta_star, "*", label=r"$\beta_k^*$")
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(r"$\beta^{(i)}$")
    ax1.set_xlim(xlims)
    ax1.set_ylim([-ylim, ylim])
    handles, labels = ax1.get_legend_handles_labels()
    labels, idcs = np.unique(labels, return_index=True)
    handles = [handles[i] for i in idcs]
    ax1.legend(handles, labels, loc="best")
    
    # training error
    ax2.plot(lambda_grid, errTrain, "-", label="Train Err.")
    # testing error
    ax2.plot(lambda_grid, errVal, "-", label="Val. Err.")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("Error")
    ax2.set_xlim(xlims)
    ax2.set_ylim([0, np.max(errTrain)])
    ax2.grid()
    ax2.legend(loc="best")
    if save:
        plt.savefig("./lambda_path.pdf")
        plt.close()
    else:
        plt.show()
    return


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


def run(nSamples, mFeatures, zNonZero, sigma, corr, valRatio, screenRule_name, gridSize, seed, save):

    # setup the chosen screening rule
    if screenRule_name == "dpp":
        screenRule = get_lambda_boundaries_DPP
    elif screenRule_name == "save":
        screenRule = get_lambda_boundaries_SAVE
    elif screenRule_name == "strong":
        screenRule = get_lambda_boundaries_STRONG
    elif screenRule_name == "edpp":
        screenRule = get_lambda_boundaries_EDPP

    # create data
    Xall, yall, beta_star = load_toy_data(exms=nSamples+int(np.ceil(valRatio*nSamples)), feats=mFeatures, non_zeros=zNonZero, seed=seed, corr=corr, sigma=sigma)
    yall = np.atleast_2d(yall).T
    # split into training and validation data
    order = np.random.permutation(Xall.shape[0])
    X = Xall[order < nSamples]
    y = yall[order < nSamples]
    Xval = Xall[order >= nSamples]
    yval = yall[order >= nSamples]

    # compute maximum meaningful lambda
    lambda_max, lmax_ind, lmax_x = get_lambda_max(X, y)
    LAMBDA_MAX = lambda_max
    
    print("LAMBDA_MAX = {}".format(LAMBDA_MAX))

    # by def, the solution for lambda_max is beta = 0
    beta = np.atleast_2d(np.zeros(X.shape[1])).T

    all_lambda_new = -1 * np.ones([mFeatures])
    all_lambda_new[lmax_ind] = LAMBDA_MAX
    existing_bounds = np.ones(X.shape[1]) * -1.0
    # first boundary is already known by definition
    existing_bounds[lmax_ind] = LAMBDA_MAX
    
    # evaluate LASSO model a number of times to obtain the ground truth
    # boundaries of when betas turn non-zero
    lambda_grid = np.linspace(0, LAMBDA_MAX, gridSize, endpoint = True)
    grid_res = lambda_grid[1] - lambda_grid[0]
    lambda_grid = np.append(lambda_grid, LAMBDA_MAX+grid_res)
    lambda_paths = np.zeros([mFeatures, len(lambda_grid)])

    err_train, err_val = [], []
    for i, lmda in enumerate(lambda_grid):
        if lmda == 0:
            # lasso with lambda=0 is equiv. to OLS. Use LinearReg. object for numerical reasons.
            clf = linear_model.LinearRegression(fit_intercept=False)
        else:
            clf = linear_model.Lasso(alpha=lmda, fit_intercept = False, tol = 1e-9, max_iter=1e6)
        clf.fit(X*np.sqrt(nSamples), y*np.sqrt(nSamples))
        # save solution
        lambda_paths[:, i] = clf.coef_
        # compute error
        err_train.append(mse(clf.predict(X*np.sqrt(X.shape[0])), y*np.sqrt(y.shape[0])))
        err_val.append(mse(clf.predict(Xval*np.sqrt(Xval.shape[0])), yval*np.sqrt(yval.shape[0])))


    
    # compute screening boundaries for all features
    all_gaps = -1 * np.ones([mFeatures, mFeatures])
    idcs = {x for x in np.arange(mFeatures)} - {lmax_ind}
    for step in range(mFeatures - 1):
        # compute next boundary
        if "edpp" in screenRule_name:
            lambda_boundaries, gaps = screenRule(X, y, beta, lambda_max, existing_bounds, lmax_x if step==0 else None)
        else:
            lambda_boundaries = screenRule(X, y, beta, lambda_max, existing_bounds)

        # only update screening bound for betas for which no bound has been found yet
        #idcs = np.where(existing_bounds == -1.0)[0]
        if "edpp" in screenRule_name:
            all_gaps[step] = gaps
        selected = list(idcs)[np.argmax(lambda_boundaries[list(idcs)])]
        lambda_new = np.squeeze(lambda_boundaries[selected])
        existing_bounds[selected] = lambda_new
        all_lambda_new[selected] = lambda_new
        idcs = idcs - {selected}
        # fit new LASSO with new lambda
        clf = linear_model.Lasso(alpha=lambda_new, fit_intercept = False)
        # the LASSO objective in scikit learn differs by a factor of n:
        # 1/(2n) L2-term + lambda * L1-term. 
        clf.fit(X*np.sqrt(X.shape[0]), y*np.sqrt(y.shape[0]))

        # update lambda and beta
        lambda_max = lambda_new
        beta = np.atleast_2d(clf.coef_).T
    
    # visualize results
    assert_boundary_validity(all_lambda_new, lambda_paths, lambda_grid)
    createFigure(lambda_paths, lambda_grid, all_lambda_new, beta_star, err_train, err_val, save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of training samples", default="20", type=int)
    parser.add_argument("-m", help="number of features", default="10", type=int)
    parser.add_argument("-nz", help=r"percentage of features whose coefficient $\beta != 0$", default=0, type=int)
    parser.add_argument("-s", help="std. of noise in generated data", default=0.0, type=float)
    parser.add_argument("-c", help="correlation of features in generated data", default=0.0, type=float)
    parser.add_argument("-r", help="screening rule", default="dpp", choices=["dpp", "save", "strong", "edpp"], type=str)
    parser.add_argument("-g", help="number of lambdas for which the model is fit to obtain the \
                                    ground truth solution of (non-)zero betas", default=100, type=int)
    parser.add_argument("-v", help="number of validation samples in relation to training samples", default=1.0, type=float)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save", help="IF True figure is saved, else figure is shown", default=False, action="store_true")
    args = parser.parse_args()

    # Call main routine
    run(args.n, args.m, args.nz, args.s, args.c, args.v, args.r, args.g, args.seed, args.save)



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

