# simple implementation of EM algorithm for GMM with univariate Gaussians
# at this point not numerically stable
import numpy as np
from numpy.random import  normal, randint, gamma
from scipy.stats import norm


def log_lik_helper(params, X):
    class_weights = params[:K]
    normal_params = params[K:].reshape((K,2))
    new_X = X.reshape((X.shape[0], 1)).repeat(K, 1)
    for k in range(K):
        new_X[:, k] = norm.pdf(new_X[:, k], loc = normal_params[k, 0], scale = normal_params[k, 1] ** .5) * class_weights[k]
    return new_X
    
def log_lik(params, X):
    # compute log of the likelihood function for mixture of Gaussians
    return np.log(log_lik_helper(params, X).sum(1)).sum()
 
def estep(params, X):
    # Compute new class weights
    f_ij = log_lik_helper(params, X)
    w_ij = f_ij / f_ij.sum(1).reshape((f_ij.shape[0], 1))
    return w_ij.sum(0)/w_ij.sum()
    
def mstep(params, X):
    # Compute new Gaussian parameters
    f_ij = log_lik_helper(params, X)
    w_ij = f_ij / f_ij.sum(1).reshape((f_ij.shape[0], 1))
    mu = (w_ij * X.reshape((X.shape[0], 1))).sum(0) / w_ij[:, 0].sum()
    sigma = np.sqrt(np.multiply((X.reshape((X.shape[0], 1)).repeat(K, 1) - mu) ** 2, w_ij).sum(0)  / w_ij.sum(0))
    return np.array([mu, sigma]).reshape((K * 2, 1), order = 'F').flatten()

if __name__ == '__main__':
    ### create a mixture of normals from K distributions with N variables from each, so true class weights are 1 / K
    K = 3
    N = 1e4
    mu = randint(-10, 10, K)
    sigma = gamma(.2, 1, K)
    ### generate data from which to recover parameters
    X = np.zeros((N, K))
    for k in range(K):
        X[:, k] = normal(mu[k], sigma[k], N)
    X = X.flatten()
    
    #initialize parameters of form [K class weights, interchange K mean and variance parameters]
    #results very sensitive here. hand coding initial values may be more successful
    init_class_weights = np.array([float(randint(1, 10,1)) for k in range(K)])
    mu_guess = randint(-10, 10, K)
    sigma_guess = gamma(.5, 1, K)
    params = np.hstack((init_class_weights / sum(init_class_weights).flatten(), 
                       np.array([mu_guess, sigma_guess]).reshape((K * 2, 1), order = 'F').flatten()))
    
    ll_old = np.inf
    ll_new = -np.inf
    
    while abs(ll_old - ll_new) > 1e-10:
        ll_old = ll_new
        params[:K] = estep(params, X)
        params[K:] = mstep(params, X)
        ll_new = log_lik(params, X)
        print "log likelihood: " + str(ll_new)
      
    print "estimated class probabilities:\t", sorted(params[:K])
    print "estimated means:\t", sorted(params[K:].reshape((K,2))[:,0])
    print "true means:\t", sorted(mu)
    print "estimated sigmas:\t", sorted(params[K:].reshape((K,2))[:,1])
    print "true sigmas:\t", sorted(sigma)