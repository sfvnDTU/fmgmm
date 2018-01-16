import numpy as np
from scipy import linalg
from sklearn.mixture.base import BaseMixture
from sklearn.mixture.gaussian_mixture import _estimate_log_gaussian_prob, _estimate_gaussian_parameters, \
    _check_means, _check_weights, _check_precisions,_compute_precision_cholesky
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted


class GaussianMixtureWithForwardModel(BaseMixture):
    #TODO: Add tied-diag and tied-spherical
    #TODO: Add zero-mean option
    #TODO: Overload score_samples - currently doesnt work
    """Gaussian Mixture with a fixed forward model.
    Build on top of Scikit-learns Gaussian Mixture
    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Parameters
    ----------
    forward_model: array-like or int
                   if array is specified it should be a linear mapping matrix from subspace (size D)
                   to data space (size P)
                   if int D is specified a PCA is performed on the input and D
                   PCs are retained
    n_components : int, defaults to 1.
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::
            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).
    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
    max_iter : int, defaults to 100.
        The number of EM iterations to perform.
    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.
    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.
    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.
    means_init : array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.
    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.
    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.
    verbose_interval : int, default to 10.
        Number of iteration done before the next print.
    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.
    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.
    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.
    lower_bound_ : float
        Log-likelihood of the best fit of EM.

        Written by: Søren Føns Vind Nielsen (sfvn at dtu dot dk)
        See also GaussianMixture from sklearn
    """

    def __init__(self, forward_model=None, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(GaussianMixtureWithForwardModel, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.forward_model = forward_model
        self.noise = 1.0 # TODO: Add estimation step of noise

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.forward_model is not None:
            self.forward_model = _check_forward_model(self.forward_model, X)
        else:
            self.forward_model = _calc_init_forward_model(X)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features,
                                           self.forward_model)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features,
                                                     self.forward_model)

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', 'precisions_cholesky_'])

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _initialize(self, X, resp):
        # TODO: Initialize A * A.T
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        self.y_sub = self._estimate_subspace_repr(X)

        weights, means, covariances = _estimate_gaussian_parameters(
            self.y_sub, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape

        self.y_sub = self._estimate_subspace_repr(X, lr=log_resp)
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(self.y_sub, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)
        self.noise = self._estimate_noise(X)

    def _estimate_log_prob(self, X):
        # TODO: Smarter solution than repeating n_components times?
        return _estimate_log_gaussian_prob(
            self.y_sub, self.means_, self.precisions_cholesky_, self.covariance_type) \
               + 1/self.n_components*np.repeat(_estimate_log_gaussian_prob_forward_part(X, self.y_sub,
                                                                              self.forward_model, self.noise),
                                                      self.n_components, axis=1)

    def score_samples(self, X):
        """ Calculates predictive likelihood on data X
            - Marginalization over subspace repr and clustering
            ...
        """


    def _estimate_subspace_repr(self, X, lr=None):
        """ Estimates subspace representation y_sub
            Both used in m_step and initialization

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
            lr : array-like, shape (n_samples, n_components)
                Logarithm of the posterior probabilities (or responsibilities) of
                the point of each sample in X.
        """

        if lr is None:
            return np.dot(X, self.forward_model)
        else:
            # TODO: Naive solution (looping over everything)
            n = X.shape[0]
            d = self.forward_model.shape[1]
            y = np.empty((n, d))
            y.fill(np.nan)

            # Loop over observations
            for n in range(y.shape[0]):
                # Linear term part
                lin_term = 1.0/self.noise*np.dot(X[n], self.forward_model)
                for k in range(self.n_components):
                    # TODO: Only works for full matrix
                    if self.covariance_type == 'full':
                        lin_term += np.exp(lr[n,k])*np.dot(self.means_[k,:], linalg.inv(self.covariances_[k,:,:]))
                    else:
                        raise NotImplementedError

                # Square term part
                sq_term = 1.0/self.noise*np.dot(np.transpose(self.forward_model), self.forward_model)
                for k in range(self.n_components):
                    # TODO: Only works for full matrix
                    if self.covariance_type == 'full':
                        sq_term += np.exp(lr[n,k])*linalg.inv(self.covariances_[k,:,:])
                    else:
                        raise NotImplementedError

                # Calc subspace repr
                y[n,] = np.dot(lin_term, linalg.inv(sq_term))

            return y

    def _estimate_noise(self, X):
        """ ... """ # TODO: write nice description in noise estimator
        residual = X - np.dot(self.y_sub, self.forward_model.T)
        return np.sum(residual*residual)/(X.shape[0]*X.shape[1]*np.log(2*np.pi))

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        # TODO: Update how n_parameters is calculated
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)


#######################################################################################################################
#### HELPER FUNCTIONS
#######################################################################################################################

def _estimate_log_gaussian_prob_forward_part(X, sub, forward_model, iso_noise_var):
    res = X - np.dot(sub,np.transpose(forward_model))
    exp_term_pr_samp = 1/iso_noise_var*np.sum(np.multiply(res,res), axis=1)
    vals = -.5 * (X.shape[1] * np.log(2 * np.pi * iso_noise_var) + exp_term_pr_samp)
    return np.expand_dims(vals, axis=1)

def _calc_init_forward_model(X, n_pcacomps = 2):
    """ Calculate forward model in case none is specified in initialization of object
        Initial choice is a PCA with 2 components fitted to the data
    """
    pca = PCA(n_components=n_pcacomps)
    pca.fit(X)
    return np.transpose(pca.components_)

def _check_forward_model(forward_model, X):
    if type(forward_model) == int:
        # Calc forward model using PCA
        forward_model_return = _calc_init_forward_model(X, n_pcacomps=forward_model)
    else:
        if X.shape[1] != forward_model.shape[0]:
            raise ValueError("The specified forward model does not have the same number of rows %i" 
                             "as data has columns %i " %(forward_model.shape[0], X.shape[1]))
        if X.shape[1]< forward_model.shape[1]:
            raise ValueError("The specified forward model maps to a larger space of dimension %i than the data %i"
                             % (forward_model.shape[1], X.shape[1]))
        else:
            forward_model_return = forward_model
    return forward_model_return
