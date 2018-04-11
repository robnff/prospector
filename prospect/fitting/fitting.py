import numpy as np
from scipy.optimize import minimize, least_squares

from fitting import run_emcee_sampler, run_dynesty
from ..likelihood import lnlike_spec, lnlike_phot

def lnprobfn(theta, model=None, obs=None, sps=None, noisemodel=(None, None),
             residuals=False, nested=False, verbose=False):
    """Given a parameter vector and optionally a dictionary of observational
    ata and a model object, return the ln of the posterior. This requires that
    an sps object (and if using spectra and gaussian processes, a GP object) be
    instantiated.

    :param theta:
        Input parameter vector, ndarray of shape (ndim,)

    :param model:
        bsfh.sedmodel model object, with attributes including ``params``, a
        dictionary of model parameters.  It must also have ``prior_product()``,
        and ``mean_model()`` methods defined.

    :param obs:
        A dictionary of observational data.  The keys should be
          *``wavelength``
          *``spectrum``
          *``unc``
          *``maggies``
          *``maggies_unc``
          *``filters``
          * and optional spectroscopic ``mask`` and ``phot_mask``.

    :returns lnp:
        Ln posterior probability.
    """

    # Calculate prior probability and exit if not within prior
    lnp_prior = model.prior_product(theta, nested=nested)
    if not np.isfinite(lnp_prior):
        return -np.infty

    # Generate mean model
    try:
        t1 = time.time()
        spec, phot, x = model.mean_model(theta, obs, sps=sps)
        d1 = time.time() - t1
    except(ValueError):
        return -np.infty

    # Return chi vectors for least-squares optimization
    # note this does not include priors!
    if residuals:
        chispec = chi_spec(spec, obs)
        chiphot = chi_phot(phot, obs)
        return np.concatenate([chispec, chiphot])
    
    # Noise modeling
    if noisemodel is not None:
        spec_noise, phot_noise = noisemodel
        if spec_noise is not None:
            spec_noise.update(**model.params)
        if phot_noise is not None:
            phot_noise.update(**model.params)
        vectors = {'spec': spec, 'unc': obs['unc'],
                   'sed': model._spec, 'cal': model._speccal,
                   'phot': phot, 'maggies_unc': obs['maggies_unc']}

    # Calculate likelihoods
    t1 = time.time()
    lnp_spec = lnlike_spec(spec, obs=obs, spec_noise=spec_noise, **vectors)
    lnp_phot = lnlike_phot(phot, obs=obs, phot_noise=phot_noise, **vectors)
    d2 = time.time() - t1
    if verbose:
        write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)

    return lnp_prior + lnp_phot + lnp_spec




def fit_model(obs, model, sps, noise, lnprobfn=lnprobfn, **args):

    if args["optimize"]:
        mr, best, time = run_minimize(obs, model, sps, noise,
                                      lnprobfn=lnprobfn, **min_kwargs)
        # set to the best
        model.set_parameters(mr[best].x)

    if args["emcee"]:
        sampler = run_emcee()

        
    if args["dynesty"]:
        result = run_dynesty()


def run_minimize(obs, model, sps, noise, lnprobfn=None,
                 method='lm', **min_opts):

    initial = model.theta.copy()
    
    lsq = ['lm']
    scalar = ['powell']
    opts = min_kwargs
    
    if method in lsq:
        algorithm = least_squares
        residuals = True
        opts["x_scale"] = "jac"
    elif method in scalar:
        algorithm = minimize
        residuals = False

    args = []
    loss = partial(lnprobfn, obs=obs, model=model, sps=sps,
                   noisemodel=noise, residuals=residuals)

    minimizer = minimize_wrapper(loss, algorithm, method, opts)
    qinit = minimizer_ball(initial, size, model)

    if pool is not None:
            M = self.pool.map
    else:
            M = map

    t = time.time()
    results = list(M(minimizer,  [np.array(q) for q in qinit]))
    tm = time.time() - t

    if method in lsq:
        chisq = [np.sum(r.fun**2) for r in results]
        best = np.argmin(chisq)
    elif method in scalar:
        best = np.argmin([p.fun for p in results])

    return results, best, tm


def run_emcee(obs, model, sps, noise, lnprobfn=lnprobfn, hfile=None):

    q = model.theta.copy()
    
    postkwargs = {"obs": obs,
                  "model": model,
                  "sps": sps,
                  "noisemodel"; noise,
                  "nested": False,
                  }

    
    t = time.time()
    out = run_emcee_sampler(lnprobfn, q, model, kwargs=postkwargs,
                            hdf5=hfile, pool=pool,**kwargs)
    sampler, burn_p0, burn_prob0 = out
    ts = time.time() - t

    return sampler


def run_dynesty(obs, model, sps, noise):

    def prior_transform(u, model=None):
        return model.prior_transform(u)

    
    t = time.time()
    dynestyout = fitting.run_dynesty_sampler(lnprobfn, prior_transform, model.ndim,
                                             pool=pool, queue_size=nprocs, 
                                             stop_function=stopping_function,
                                             wt_function=weight_function,
                                             **kwargs)
    ts = time.time() - t
