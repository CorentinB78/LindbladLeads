import numpy as np
from copy import copy
from scipy import optimize
import toolbox as tb
from .continued_fractions import eval_and_gradient


def fit_diagonal_lambda(w, g, nr_sites, start=None, scale=1., min_lambda=None, nr_shots=1, test_gradient=False, verbose=False, **kwargs):
    w = np.asarray(w)
    g = np.asarray(g, dtype=complex)
    N = nr_sites
    if min_lambda is None:
        min_lambda = np.max(np.diff(w))
    
    cost_ref = np.sum((g.imag)**2)
    
    def a_b_from_x(x, d):
        x = np.asarray(x)
        N = len(x) // 3
        a = x[:N] - 1j * np.sqrt(d**2 + x[N:2*N]**2)
        b = x[2*N:3*N]
        return a, b
    
    def x_from_a_b(a, b, d):
        N = len(a)
        x = np.empty(3*N, dtype=float)
        x[:N] = np.real(a)
        x[N:2*N] = np.sqrt(np.imag(a)**2 - d**2)
        x[2*N:3*N] = b
        return x
   
    def cost(params):
        a, b = a_b_from_x(params, min_lambda)
        f, grad_a, grad_b = eval_and_gradient(a, b, w)
        err = np.sum((f.imag - g.imag)**2)
        
        grad_ap = [grad_a[i] * 1j * params[N+i] / a[i].imag for i in range(N)]
        grad = grad_a + grad_ap + grad_b
        grad = [2 * np.sum((f.imag - g.imag) * x.imag) for x in grad]
        
        return err / cost_ref, grad / cost_ref
    
    if test_gradient:
        def check_cost():
            eps = 0.00001
            p = np.array([ 0.8811405 ,  0.9939651 ,  0.66390169,  0.91348942, -1.23472239,
            1.66377897,  0.56383325,  0.15119808,  0.12072565,  0.66766022,
            0.05949701,  0.86887018,  0.80789845,  0.17949964,  0.23326504,
            0.73629257,  0.68129841,  1.75643216,  0.55568887,  0.46688056,
           -1.50449913, -1.21009824, -1.14956364,  0.72275348, -0.15638328,
           -0.66687861, -0.54695415, -0.11980272,  1.12989344, -0.22024956])[:3*N]

            err, grad = cost(p)
            for i in range(len(grad)):
                p2 = copy(p)
                p2[i] += eps
                err2, _ = cost(p2)

                if verbose:
                    print(grad[i], '\t', (err2 - err) / eps)
                np.testing.assert_allclose(grad[i], (err2 - err) / eps, rtol=1e-3)
    
        check_cost()
        return
    
    p_best = None
    error_best = np.inf
    
    iter = tb.progress_bar(range(nr_shots)) if verbose else range(nr_shots)
    
    for k in iter:
        if k > 0 or (start is None):
            p0 = np.random.normal(0., 1., 3 * N) * scale
        else:
            a0, b0 = start
            p0 = x_from_a_b(a0, b0, min_lambda)
        res = optimize.minimize(cost, p0, jac=True, **kwargs)
        
        if verbose > 1:
            print(np.sqrt(res.fun))
        
        if res.fun < error_best:
            p_best = res.x
            error_best = res.fun
    
    a, b = a_b_from_x(p_best, min_lambda)
    
    return a, b, np.sqrt(error_best)