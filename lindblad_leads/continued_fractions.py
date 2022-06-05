import numpy as np
from scipy import linalg
from copy import copy
from numpy.polynomial import Polynomial


def eval_cont_frac(a_list, b_list, omega, first=0.):
    if len(a_list) == 0:
        return first
    return abs(b_list[0])**2 / (omega - a_list[0] - eval_cont_frac(a_list[1:], b_list[1:], omega, first=first))

def eval_and_gradient(a_list, b_list, omega):
    
    def comp_rec(a_list, b_list):
        if len(a_list) == 0:
            return 0., [], []

        f, grad_a, grad_b = comp_rec(a_list[1:], b_list[1:])

        f = abs(b_list[0])**2 / (omega - a_list[0] - f)

        c = (f / b_list[0])**2

        grad_a.append(c)
        grad_b.append(2 * f / b_list[0])

        for i in range(len(grad_a) - 1):
            grad_a[i] *= c
            grad_b[i] *= c

        return f, grad_a, grad_b
    
    f, grad_a, grad_b = comp_rec(a_list, b_list)
    
    return f, grad_a[::-1], grad_b[::-1]

def unroll_cont_frac(a_list, b_list, omega, value):
    if len(a_list) == 0:
        return value
    return 1. / eval_cont_frac(a_list[::-1], np.append([1.], b_list[:0:-1]), omega, first=b_list[0]**2 / value)
    

def cont_frac_to_matrices(a_list, b_list):
    """
    Returns v, H, L
    """
    N = len(a_list)
    H = np.zeros((N, N), dtype=complex)
    L = np.zeros((N, N), dtype=complex)
    
    for i in range(N):
        H[i, i] = a_list[i].real
        L[i, i] = -a_list[i].imag
        
    for i in range(N-1):
        H[i, i+1] = b_list[i]
        H[i+1, i] = np.conj(b_list[i])
        
    return b_list[0], H, L

def make_GF_function(H, L):
    """
    Returns function (n)->(n,N,N)
    """
    M = H - 1j * L
    N = len(H)
    
    def G(w):
        return linalg.inv(np.eye(N) * w - M)
    
    return np.vectorize(G, signature="()->(i,j)")


def cont_frac_to_rat_func(a_list, b_list):
    if len(a_list) == 0:
        return Polynomial([0.]), Polynomial([1.])
    
    P2, Q2 = cont_frac_to_rat_func(a_list[1:], b_list[1:])
    
    P = Polynomial([abs(b_list[0])**2]) * Q2
    Q = Polynomial([-a_list[0], 1.]) * Q2 - P2
    
    return P, Q

### tests ###

def test_unroll():
    r = unroll_cont_frac([1., -2.], np.sqrt([1., 3.]), 2.j, 1.)
    r_ref = 2.j + 2. - 3. / (2.j - 2.)
    assert(r == r_ref)
    
test_unroll()
    
def test_eval_and_gradient():
    a = [1. - 1.j, 2. - 2.j, -1.5j]
    b = [3., 4., -2.]
    w = -6.

    f, grad_a, grad_b = eval_and_gradient(a, b, w)

    assert(f == eval_cont_frac(a, b, w))

    eps = 0.0001
    a1 = copy(a)
    a1[0] += eps
    a2 = copy(a)
    a2[1] += eps
    b1 = copy(b)
    b1[0] += eps
    b2 = copy(b)
    b2[1] += eps

    np.testing.assert_allclose(grad_b[0], (eval_cont_frac(a, b1, w) - eval_cont_frac(a, b, w)) / eps, rtol=1e-3)
    np.testing.assert_allclose(grad_b[1], (eval_cont_frac(a, b2, w) - eval_cont_frac(a, b, w)) / eps, rtol=1e-3)
    np.testing.assert_allclose(grad_a[0], (eval_cont_frac(a1, b, w) - eval_cont_frac(a, b, w)) / eps, rtol=1e-3)
    np.testing.assert_allclose(grad_a[1], (eval_cont_frac(a2, b, w) - eval_cont_frac(a, b, w)) / eps, rtol=1e-3)
    
test_eval_and_gradient()

def test_cont_frac_to_rat_func():
    N, D = cont_frac_to_rat_func([1., 2., 3.], [4., 5., 6.])

    x = np.linspace(-5, 3, 10)
    np.testing.assert_allclose(N(x) / D(x), eval_cont_frac([1., 2., 3.], [4., 5., 6.], x))

test_cont_frac_to_rat_func()

