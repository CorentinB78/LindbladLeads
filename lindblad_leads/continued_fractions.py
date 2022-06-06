import numpy as np
from scipy import linalg
from copy import copy
from numpy.polynomial import Polynomial


def eval_cont_frac(a_list, b_list, omega, first=0.):
    if len(a_list) == 0:
        return first
    return b_list[0]**2 / (omega - a_list[0] - eval_cont_frac(a_list[1:], b_list[1:], omega, first=first))

def eval_and_gradient(a_list, b_list, omega):
    
    def comp_rec(a_list, b_list):
        if len(a_list) == 0:
            return 0., [], []

        f, grad_a, grad_b = comp_rec(a_list[1:], b_list[1:])

        f = b_list[0]**2 / (omega - a_list[0] - f)

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
        H[i, i+1] = np.real(b_list[i+1])
        H[i+1, i] = H[i, i+1]
        L[i, i+1] = -np.imag(b_list[i+1])
        L[i+1, i] = L[i, i+1]
        
    return b_list[0], H, L

def make_GF_function(H, L):
    """
    Make function w -> [w - H + iL]^-1
    
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
    
    P = Polynomial([b_list[0]**2]) * Q2
    Q = Polynomial([-a_list[0], 1.]) * Q2 - P2
    
    return P, Q


def rat_func_2_cont_frac(num, denom):
    a_list = []
    b_list = []

    while True:
        q = denom // num
        if q.degree() != 1:
            print(q)
            print(num)
            print(denom)
            print(denom % num)
        assert(q.degree() == 1) # TODO: fix this
        a_list.append(-q.coef[0] / q.coef[1])
        b_list.append(np.sqrt(1. / q.coef[1] + 0.j))

        num_tmp = num.copy()
        num = -denom % num / q.coef[1]
        denom = num_tmp

        if num.degree() <= 0:
            if denom.degree() == 1:
                a_list.append(-denom.coef[0] / denom.coef[1])
                b_list.append(np.sqrt(num.coef[0] / denom.coef[1] + 0.j))
            elif denom.degree() == 0:
                a_list[-1] += - num.coef[0] / denom.coef[0]
            else:
                raise RuntimeError
            break

    return np.array(a_list), np.array(b_list)

### tests ###

def test_eval():
    a = [1., 3.]
    b = [2., 4.]
    
    w = np.linspace(-5, 5, 10)
    
    val_ref = 2.**2 / (w - 1. - 4.**2 / (w - 3.))
    np.testing.assert_allclose(eval_cont_frac(a, b, w), val_ref)
    
    ###
    a = [1.j, 3.]
    b = [-2., 4.j]
    
    w = np.linspace(-5, 5, 10)
    
    val_ref = 2.**2 / (w - 1.j + 4.**2 / (w - 3.))
    np.testing.assert_allclose(eval_cont_frac(a, b, w), val_ref)
    
    
test_eval()

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


def test_rat_func_2_cont_frac():
    P, Q = Polynomial([19., -8., 1.]), Polynomial([-29., 29., -9., 1.])
    a_list, b_list = rat_func_2_cont_frac(P, Q)

    np.testing.assert_allclose(a_list, [1., 3., 5.])
    np.testing.assert_allclose(b_list, [1., 1j * np.sqrt(2.), 2.j])
    
    x = np.linspace(-5, 3, 10)
    np.testing.assert_allclose(P(x) / Q(x), eval_cont_frac(a_list, b_list, x))
    
    ###
    P, Q = Polynomial([2.]), Polynomial([3., -5.])
    a_list, b_list = rat_func_2_cont_frac(P, Q)

    np.testing.assert_allclose(a_list, [3. / 5.])
    np.testing.assert_allclose(b_list, [1j * np.sqrt(2. / 5.)])

    x = np.linspace(-5, 3, 10)
    np.testing.assert_allclose(P(x) / Q(x), eval_cont_frac(a_list, b_list, x))
    
test_rat_func_2_cont_frac()
