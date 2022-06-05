"""
Interpolation using Thiele continued fraction.
ALgorithm from O. S. Celis, arXiv:2109.10529.
"""
import numpy as np
from numpy.polynomial import Polynomial


def eval_thiele_frac(aa, zz, x):
    #backward evaluation of continued fraction aa1 + (x-zz1)/aa2+...
    aa = np.asarray(aa)
    zz = np.asarray(zz)
    x = np.asarray(x)
    j = np.max(np.argwhere(np.isfinite((aa))))
    res = np.zeros(len(x))
    if j > 0:
        for i in range(j, 0, -1):
            res = (x - zz[i-1]) / (aa[i] + res)
    return aa[0] + res


def thiele_frac_interpolate(xx, ff, tol=5e-15, n=0, force_even=True):
    """
    n: max order to use
    """
    xx = np.asarray(xx)
    ff = np.asarray(ff)
    
    # tb.cpx_plot(xx, ff)
    # plt.show()
    
    N = len(ff)
    if n <= 0 or n > len(xx):
        n = len(xx) #use tol criteria to stop

    if force_even and (n % 2 == 1):
        n -= 1
        
    dtype = (xx[:1] * ff[:1]).dtype
    aa = np.zeros(n, dtype=dtype) * np.nan # TODO: better way than with NaN?
    zz = np.zeros(n, dtype=dtype) * np.nan
    rr = np.zeros(N, dtype=dtype)
    
    for k in range(n): #main loop
        if k == 0: #init
            rr[:] = ff[:] #inverse differences
            # i = np.argmin(np.abs(ff - np.mean(ff)))
            # i = np.argmax(np.abs(ff))
            i = np.argmin(np.abs(ff)) #smallest value
        else:
            i = np.argmax(np.abs(eval_thiele_frac(aa, zz, xx) - ff)) #adaptive choice
            rr = (xx - zz[k-1]) / (rr - aa[k-1]) #inverse differences

        # store cfrac coef
        # if k == 1:
        #     aa[k] = 1.
        # elif (k % 2 == 1):
        #     aa[k] = 0.
        # else:
        #     aa[k] = rr[i]
            
        aa[k] = rr[i]
        zz[k] = xx[i]

        # reduce data
        ff = ff[xx != xx[i]]
        rr = rr[xx != xx[i]]
        xx = xx[xx != xx[i]]

        # plt.plot(xx, eval_thiele_frac(aa, zz, xx), label=f'step {k}')
        # print(np.max(np.abs(eval_thiele_frac(aa, zz, xx) - ff)))

        if force_even and (k % 2 == 0):
            continue
        
        if k < n - 1:
            if np.max(np.abs(eval_thiele_frac(aa, zz, xx) - ff)) < tol * np.max(np.abs(ff)):
                print(f"target precision reached earlier at n={k}")
                break

    return aa[np.isfinite(aa)], zz[np.isfinite(zz)]

def thiele_frac_2_rat(aa, zz):
    # num = Polynomial([aa[-1] - zz[-1], 1.])
    num = Polynomial([aa[-1]])
    denom = 1.

    for i in range(len(aa)-2, -1, -1):
        num_tmp = num.copy()
        num = num * aa[i] + denom * Polynomial([-zz[i], 1.])
        denom = num_tmp.copy()
        
    return num, denom

### tests ###

def test_thiele_frac_interpolate():
    def f(x):
        return np.cos(np.exp(x))

    x = np.linspace(-1, 1, 100)
    y = f(x)
    aa, zz = thiele_frac_interpolate(x, y)

    xx = np.linspace(-1, 1, 1000)

    # plt.plot(x, y, label='ref')
    # plt.plot(xx, eval_thiele_frac(aa, zz, xx), '--', lw=3, label='interp')
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(f(xx), eval_thiele_frac(aa, zz, xx))

test_thiele_frac_interpolate()

def test_thiele_frac_2_rat():
    N, D = thiele_frac_2_rat([1., 2., 3.], [4., 5., 6.])

    # print(N)
    # print(D)

    x = np.linspace(-5, 3, 10)
    np.testing.assert_allclose(N(x) / D(x), eval_thiele_frac([1., 2., 3.], [4., 5., 6.], x))

test_thiele_frac_2_rat()
