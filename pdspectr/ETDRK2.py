from scipy.integrate._ivp.common import (validate_max_step, validate_tol,
                                         select_initial_step, norm, warn_extraneous)
from scipy.integrate import OdeSolver
import numpy as np
from pdspectr.SBDF2 import PrevStep2
from pdspectr.pseudo_spectral import ifDFT

def five_term_taylor1(A, h):
    return 1.0 * h * np.ones_like(A) + 0.5 * pow(h,2) * A + 1/6 * pow(h,3) * pow(A, 2) \
           + 1/24 * pow(h, 4) * pow(A,3) + 1/120 * pow(h,5) * pow(A, 4)

def five_term_taylor2(A,h):
    return 0.5 * pow(h,1) * np.ones_like(A) + 1/6 * pow(h,2) * pow(A, 1) \
           + 1/24 * pow(h, 3) * pow(A,2) + 1/120 * pow(h,4) * pow(A, 3)

def hybrid_approx(A,h, type=1):
    s = abs(A).max()
    if A[len(A)//2]!=0:
        if s > 0.0001:
            if type == 1:
                return np.multiply(np.reciprocal(A),
                                   np.exp(A * h) - np.ones_like(A))
            else:
                return np.multiply(pow(np.reciprocal(A), 2), np.exp(A * h) - np.ones_like(A)- A * h) /h
        else:
            if type == 1:
                # print("called")
                return five_term_taylor1(A,h)
            else:
                # print("called2")
                return five_term_taylor2(A,h)
    else:
        if s > 0.00001:

            if type ==1:
                left = np.multiply(np.reciprocal(A[:len(A)//2]),
                                   np.exp(A[:len(A)//2] * h) - np.ones_like(A[:len(A)//2]))
                middle = np.array([0])

                right = np.multiply(np.reciprocal(A[len(A)//2+1:]),
                                   np.exp(A[len(A)//2+1:]* h) - np.ones_like(A[len(A)//2+1:]))
                return np.hstack([left, middle, right])
            else:
                left = np.multiply(pow(np.reciprocal(A[:len(A)//2]), 2), np.exp(A[:len(A)//2] * h)
                                   - np.ones_like(A[:len(A)//2])- A[:len(A)//2] * h) /h
                middle = np.array([0])
                right = np.multiply(pow(np.reciprocal(A[len(A)//2+1:]), 2), np.exp(A[len(A)//2+1:] * h)
                                   - np.ones_like(A[len(A)//2+1:])- A[len(A)//2+1:] * h) /h
                return np.hstack([left, middle, right])


class ETDRK2(OdeSolver):
    def __init__(self, Linear:"callable", NonLinear:callable, t0, y0, t_bound, max_step=np.inf,
        rtol=1e-3, atol=1e-6, vectorized=False, first_step=None, ** extraneous):
        warn_extraneous(extraneous)
        fun = lambda t, y: np.multiply(Linear(t, y), y) + NonLinear(t,y)
        super(ETDRK2, self).__init__(fun, t0, y0, t_bound, vectorized,
                                     support_complex=True)
        self.h = max_step
        self.Linear = Linear
        self.NonLinear = NonLinear
        self.curr_NL = NonLinear(t0, y0)

    def _step_impl(self):
        t = self.t
        curr_y = self.y
        A = self.Linear(t, curr_y)
        curr_NL = self.curr_NL
        h = self.h
        expA = np.exp(h * A)

        # print((expA - np.multiply(A, hybrid_approx(A,h, type=1)))[0:4])
        predictor = np.multiply(expA, curr_y) + np.multiply(
            hybrid_approx(A, h, type=1),
            curr_NL)
        predictor_NL = self.NonLinear(t + h, predictor)
        # print(t,ifDFT(predictor-curr_y).max(),ifDFT(predictor_NL-curr_NL).max())
        new_y = predictor + np.multiply(
            hybrid_approx(A, h, type=2),
                        predictor_NL-curr_NL)
        # print("newy",t,ifDFT(new_y).max())
        self.y = new_y
        self.t = t + h
        self.t_old = t
        self.curr_NL = self.NonLinear(self.t, new_y)
        return True, None

