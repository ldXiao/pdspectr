from scipy.integrate._ivp.common import (validate_max_step, validate_tol,
                                         select_initial_step, norm, warn_extraneous)
from scipy.integrate import OdeSolver
import numpy as np
from pdspectr.pseudo_spectral import ifDFT

def PrevStep2(f:callable,t0:float,y0:"numpy.array", dt:float, type= "MidPoint"):
    """
    second order accurate one step methods to generate the first step
    :param f: callable function to evaluate y' of the form f(y, t)
    :param t0: starting time
    :param y0: ndarray
    :param dt: time step
    :param type: second order one step methods names allowed to be "MidPoint" or "Heun"
    :return: ym1 numpy array
    """
    if type == "MidPoint":
        y_half = y0 - 0.5 * dt * f(t0, y0)
        ym1 = y0 - dt * f (t0- dt * 0.5, y_half)
    elif type == "Heun":
        kstar = f(t0, y0)
        ystar = y0 - dt * kstar
        ym1 = y0 - dt * 0.5 * (f(t0 - dt, ystar) + kstar)

    return ym1


def SBDF2_step(A:"numpy array", t0:float,
               y0:"numpy array", ym1:"numpy array",
               curr_NL:"numpy array",
               prev_NL:"numpy array", h:float):
    """
    Stiff backward differential function of order 2
    :param A: An ndarray representing the diagonal linear part of the differential eq
     which is of form (n , 1)
    :param t0: time at step 0
    :param y0: value at step n
    :param ym1: value at step n-1
    :param curr_NL: current nonlinear contribution
    :param prev_NL: previous nonlinear contribution
    :param h: time step
    :return y1: value at step n+1
    """
    M = np.ones_like(A, float) - 2 * h /3 * A

    # diag(M) y1 = 4/3 y0 -1 /3 ym1 + 2 h /3( 2 NonLinear(y0) - Nonlinear(ym1)
    rhs = 4 /3 * y0 - 1/3 * ym1 + 2 * h /3 * ( 2 * curr_NL - prev_NL)
    y1 = np.multiply(np.reciprocal(M), rhs)
    return y1


class SBDF2(OdeSolver):
    def __init__(self, Linear:"callable", NonLinear:callable, t0, y0, t_bound, max_step=np.inf,
        rtol=1e-3, atol=1e-6, vectorized=False, first_step=None, ** extraneous):
        warn_extraneous(extraneous)
        fun = lambda t, y: np.multiply(Linear(t, y), y) + NonLinear(t,y)
        super(SBDF2, self).__init__(fun, t0, y0, t_bound, vectorized,
                                     support_complex=True)
        self.h = max_step
        self.Linear = Linear
        self.NonLinear = NonLinear
        self.prev_y = PrevStep2(self.fun, t0, y0, self.h)
        self.prev_NL = NonLinear(t0 - self.h, self.prev_y)
        self.curr_NL = NonLinear(t0, y0)

    def _step_impl(self):
        t = self.t
        curr_y = self.y
        prev_y = self.prev_y
        A = self.Linear(t, curr_y)
        # print(A.max())
        curr_NL = self.curr_NL
        prev_NL = self.prev_NL
        h = self.h
        new_y = SBDF2_step(A, t, curr_y, prev_y, curr_NL, prev_NL, h)
        # print(t, ifDFT(new_y - curr_y).max(), ifDFT(prev_NL - curr_NL).max())
        self.t = t + h
        self.t_old = t
        self.prev_y = curr_y
        self.y = new_y
        self.prev_NL = curr_NL
        self.curr_NL = self.NonLinear(t, new_y)
        return True, None




