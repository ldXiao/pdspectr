from scipy.integrate import OdeSolver
import numpy as np
import inspect
from scipy.optimize import OptimizeResult

MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}


class OdeResult(OptimizeResult):
    pass

def stiff_solve_ivp(A:callable, B:callable, t_span:list, y0:"numpy array", method, **options):
    """
    :param A: callable
    return a diagonal list of the matrix representing the linear part
    should be of the same shape as y0.
    take parameters A(t, y)
    :param B: callable
    return the value of nonlinear part of the differential equation
    take parameters B(t, y)
    :param t_span: [t0, tf] the evaluateing range in time
    :param y0: initial condition
    :param method: integrating methods should be a sub class of OdeSolver
    any such methods should be initiallized with the callable in the form of A and B
    :return: an numpy array of
    """
    if not (inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError("`method` must be OdeSolver class.")
    t0, tf = float(t_span[0]), float(t_span[1])

    solver = method(A, B, t0, y0, tf, **options)

    ts = [t0]
    ys = [y0]

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y
        ts.append(t)
        ys.append(y)

    message = MESSAGES.get(status, message)
    ts = np.array(ts)
    ys = np.vstack(ys).T

    return OdeResult(t=ts, y=ys)