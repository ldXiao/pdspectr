from pdspectr.SBDF2 import SBDF2
from pdspectr.ETDRK2 import ETDRK2
from pdspectr.utils import stiff_solve_ivp
from pdspectr.pseudo_spectral import Linear, NonLinear, \
    ifDFT, fDFT, extend_sample
from scipy.integrate import solve_ivp, RK23, RK45

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import animation

import argparse



def animate_1d_wave(sol,xsample, frames=100, interval=200, multiplier=20,filename="test.mp4"):
    """
    :param sol: OdeResult
    :param frames: int
    :param interval: int
    :param filename: string
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim((-30, 30))
    ax.set_ylim((-3, 3))

    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return (line,)

    # animation function. This is called sequentially
    def animate(num):
        yhat = sol.y[:,num * multiplier]
        line.set_data(xsample, ifDFT(yhat))
        return (line,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval, blit=True)

    anim.save(filename)

def sech(x):
    return 1/np.cosh(x)

def soliton(x, t, c):
    # if c > 0:

    return abs(c)/2 * (sech(np.sqrt(abs(c))/2 * (x - c * t)))**2

def soliton_initial(x):
    return soliton(x, 0, 1)

def soliton_dt_initial(x):
    return 0.5 * (pow(1, 2.5)) * np.sinh(0.5 *np.sqrt(1) * x)/ (np.cosh(0.5 * np.sqrt(1) * x) ** 3)

def KdV_collision_animate(L=60, N=128, Integrtor="SBDF2", dt=0.01, c1=2, c2=5, file_name="soliton_collision.mp4"):
    T = L / c2
    t0 = - 0.4 * T
    slt1 = lambda x: soliton(x, t0, c1)
    slt2 = lambda x: soliton(x, t0, c2)


    x = np.arange(-L/2, L/2, L/N)

    fsample = slt1(x) + slt2(x)
    fhat = fDFT(fsample)

    L1 = Linear(-1, [(3, None)], L)

    degrees1 = [(1, 2)]
    degrees2 = [(1, 3)]
    NL1 = NonLinear(-3, degrees1, L)

    NL2 = NonLinear(-0.1, degrees2, L)


    def A(t, phihat):
        # print(t)
        return L1(phihat)

    def B(t, phihat):
        return NL1(phihat)

    fun = lambda t, y: np.multiply(A(t, y), y) + B(t, y)
    dictionary = {"SBDF2":SBDF2, "ETDRK2":ETDRK2}
    if Integrtor in dictionary:
        sol = stiff_solve_ivp(A, B, [0,T], fhat, dictionary[Integrtor], max_step =dt)
    else:
        sol = solve_ivp(fun, [0, T], fhat, Integrtor, max_step=dt, t_eval=np.linspace(0, 10, 500))

    animate_1d_wave(sol, xsample=x, frames=int((T/dt)/10), interval=50, multiplier=10, filename=file_name)

    for i, t in enumerate(sol.t[::]):
        yhat = sol.y[:, i ]
        plt.plot(x, ifDFT(yhat), label="t={}".format(t))
        plt.title("N={}, dt={}".format(N, dt))
    # plt.legend()
    plt.show()

def error_test(N, dt, Integrator="SBDF2"):
    L = 60
    T = L/1.0
    t0= 0
    slt1 = lambda x: soliton(x, t0, 1)
    x = np.arange(-L / 2, L / 2, L / N)

    fsample = slt1(x)

    fhat = fDFT(fsample)

    L1 = Linear(-1, [(3, None)], L)

    degrees1 = [(1, 2)]

    NL1 = NonLinear(-3, degrees1, L)

    def A(t, phihat):
        # print(t)
        return L1(phihat)

    def B(t, phihat):
        return NL1(phihat)

    dictionary = {"SBDF2": SBDF2, "ETDRK2": ETDRK2}
    sol = stiff_solve_ivp(A, B, [0, T/8], fhat, dictionary[Integrator], max_step=dt)
    yhat = sol.y[:, -1]
    sol1 = stiff_solve_ivp(A, B, [0,T/8], fhat, dictionary[Integrator],max_step=dt * 0.5)
    yhat1 = sol1.y[:,-1]
    fnorm = np.linalg.norm(yhat)
    rel_norm = np.linalg.norm(yhat-yhat1)
    return np.real(rel_norm/ fnorm)

def error_exact(N, dt, Integrator="SBDF2", M=1000, norm_type="L2"):
    L = 60
    T = L / 1.0
    t0 = -T/4
    h = L/M
    slt1 = lambda x: soliton(x, t0, 1)
    x = np.arange(-L / 2, L / 2, L / N)
    xsample = np.arange(-L/2,L/2, L/M)
    fsample = slt1(x)
    # fsample2 = slt2(xsample)
    fhat = fDFT(fsample)
    # fhat2 = fDFT(fsample2)

    L1 = Linear(-1, [(3, None)], L)

    degrees1 = [(1, 2)]

    NL1 = NonLinear(-3, degrees1, L)

    def A(t, phihat):
        # print(t)
        return L1(phihat)

    def B(t, phihat):
        return NL1(phihat)

    dictionary = {"SBDF2": SBDF2, "ETDRK2": ETDRK2}
    sol = stiff_solve_ivp(A, B, [0, -t0], fhat, dictionary[Integrator], max_step=dt)
    yhat = sol.y[:, -1]
    t = sol.t[-1]
    slt2 = lambda x: soliton(x, t0+t, 1)
    fsample2 = slt2(xsample)
    ysample = extend_sample(yhat,M)
    norm_dict = {"L2":2, "L1":1, "Linf":np.inf}
    func_norm = np.linalg.norm(fsample2, norm_dict[norm_type])
    err_norm = np.linalg.norm(fsample2-ysample, norm_dict[norm_type])
    return err_norm/func_norm


def ode_err(dts):
    dts = np.array(dts)
    err1 = []
    err2 = []
    for dt in dts:
        err1.append(error_test(256, dt, Integrator="ETDRK2"))
        err2.append(error_test(256, dt, Integrator="SBDF2"))
        print(dt)
    plt.loglog(dts, err1, '-o', label="ETDRK2")
    plt.loglog(dts, err2, '-x', label="SBDF2")
    plt.loglog(dts, dts * 2, '-s', label="order2_base_line")
    plt.xlabel("dt")
    plt.ylabel("rel_error")
    plt.title("vector norm error of $\hat{\phi}$")
    plt.legend()
    plt.show()

def pde_err(dts, N):
    err_SBDF2 = {"L1":[], "L2":[], "Linf":[]}
    err_ETDRK2 = {"L1":[], "L2":[], "Linf":[]}
    dts = np.array(dts)
    for dt in dts:
        print(dt)
        for norm_type in err_SBDF2:
            err_SBDF2[norm_type].append(error_exact(N, dt, "SBDF2", norm_type=norm_type))
            err_ETDRK2[norm_type].append(error_exact(N, dt, "ETDRK2", norm_type=norm_type))

    plt.loglog(dts, 2 * dts, '-o',label="order 2 base_line")
    for norm_type in err_SBDF2:
        plt.loglog(dts, err_SBDF2[norm_type], '-x', label="SBDF "+norm_type)
        plt.loglog(dts, err_ETDRK2[norm_type], '--s', label="ETDRK2 " + norm_type)
    plt.legend()
    plt.xlabel("dt")
    plt.ylabel("rel_err, func_norm")
    plt.show()

def robustness_analysis(pairs, div):
    L = 60
    T = L / 1.0
    t0 = -T/div
    M = 1000
    slt1 = lambda x: soliton(x, t0, 1)
    slt2 = lambda x: soliton(x, 0, 1)

    xsample = np.arange(-L / 2, L / 2, L / M)
    fsample2 = slt2(xsample)
    # fhat2 = fDFT(fsample2)

    L1 = Linear(-1, [(3, None)], L)

    degrees1 = [(1, 2)]

    NL1 = NonLinear(-3, degrees1, L)

    def A(t, phihat):
        # print(t)
        return L1(phihat)

    def B(t, phihat):
        return NL1(phihat)

    dictionary = {"SBDF2": SBDF2, "ETDRK2": ETDRK2}

    plt.plot(xsample, fsample2, label="exact sol")

    for N, dt in pairs:
        print( N, dt)
        x = np.arange(-L / 2, L / 2, L / N)
        fsample=slt1(x)
        fhat = fDFT(fsample)
        sol = stiff_solve_ivp(A, B, [0, -t0], fhat, dictionary["SBDF2"], max_step=dt)
        yhat = sol.y[:, -1]
        ysmample = extend_sample(yhat, M)
        plt.plot(xsample, ysmample, '-',label="SBDF2"+" N={}, dt={}".format(N,dt))
    plt.legend()
    plt.show()

    plt.plot(xsample, fsample2, label="exact sol")
    for N, dt in pairs:
        print( N, dt)
        x = np.arange(-L / 2, L / 2, L / N)
        fsample=slt1(x)
        fhat = fDFT(fsample)
        sol = stiff_solve_ivp(A, B, [0, -t0], fhat, dictionary["ETDRK2"], max_step=dt)
        yhat = sol.y[:, -1]
        ysmample = extend_sample(yhat, M)
        plt.plot(xsample, ysmample, '--',label="ETDRK2" + " N={}, dt={}".format(N, dt))

    plt.legend()
    plt.show()



def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--Integrator", type=str, default="SBDF2")
    argparser.add_argument("--file-name", type=str, default="test.mp4")
    args = argparser.parse_args()

    KdV_collision_animate(Integrtor=args.Integrator, file_name=args.file_name)


if __name__ == "__main__":
    main()
    # ode_err([0.04, 0.02, 0.01, 0.005, 0.002, 0.0011, 0.0005])
    # pde_err([0.015,0.01, 0.008,0.005, 0.004], 256)
    # robustness_analysis([(32, 0.4),(64, 0.33),(128,0.3),(256,0.275)], 10)
