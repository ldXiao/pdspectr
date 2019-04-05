This package is a pseudospectral PDE solver for some speccific stiff equations.

When solving a PDE in pseudosepctral methods, it is quite common that it can be finally reduced to the form like this

$$
\frac{\partial}{\partial t}\hat{\phi}= A \hat{\phi}+ B(\hat{phi})
$$

where $A$ is a linear stiff part and $B$ is a nonlinear but (hopefully) non-stiff part.

This package is designed to be as compatible with the latest **scipy** as possible.

User can initialize the ODE with the customized class **Linear** and **NonLinear** of differential operators. 

For example
```python
from pdspectr.utils import stiff_solve_ivp
from pdspectr.pseudo_spectral import Linear, NonLinear
from pdspectr.ETDRK2 import ETDRK2

# region for periodic boudary condition
L = 60
T = 10

# initialize a linear operator of form -(d/dx)^3 q(x) phi(x) over L
L1 = Linear(-1, [(3, qsample)], L)

degrees1 = [(1, 2)]
# initialize a Nonlinear operotor of the form  -3 * (d/dx) phi(x)^2
NL1 = NonLinear(-3, degrees1, L)

def A(t, phihat):
    return L1(phihat)

def B(t, phihat):
    return NL1(phihat)

sol = stiff_solve_ivp(A, B, [0, T], fhat, ETDRK2, max_step=dt)
```

 
