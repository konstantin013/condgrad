import numpy as np
from scipy import optimize

class Armijo:
    def __init__(self, e, alpha_s, theta):
        self.e = e
        self.alpha_s = alpha_s
        self.theta = theta
    def __call__(self, f, fd, x, d):
        return 0.001




def condgrad(u0, f, df, one_dim_method, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    print(u0)
    bnd = [(None, None)] * u0.size

    for i in range(10000):
        #print('step: ' + str(i))
        c = df(u0)
        #print(c)

        u = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bnd).x
        #print(u)
        alpha = one_dim_method(f, df, u0, u - u0)
        u1 = u0 + alpha * (u - u0)
        print(u1)
        u0 = u1
    return u0



A_ub = np.array([[1, 0],
                 [-1, 0],
                 [0, 1],
                 [0, -1]], dtype='float64')
b_ub = np.array([1, 1, 1, 1], dtype='float64')





def f(u):
    x = u[0]
    y = u[1]
    return (x + 0.3) ** 2 + (y - 0.7) ** 2

def df(u):
    x = u[0]
    y = u[1]
    return np.array([2 * (x + 0.3), 2 * (y - 0.7)])
u0 = np.array([0, 0], dtype='float64')

armijo = Armijo(0.3, 0.6, 0.9)

condgrad(f=f, df=df, one_dim_method=armijo, A_ub=A_ub, b_ub=b_ub, u0=u0)
