import numpy as np
from scipy import optimize


#this is Armijo rule
#it is set by epsilon, alpha_s and theta.
#we overload () operator for this class
class Armijo:
    def __init__(self, e, alpha_s, theta):
        self.e = e
        self.alpha_s = alpha_s
        self.theta = theta
    def __call__(self, f, fd, x, d):
        alpha = self.alpha_s
        while f(x + alpha * d) > f(x) + self.e * alpha * np.dot(fd(x), d):
            alpha *= self.theta

        return alpha


#this is our method.
# it solve problem only when U = {u | A_ub * u <= b_ub and A_eq * u <= b_eq}
def condgrad(u0, f, df, one_dim_method, eps_stop, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    print(u0)
    bnd = [(None, None)] * u0.size

    while True:
        c = df(u0)

        #here we use simplex method
        u = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bnd).x
        alpha = one_dim_method(f, df, u0, u - u0)
        u1 = u0 + alpha * (u - u0)
        print(u1)

        if alpha * np.linalg.norm(u - u0) < eps_stop:
            break

        u0 = u1
    return u0

#this is our constrains
#this is square { (1,1), (1,-1), (-1,-1), (-1,1)}
A_ub = np.array([[1, 0],
                 [-1, 0],
                 [0, 1],
                 [0, -1]], dtype='float64')
b_ub = np.array([1, 1, 1, 1], dtype='float64')


#this is test function
def f(u):
    x = u[0]
    y = u[1]
    return (x + 4) ** 2 + (y + 6) ** 2

#and this is its gradient
def df(u):
    x = u[0]
    y = u[1]
    return np.array([2 * (x + 4), 2 * (y + 6)])

#set start point
u0 = np.array([0, 0], dtype='float64')

#set paramethers for Armijo rule

armijo = Armijo(0.99, 1, 0.5)

condgrad(f=f, df=df, one_dim_method=armijo, eps_stop=0.0000000001, A_ub=A_ub, b_ub=b_ub, u0=u0)
