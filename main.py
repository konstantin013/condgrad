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
#this is square { (1,2), (1,2), (1,2), (1,2), (1,2)}



A_ub = np.zeros((10, 5))
b_ub = np.zeros(10)
for i in range(5):
    A_ub[2 * i][i] = 1
    A_ub[2 * i + 1][i] = -1
    b_ub[2 * i] = 2
    b_ub[2 * i + 1] = -1
print(b_ub)


#this is test function

def f(u):
    ans = 100 * (u[1] - 2 * u[0]) ** 2
    ans += 100 * (u[2] - 3 * u[0]) ** 2
    ans += 100 * (u[3] - 4 * u[0]) ** 2
    ans += 100 * (u[4] - 5 * u[0]) ** 2
    ans += (u[0] - 2) ** 2
    return ans

#and this is its gradient

def df(u):
    return np.array([100 * 2 * 2 * (2 * u[0] - u[1]) + 100 * 2 * 3 * (3 * u[0] - u[2]) + 100 * 4 * 2 * (4 * u[0] - u[3]) + 100 * 5 * 2 * (5 * u[0] - u[4]) + 2 * (u[0] - 2),
                     100 * 2 * (u[1] - 2 * u[0]),
                     100 * 2 * (u[2] - 3 * u[0]),
                     100 * 2 * (u[3] - 4 * u[0]),
                     100 * 2 * (u[4] - 5 * u[0])])

#set start point
u0 = np.array([1, 1, 1, 1, 1], dtype='float64')

#set paramethers for Armijo rule

armijo = Armijo(0.99, 1, 0.5)

condgrad(f=f, df=df, one_dim_method=armijo, eps_stop=0.00000001, A_ub=A_ub, b_ub=b_ub, u0=u0)
