import numpy as np
from scipy import optimize


def condgrad(f, df, A_ub, b_ub, A_eq, b_eq, u0):

