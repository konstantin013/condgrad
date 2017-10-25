# Conditional gradient method

## What does it do
this method solve problem
f(u) -> inf, u in U. Where U - insular and bounded.

in k-th step we find u1_k as solution
<f'(u_k), u - u_k> -> min, u in U.

then d_k = u1_k - u_k and u_k+1 = u_k + alpha_k * d_k.

alpha_k we search by Armijo rule. 


This program work in case when U is set as

U = {u | A_ub * u <= b_ub and A_eq * u <= b_eq}
then in each step we can solve problem 
<f'(u_k), u - u_k> -> min, u in U.

by simplex method.

## Usage
1. You should set f and df - its gradient. both take one paramether numpy.array
2. You should set paramethers for Armijo rule.
3. Set initial point
4. set eps_stop paramether. when method make step smaller than eps_stop it will return current point
5. set constrains for U
6. just call condgrad

