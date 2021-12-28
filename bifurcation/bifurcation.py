import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
import sympy as sym
from scipy.integrate import odeint

k1_val = 0.12
k1_inv_val = 0.005
k2_val_numerical = 1.05
k3_val = 0.0032
k3_inv_val = 0.004

def equation(v, t):
    x_num, y_num = v[0], v[1]
    dxdt = k1_val * (1 - x_num - y_num) - k1_inv_val * x_num - k2_val_numerical * x_num * (1 - x_num - y_num) ** 2
    dydt = k3_val * (1 - x_num - y_num) ** 2 - k3_inv_val * y_num ** 2
    dvdt = [dxdt, dydt]
    return dvdt

x = Symbol('x')
y = Symbol('y')
k1 = Symbol('k1')
k1_inv = Symbol('k1_inv')
k2 = Symbol('k2')
k3 = Symbol('k3')
k3_inv = Symbol('k3_inv')

f1 = k1 * (1 - x - y) - k1_inv * x - k2 * x * (1 - x - y) ** 2
f2 = k3 * (1 - x - y) ** 2 - k3_inv * y ** 2
y_expr = solve(f2, y)[1]
f1_tmp = f1.subs(
        {y: y_expr})
k_expr = solve(f1_tmp, k2)[0]

print(k_expr)
# print(y_expr)

a11 = sym.diff(f1, x)
a12 = sym.diff(f1, y)
a21 = sym.diff(f2, x)
a22 = sym.diff(f2, y)

detA = a11 * a22 - a12 * a21
trA = a11 + a22

k1_inv_expr_fold = solve(((a11 + a22).subs({k2: k_expr})).subs({y: y_expr}), k1_inv)[0]
k1_inv_expr_neutral = solve(((a11 * a22 - a12 * a21).subs({k2: k_expr})).subs({y: y_expr}), k1_inv)[0]
# print(k1_inv_expr_neutral)
# print(k1_inv_expr_fold)


# one-parameter
x_val = np.linspace(0.05, 0.95, 1001)
y_val = np.zeros(len(x_val))
k2_val = np.zeros(len(x_val))

a11_array = []
a22_array = []

trace = np.zeros(len(x_val))
det = np.zeros(len(x_val))
di = np.zeros(len(x_val))
x_sn, y_sn, k2_sn = [], [], []
x_s1, y_s1, k2_s1 = [], [], []
x_s2, y_s2, k2_s2 = [], [], []
x_det, y_det, k2_det = [], [], []
x_di, y_di, k2_di = [], [], []
for i in range(len(x_val)):
    y_val[i] = float(y_expr.subs(
        {k3: k3_val, k3_inv: k3_inv_val,
         x: float(x_val[i])}))

    k2_val[i] = float(k_expr.subs({k1_inv: k1_inv_val, k1: k1_val, k3: k3_val, k3_inv: k3_inv_val,
                                   x: float(x_val[i]), y: float(y_val[i])}))

    a11_val = float(a11.subs(
        {k1: k1_val, k1_inv: k1_inv_val, k2: k2_val[i], k3: k3_val, k3_inv: k3_inv_val,
         x: float(x_val[i]), y: float(y_val[i])}))
    a22_val = float(a22.subs(
        {k1: k1_val, k1_inv: k1_inv_val, k2: k2_val[i], k3: k3_val, k3_inv: k3_inv_val,
         x: float(x_val[i]), y: float(y_val[i])}))
    a12_val = float(a12.subs(
        {k1: k1_val, k1_inv: k1_inv_val, k2: k2_val[i], k3: k3_val, k3_inv: k3_inv_val,
         x: float(x_val[i]), y: float(y_val[i])}))
    a21_val = float(a21.subs(
        {k1: k1_val, k1_inv: k1_inv_val, k2: k2_val[i], k3: k3_val, k3_inv: k3_inv_val,
         x: float(x_val[i]), y: float(y_val[i])}))
    trace[i] = a11_val + a22_val
    det[i] = a11_val * a22_val - a12_val * a21_val
    # di[i] = trace[i] ** 2 - 4 * det[i]
    a11_array.append(a11_val)
    a22_array.append(a22_val)
    if i != 0:
        if trace[i] * trace[i - 1] <= 0:    # saddle-node
            x_sn.append(x_val[i])
            y_sn.append(y_val[i])
            k_exact = k2_val[i - 1] - trace[i - 1] * (k2_val[i] - k2_val[i - 1]) / (trace[i] - trace[i - 1])
            # k2_sn.append(k2_val[i])
            k2_sn.append(k_exact)
        if det[i] * det[i - 1] <= 0:    # adronov-hopf
            x_det.append(x_val[i])
            y_det.append(y_val[i])
            k_exact = k2_val[i - 1] - det[i - 1] * (k2_val[i] - k2_val[i - 1]) / (det[i] - det[i - 1])
            k2_det.append(k_exact)
            # k2_det.append(k2_val[i])
        # if di[i] * di[i - 1] <= 0:
        #     x_di.append(x_val[i])
        #     y_di.append(y_val[i])
        #     k2_di.append(k2_val[i])
        if a11_array[i] * a11_array[i - 1] <= 0:    # turing
            x_s1.append(x_val[i])
            y_s1.append(y_val[i])
            k2_s1.append(k2_val[i])


print("S-N")
print(k2_det)
print("A-H")
print(k2_sn)
fig = plt.figure(figsize=(10, 10))

plt.plot(k2_val, x_val, color='black', label='x')
plt.plot(k2_val, y_val, color='black', linestyle='--', label='y')

plt.plot(k2_det, y_det, 'ro', label='S-N')
plt.plot(k2_det, x_det, 'go', label='S-N')

plt.plot(k2_sn,  x_sn, '*', markersize=15, color='green', label='A-H')
plt.plot(k2_sn,  y_sn, '*', markersize=15, color='red', label='A-H')
plt.xlim(left=0.0)
plt.xlabel('k2')
plt.ylabel('x, y')
plt.legend()
plt.savefig('bifurcation_points_k1_002.png')
#plt.show()
#plt.plot([0, 0], [1, 0])
#plt.xlim(left=4.0)
# plt.ylim(0, 1)
# plt.xlim(left=0.001)

# turing
fig = plt.figure(figsize=(10, 10))
plt.plot(k2_val, x_val, color='black', label='x')
plt.plot(k2_val, y_val, color='black', linestyle='--', label='y')
plt.plot(k2_sn,  x_sn, '*', markersize=15, color='green', label='A-H')
plt.plot(k2_sn,  y_sn, '*', markersize=15, color='red', label='A-H')

plt.plot(k2_s1, y_s1, 'go', label='turing')
plt.plot(k2_s1, x_s1, 'go', label='turing')
plt.xlim(0.0)
plt.xlabel('k2')
plt.ylabel('x, y')
plt.legend()
plt.savefig('turing.png')

# coeffs
fig = plt.figure(figsize=(10, 10))
plt.plot(k2_val, a11_array, color='black', label='a11')
plt.plot(k2_val, a22_array, color='black', linestyle='--', label='a22')
plt.plot(k2_val, trace, color='green', linestyle='-.', label='trace')
plt.plot(k2_val, np.zeros(len(k2_val)), color='red', linestyle='-.', label='zero')
plt.xlim(0.0)
# plt.ylim(0.0)
plt.xlabel('k2')
plt.ylabel('x, y')
plt.legend()
plt.savefig('coeffs.png')

# two-parameter
x_val = np.linspace(0.05, 0.95, 1001)
y_val = np.zeros(len(x_val))
k2_val_fold = np.zeros(len(x_val))
k1_inv_val_fold = np.zeros(len(x_val))
for i in range(len(x_val)):
    try:
        k1_inv_val_fold[i] = float(k1_inv_expr_fold.subs({k1: k1_val, k3: k3_val, k3_inv: k3_inv_val, x: float(x_val[i])}))
        y_val[i] = float(y_expr.subs({k3_inv: k3_inv_val, k3: k3_val, x: float(x_val[i])}))
        k2_val_fold[i] = float(k_expr.subs({k1_inv: k1_inv_val_fold[i], k1: k1_val, k3_inv: k3_inv_val, k3: k3_val, x: float(x_val[i]), y: float(y_val[i])}))
    except TypeError:
        print(i)
        break
y_val = np.zeros(len(x_val))
k2_val_neutral = np.zeros(len(x_val))
k1_inv_val_neutral = np.zeros(len(x_val))
for i in range(len(x_val)):
    try:
        k1_inv_val_neutral[i] = float(k1_inv_expr_neutral.subs({k1: k1_val, k3: k3_val, k3_inv: k3_inv_val, x: float(x_val[i])}))
        # y_val[i] = float(y_expr.subs({k3_inv: k3_inv_val, k3: k3_val, x: float(x_val[i])}))
        k2_val_neutral[i] = float(k_expr.subs({k1_inv: k1_inv_val_fold[i], k1: k1_val, k3_inv: k3_inv_val, k3: k3_val, x: float(x_val[i]), y: float(y_val[i])}))
    except TypeError:
        print(i)
        break

fig1 = plt.figure(figsize=(10, 10))
plt.plot(k1_inv_val_fold, k2_val_fold, color='black', label='mult')
plt.plot(k1_inv_val_neutral, k2_val_neutral, color='black', linestyle='--', label='neutral')
# plt.plot([k1_inv_val, k1_inv_val], k2_det, 'go', label='S-N')
# plt.plot([k1_inv_val, k1_inv_val],  k2_sn, '*', markersize=15, color='green', label='A-H')
plt.legend()
plt.xlim(left=0.0, right=0.02)
plt.ylim(top=1.6, bottom=0.6)
plt.xlabel('k1_inv')
plt.ylabel('k2')
plt.savefig('two_parameter.png')
# plt.show()

# numerical solution
k1_inv_val = 0.01
k2_val_numerical = 1.0
initial = [0.2, 0.3]
dt = 0.1
t_max = 4000
t = np.linspace(0, t_max, int(t_max / dt))
sol = odeint(equation, initial, t)

fig = plt.figure(figsize=(20, 10))
plt.plot(t, sol[:, 0], color='black', label='x')
plt.plot(t, sol[:, 1], color='black', linestyle='--', label='y')
plt.legend()
plt.xlabel('t')
plt.savefig('numerical.png')

additional_solutions = []
additional_initial = [(0.5, 0.15), (0.5, 0.4)]
for init in additional_initial:
    sol_add = odeint(equation, init, t)
    additional_solutions.append(sol_add)

sol_end = odeint(equation, [sol[len(t) - 1, 0], sol[len(t) - 1, 1]], t)
fig = plt.figure(figsize=(10, 10))
plt.plot(sol_end[:, 0], sol_end[:, 1], linewidth = 2, color='black', label='cycle')
plt.plot(sol[:, 0], sol[:, 1], color='red',  linewidth = 0.7, linestyle='--', label='solution')
for i, sol_add in enumerate(additional_solutions):
    plt.plot(sol_add[:, 0], sol_add[:, 1], linewidth=0.5, label='additional solution {}'.format(i + 1))
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('cycle.png')
# plt.show()
