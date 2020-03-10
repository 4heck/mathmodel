import math

import eiler
import matplotlib.pyplot as plt
import numpy as np
import rungekutt
from scipy.integrate import odeint


def model(x, t, params):
    dxdt = params[0] * (params[1] - x)
    return dxdt


def x_t(t, params, t0):
    return params[1] * (1 - math.e ** (-params[0] * (t - t0)))

# количество насыщенного раствора
P = 420

# масса вещества
x0 = 0

# начальное время
t_min = 0

# конечное время
t_max = 2
k = 5

params = [k, P]

# количество чисел
step_count = 20
t = np.linspace(t_min, t_max, step_count)

y = odeint(model, x0, t, args=(params, ))
y1 = eiler.diffur_steps_params(model, params, t_min, x0, t_max, step_count)
y2 = rungekutt.diffur_steps_params(model, params, t_min, x0, t_max, step_count)
fun_values = []

for t_i in t:
    fun_values.append(x_t(t_i, params, t_min))

plt.step(t, y, 'b-', linewidth=1, label='Численный метод')
# plt.step(t, y1, 'g:', linewidth=2, label='Метод Эйлера')
plt.step(t, y2, 'y-', linewidth=2, label='Метод Рунге-Кутты')
plt.plot(t, fun_values, 'r--', linewidth=2, label='Конечная функция')

plt.legend()
plt.xlabel('Время')
plt.ylabel('x(t)')
plt.show()
